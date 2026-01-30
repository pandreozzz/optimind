import os
from time import time
import gc
from dask.distributed import wait,progress
from collections import namedtuple

import numpy as np
import xarray as xr

from ..utils.memory import trim_memory
from ..main.config import CONFIGDICT, TMPFLDDIR, ERA5MLFILESIGN, ERA5SFCFILESIGN, ERA5TENDFILESIGN, OPENDS_ZARR_KWARGS
from ..launchers.launch_tuning import LOGDIC

from .masks import get_cos_sza_mask, get_localtime_mask
from .levels import populate_mlfields

from ..tools.aerosol import compute_ccn_ifs

from ..tools.constants import GCNST, CPAIR

def get_ifs_fields(year : int) -> xr.Dataset:
    """
    Load IFS/ERA5 meteorological fields (model levels + surface).
    
    Opens ERA5 reanalysis data on model levels and surface, handling both
    netcdf4 and zarr formats based on configuration.
    
    Args:
        year: int - Year of data to load
    
    Returns:
        xr.Dataset: ERA5 fields on model levels with surface fields
    """
    print("Opening files...", flush=True)
        #ifs_fields = xr.merge(
        #    [xr.open_dataset(os.path.join(TMPFLDDIR, f"era5_{year:4d}_r30_{CONFIGDICT['hourly']}_{typ}.nc"))
        #     for typ in ["ml_sel", "sfc"]]).sel(time=timeslice).astype(np.float32)
    opends_kwargs = OPENDS_ZARR_KWARGS if CONFIGDICT["use_zarr"] else dict(engine="netcdf4")
    dataext = "_zarr" if CONFIGDICT["use_zarr"] else ".nc"
    ifs_fields = xr.open_dataset(filename_or_obj=os.path.join(
        TMPFLDDIR,
        f"era5_{year:4d}_{CONFIGDICT['gridspec']}_{CONFIGDICT['hourly']}_{ERA5MLFILESIGN}{dataext}"
        ), **opends_kwargs) # pyright: ignore[reportArgumentType]

    additional_fields_list = []

    # OPTIONAL
    try:
        additional_fields_list.append(xr.open_dataset(
            filename_or_obj=os.path.join(
                TMPFLDDIR,
                f"era5_{year:4d}_{CONFIGDICT['gridspec']}_{CONFIGDICT['hourly']}_{ERA5TENDFILESIGN}{dataext}"
                ), **opends_kwargs)) # pyright: ignore[reportArgumentType]
    except Exception as exc:
        print(f"Could not open tendencies file: {exc} - ignoring")

    # REQUIRED
    additional_fields_list.append(xr.open_dataset(
        filename_or_obj=os.path.join(
            TMPFLDDIR,
            f"era5_{year:4d}_{CONFIGDICT['gridspec']}_{CONFIGDICT['hourly']}_{ERA5SFCFILESIGN}{dataext}"
            ), **opends_kwargs)) #type: ignore

    for add_ds in additional_fields_list:
        for var in add_ds.variables:
            if var not in ifs_fields:
                if var == "lsm":
                    print("Squeezing lsm...")
                    add_ds[var] = add_ds[var].isel(**{dim: 0 for dim in add_ds[var].dims if dim not in ["lon", "lat"]}, drop=True)

            ifs_fields[var] = add_ds[var]
    print("done.", flush=True)

    return ifs_fields

def get_wstar(blh, tsurf, q_flx, heat_flx) -> xr.DataArray:
    """
     Calculate convective velocity scale (w*) from boundary layer parameters.
    
    Args:
        blh: Boundary layer height
        tsurf: Surface temperature
        q_flx: Specific humidity flux (Kg/m2/s)
        heat_flx: Heat flux (J/m2/s)
    
    Returns:
        xr.DataArray: Convective velocity scale (w*)
    """
    lat_vap = 2.46*1.e6 #J/Kg
    rho_air = 1. # Kg/m3

    wst1 = (GCNST/tsurf*blh*(-q_flx * lat_vap / (CPAIR*rho_air)))
    wst2 = (GCNST/tsurf*blh*(-heat_flx / (CPAIR*rho_air)))
    return (wst1+wst2).clip(min=0)**0.333

def get_cum_tau_c(ds, prior_ws=1, cc_thresh = 0.1, maxtauc : bool = False) -> xr.DataArray:
    """
    Calculate cumulative cloud optical depth from cloud cover and water content.
    
    Args:
        ds: xarray Dataset with cloud variables
        prior_ws: Prior water cloud scale
        cc_thresh: Cloud cover threshold for filtering
        maxtauc: If True, use in-cloud optical depth instead of grid-box average
    
    Returns:
        xr.DataArray: Cumulative optical depth (or maximum if maxtauc=True)
    """

    maxtauc_factor = 1 if maxtauc else ds["cc"]

    ifs_ccns = compute_ccn_ifs(ws=xr.DataArray(data=prior_ws), lsm=ds["lsm"]).astype(np.float32)

    return (np.float32((9/2*np.pi*0.8)**0.333)*\
            (ds["clwc"]/ds["cc"].where(ds["cc"] > cc_thresh, np.inf))**np.float32(0.666)*\
            ifs_ccns**np.float32(0.333)*\
            ds["dp"]/(np.float32(9.8)*ds["rho_air"]**np.float32(0.333))*maxtauc_factor\
            ).cumsum(dim="lev")

def get_rel_hum(pres_ml, sphum_ml, temp_ml):
    """Calculate relative humidity from pressure, specific humidity, and temperature."""
    return np.float32(0.00263)*pres_ml*sphum_ml/(np.exp((np.float32(17.67)*(temp_ml-np.float32(273.16)))/(temp_ml-np.float32(29.65))))

CloudyLevel = namedtuple("cloudy_level", ["representative_cloudy_level", "is_cloud_thick_enough"])
def get_cloudy_level(cum_tau_c, min_tot_tau_c, min_top_tau_c):
    """
    Find the optically representative cloud layer for cloud properties.
    
    Identifies the cloud layer that represents optical cloud depth for use in
    cloud phase and temperature calculations. Validates minimum cloud thickness.
    
    Args:
        cum_tau_c: xarray DataArray - Cumulative cloud optical depth from TOA
        min_tot_tau_c: float - Minimum total cloud optical depth threshold
        min_top_tau_c: float - Maximum optical depth to consider as cloud top
    
    Returns:
        CloudyLevel namedtuple with:
            - representative_cloudy_level: Index of representative cloud layer
            - is_cloud_thick_enough: Boolean mask (True if tau_c > min_tot_tau_c)
    """
    tot_tau_c = cum_tau_c.isel(lev=-1).fillna(0)
    is_cloud_thick_enough = (tot_tau_c > min_tot_tau_c).compute()
    is_cloud_layer_deep_enough = (cum_tau_c.fillna(0) >= tot_tau_c.clip(max=min_top_tau_c)) & is_cloud_thick_enough
    representative_cloudy_level = cum_tau_c.where(is_cloud_layer_deep_enough).fillna(np.inf).argmin(
        dim="lev", skipna=True).expand_dims(lev=[-1], axis=-1).astype(np.int16)
    return CloudyLevel(representative_cloudy_level.compute(), is_cloud_thick_enough.compute())


def get_cloud_base_level(cum_tau_c, frac_bot_tau_c, nlevelsbelowcloudbase: int = 0):
    """
    Find cloud base level accounting for below-cloud aerosols.
    
    Identifies cloud base by finding where cumulative optical depth exceeds
    a fraction of total depth, optionally moving down by specified levels.
    
    Args:
        cum_tau_c: xarray DataArray - Cumulative cloud optical depth from TOA
        frac_bot_tau_c: float - Fraction of total tau_c (0-1) defining cloud base
        nlevelsbelowcloudbase: int - Levels below cloud base to use for aerosol (default: 0)
    
    Returns:
        xarray DataArray: Cloud base level indices (int16)
    """
    tot_tau_c = cum_tau_c.isel(lev=-1).fillna(0)
    cloud_base_level = cum_tau_c.where(
        (cum_tau_c.fillna(0) > frac_bot_tau_c * tot_tau_c),
        np.inf).argmin(dim="lev", skipna=True) + nlevelsbelowcloudbase
    print("got cloud_base level", flush=True)
    return cloud_base_level.clip(0, len(cum_tau_c.lev)-1).astype(np.int16)


def get_gros_aerolevel(cum_tau_c):
    """
    Find aerosol level using Grosvenor's cloud penetration correction.
    
    Computes penetration depth into cloud based on Grosvenor et al. method
    for correcting Nd penetration bias in satellite retrievals.
    
    Args:
        cum_tau_c: xarray DataArray - Cumulative cloud optical depth from TOA
    
    Returns:
        xarray DataArray: Aerosol level indices for cloud penetration correction
    """
    def grosv_tau_depth(tau_c, wl="2.1um"):
        b_coeffs = {
            "2.1um": [0.3216, 0.5754, -0.021, 3.931e-4, -3.174e-6],
            "3.7um": [0.6005, 0.4168, -0.03304, 1.099e-3, -1.281e-5]
        }
        return (b_coeffs[wl][0] + tau_c.clip(max=35)*b_coeffs[wl][1] +
                tau_c.clip(max=35)**2*b_coeffs[wl][2] +
                tau_c.clip(max=35)**3*b_coeffs[wl][3] +
                tau_c.clip(max=35)**4*b_coeffs[wl][4])

    tot_tau_c = cum_tau_c.isel(lev=-1).fillna(0)
    is_layer_deep_enough = (cum_tau_c.fillna(0) >= tot_tau_c.clip(max=grosv_tau_depth(tot_tau_c)))
    return cum_tau_c.where(is_layer_deep_enough, np.inf).argmin(
        dim="lev", skipna=True).expand_dims(lev=[-1], axis=-1).astype(np.int16)

def get_meteo_cloudy_slices(year : int, ifs_fields=None, cc_thresh : float = 0.8, t_thresh : float = 268, iwr_thresh : float = 1.e-2, prior_ws : float = 10, min_tot_tau_c : float = 4,
                            min_top_tau_c : float = 2.0, maxtauc : bool = False, use_grosvenor_tau_c : bool = False, frac_bot_tau_c : float = 0.95, thresh_valid_monthly : float = 0.1, latmin : float = -90, latmax : float = 90):

    #CLIENT.run(trim_memory)


    timeslice = slice(f"{year:4d}-01-01",f"{year:4d}-12-31")

    if ifs_fields is None:
        ifs_fields = get_ifs_fields(year=year)

    fld_list_3d = ["t", "q", "cc", "w",
                   "clwc", "ciwc", "crwc",
                   "avg_ttlwr", "avg_ttswr", "avg_ttpm"]
    fld_list_2d = ["tcc", "tclw", "tcrw", "tciw", "lsm", "10u", "10v", "2t", "sp", "blh", "ie", "ishf", "skt"]
    fld_list = fld_list_3d + fld_list_2d
    fld_list = [f for f in fld_list if f in ifs_fields.variables]
    new_fld_list = ["p", "rho_air", "dp"]
    levdefs_list = ["hyam", "hybm", "hyai", "hybi"]


    print("Setting up latitude and longitude selection", flush=True)
    latslice = slice(latmax,latmin)

    lonmin = min(CONFIGDICT["longitudes_minmax"])
    lonmax = max(CONFIGDICT["longitudes_minmax"])

    all_lons = ifs_fields['lon'].load()
    if lonmin < lonmax:
        lonsel = all_lons.sel(lon=slice(lonmin, lonmax))
    else:
        lonsel = xr.concat(
            [all_lons.where(all_lons >= lonmin, drop=True),
            all_lons.where(all_lons <= lonmax, drop=True)],
            dim="lon")

    print("Slicing data...", flush=True)
    this_ifs =  ifs_fields[["sp"]+fld_list+levdefs_list].transpose(..., "lat", "lon").sortby("lat", ascending=False).sel(lat=latslice, lon=lonsel)
    this_ifs = this_ifs.chunk({"time": np.ceil(len(this_ifs.time)/(32*CONFIGDICT["nprocs"]))})
    #print(this_ifs.__str__())

    print(f"Model data: {this_ifs.nbytes/(1024**3):.1f}GB.", flush=True)
    del ifs_fields

    gc.collect()
    trim_memory()
    #CLIENT.run(trim_memory)

    # Restrict model data on cos sza
    cos_sza_mask = get_cos_sza_mask(time=this_ifs.time, lat=this_ifs.lat, lon=this_ifs.lon)

    # Restrict model data on exact local time
    localtime_mask = get_localtime_mask(time=this_ifs.time, lon=this_ifs.lon)

    # 3D pressure, rho_air, level pressure difference
    print("Starting computations. Reading data from disk might take a while...", flush=True)
    time0 = time()
    this_ifs.persist()
    wait(this_ifs)
    time1 = time()
    print(f"Loaded. ({time1-time0:.2f}s)")
    populate_mlfields(this_ifs)
    print(f"Populated ml fields ({time()-time1:.2f}s)")
    gc.collect()
    trim_memory()

    #this_ifs = this_ifs[["p", "dp", "rho_air"]+fld_list].astype(np.float32).compute()

    #check this:
    #this_ifs["lsm"] = this_ifs["lsm"].isel(time=0, drop=True)

    print(f"cc threshold: {cc_thresh:.2f}, t threshold: {t_thresh-273.15:+.1f}C, iwr threshold: {iwr_thresh:.2f}")
    print(f"prior wind speed: {prior_ws:.1f} m/s", flush=True)
    LOGDIC["cc_thresh"] = cc_thresh
    LOGDIC["t_thresh"] = t_thresh
    LOGDIC["prior_ws"] = prior_ws
    LOGDIC["iwr_thresh"] = iwr_thresh

    #cum_tau_c = ((9/2*np.pi)**0.333*(this_ifs["clwc"]*this_ifs["rho_air"]/this_ifs["cc"].where(this_ifs["cc"] > cc_thresh)/1.e3)**0.666*\
    #(ct.tools.compute_ccn_ifs(ws=prior_ws, lsm=this_ifs["lsm"])*1.e6)**0.333*this_ifs["lev_height"]).cumsum(dim="lev")
    print("Computing cumulated cloud optical thickness for each column", flush=True)
    cum_tau_c = get_cum_tau_c(this_ifs, prior_ws=prior_ws,
                              cc_thresh=cc_thresh,
                              maxtauc=maxtauc).persist()
    wait(cum_tau_c)

    print(f"Minimum tau_c: {min_tot_tau_c:.1f}, threshold tau_c to identify cloud top: {min_top_tau_c:.1f}", flush=True)
    LOGDIC["min_tot_tau_c"] = min_tot_tau_c
    LOGDIC["min_top_tau_c"] = min_top_tau_c

    ThisCloudyLevel = get_cloudy_level(cum_tau_c=cum_tau_c, min_tot_tau_c=min_tot_tau_c, min_top_tau_c=min_top_tau_c)

    #return ThisCloudyLevel, this_ifs

    # Check temperature
    print("Filtering cloud top by temperature and phase...", flush=True)
    time0 = time()
    this_ifs_layer = this_ifs[fld_list+new_fld_list].isel(lev=ThisCloudyLevel.representative_cloudy_level.rename(lev="lev_aux"), drop=True).where(ThisCloudyLevel.is_cloud_thick_enough).drop_vars("lev_aux").rename_dims(lev_aux="lev").persist()
    wait(this_ifs_layer)
    print(f"Representative cloud level extracted. ({time()-time0:.2f}s)",
          flush=True)

    #CLIENT.run(trim_memory)
    # gc.collect()
    # trim_memory()

    this_ifs_layer["repr_cloud_lev"] = ThisCloudyLevel.representative_cloudy_level
    this_ifs_layer["tot_tau_c"] = cum_tau_c.isel(lev=-1)

    # Cloud top is warm and liquid
    is_warm_cloud_2d = (this_ifs_layer["t"] > t_thresh).compute()
    ctwc = this_ifs_layer["ciwc"] + this_ifs_layer["clwc"]
    is_liquid_cloud_2d = ((this_ifs_layer["ciwc"]/ctwc.where(ctwc>0, 1.)) < iwr_thresh).compute()

    # Do not delay these calculations
    is_warmliquid_cloud_2d = (is_warm_cloud_2d & is_liquid_cloud_2d).compute()
    print(f"Filtering only points more than {thresh_valid_monthly*100:.0f}% valid within month", flush=True)
    monthly_valid = is_warmliquid_cloud_2d.groupby(is_warmliquid_cloud_2d.time.dt.month).mean()
    is_enough_monthly_vals = monthly_valid >= thresh_valid_monthly
    for m in is_enough_monthly_vals.month:
        is_warmliquid_cloud_2d.loc[is_warmliquid_cloud_2d.time.dt.month == m] = \
                is_warmliquid_cloud_2d.loc[is_warmliquid_cloud_2d.time.dt.month == m] & \
                is_enough_monthly_vals.sel(month=m)

    total_mask = is_warmliquid_cloud_2d & (cos_sza_mask & localtime_mask)
    this_ifs_layer = this_ifs_layer.where(total_mask)
    this_ifs_layer["is_warm_cloud_2d"] = is_warm_cloud_2d
    this_ifs_layer["is_liquid_cloud_2d"] = is_liquid_cloud_2d
    this_ifs_layer["is_warmliquid_cloud_2d"] = is_warmliquid_cloud_2d
    this_ifs_layer["cos_sza_mask"] = cos_sza_mask
    this_ifs_layer["localtime_mask"] = localtime_mask
    this_ifs_layer["total_mask"] = total_mask
    del is_warmliquid_cloud_2d, cos_sza_mask, localtime_mask

    if CONFIGDICT["wspeed_type"] > 0:
        print("Getting w at cloud base...")
        cloud_base_level = get_cloud_base_level(cum_tau_c.where(ThisCloudyLevel.is_cloud_thick_enough, 0.),
                                                frac_bot_tau_c=frac_bot_tau_c,
                                                nlevelsbelowcloudbase=0).compute()
        # Convert Pa/s to m/s
        this_ifs_layer["w_clbase"] = (this_ifs["w"].rename(lev="levaux").isel(levaux=cloud_base_level,
                                                 drop=True).where(total_mask).compute() / (this_ifs_layer["rho_air"]*9.8)).assign_attrs(units="m/s")
        print(this_ifs_layer["w_clbase"].__str__())

    if CONFIGDICT["wspeed_type"] > 2:
        print("Computing Deardorff velocity scale wstar", flush=True)
        this_ifs_layer["wstar"] = get_wstar(this_ifs_layer["blh"], this_ifs_layer["2t"],
                                            this_ifs_layer["ie"], this_ifs_layer["ishf"]).compute()

    match CONFIGDICT["wspeed_type"]:
        case(1):
            this_ifs_layer["w_mean_raw"] = this_ifs_layer["w_clbase"]
        case(2):
            this_ifs_layer["w_mean_raw"] = this_ifs_layer["w_clbase"] + this_ifs_layer["avg_ttlwr"]*CPAIR/GCNST
        case(3):
            this_ifs_layer["w_mean_raw"] = this_ifs_layer["w_clbase"]
            this_ifs_layer["w_prime_raw"] = CONFIGDICT["deardorff_scale"]*this_ifs_layer["wstar"]
        case(4):
            this_ifs_layer["w_mean_raw"] = this_ifs_layer["w_clbase"] + this_ifs_layer["avg_ttlwr"]*CPAIR/GCNST
            this_ifs_layer["w_prime_raw"] = CONFIGDICT["deardorff_scale"]*this_ifs_layer["wstar"]
    # Bind to minimum
    for var in ["w_mean", "w_prime"]:
        if f"{var}_raw" in this_ifs_layer:
            minval = CONFIGDICT[f"{var}_min"]
            belowmin = this_ifs_layer[f"{var}_raw"] < minval
            occurrence = belowmin.mean().values.item()
            print(f"Setting minimum val of {var} to {minval}m/s,\naffects {occurrence*100:.2f}% of values")
            this_ifs_layer[var] = this_ifs_layer[f"{var}_raw"].clip(min=minval)

    # Where to pick mixing ratios?
    if CONFIGDICT["aerofromclimatology"] and (CONFIGDICT["fixedaeromodellevel"] is not None):
        print(f"Collecting aerosol fields at fixed level {CONFIGDICT['fixedaeromodellevel']}", flush=True)
        this_ifs_fixedlevel = this_ifs[["p", "t", "rho_air"]].sel(lev=CONFIGDICT["fixedaeromodellevel"], drop=True).where(total_mask)


    elif CONFIGDICT["nlevelsbelowcloudbase"] is not None:
        print(f"Collecting aerosol fields {CONFIGDICT['nlevelsbelowcloudbase']} levels below cloud base", flush=True)
        cloud_base_level = get_cloud_base_level(
            cum_tau_c.where(ThisCloudyLevel.is_cloud_thick_enough, 0.),
            frac_bot_tau_c=frac_bot_tau_c,
            nlevelsbelowcloudbase=CONFIGDICT["nlevelsbelowcloudbase"]).compute()
        
        this_ifs_fixedlevel = this_ifs[["p", "t", "q", "rho_air"]].rename(lev="levaux").isel(levaux=cloud_base_level, drop=True).where(total_mask)

        this_ifs_fixedlevel["r"] = get_rel_hum(this_ifs_fixedlevel["p"], this_ifs_fixedlevel["q"],
                                               this_ifs_fixedlevel["t"]
                                               ).assign_attrs(
                                                   standad_name="relative_humidity",
                                                   long_name="Relative humidity",
                                                   units="1")
        this_ifs_fixedlevel["base_cloud_lev"] = cloud_base_level
        del cloud_base_level

    elif use_grosvenor_tau_c:
        print(f"Collecting aerosol fields using at tau_c correcting Nd penetration bias", flush=True)
        representative_aero_level = get_gros_aerolevel(cum_tau_c=cum_tau_c.where(ThisCloudyLevel.is_cloud_thick_enough, 0.))
        this_ifs_fixedlevel = this_ifs[["p", "t", "rho_air"]].rename(lev="levaux").isel(levaux=representative_aero_level, drop=True).where(total_mask)
        this_ifs_fixedlevel["repr_aero_lev"] = representative_aero_level.astype(np.int16)
        del representative_aero_level

    else:
        this_ifs_fixedlevel = None

    if this_ifs_fixedlevel:
        this_ifs_fixedlevel = this_ifs_fixedlevel.compute()
    this_ifs_layer = this_ifs_layer.compute()

    print("Invoking garbage collector...")
    del this_ifs
    gc.collect()
    trim_memory()

    return this_ifs_layer, this_ifs_fixedlevel