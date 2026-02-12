"""Handling of fields containing aerosols"""
import os
from glob import glob
import numpy as np
import xarray as xr

from ..main.config import CONFIGDICT, OPENDS_ZARR_KWARGS, TMPFLDDIR
from .stage import aero_pl_namelike

AERORENAMEDIC = {
    "aermr01" : "Sea_Salt_bin1",
    "aermr02" : "Sea_Salt_bin2",
    "aermr03" : "Sea_Salt_bin3",
    "aermr04" : "Mineral_Dust_bin1",
    "aermr05" : "Mineral_Dust_bin2",
    "aermr06" : "Mineral_Dust_bin3",
    "aermr07" : "Organic_Matter_hydrophilic",
    "aermr09" : "Black_Carbon_hydrophilic",
    "aermr11" : "Sulfates",
    "aermr16" : "Nitrate_fine",
    "aermr17" : "Nitrate_coarse",
    "aermr18" : "Ammonium",
    "aermr19" : "Biogenic_Secondary_Organic",
    "aermr20" : "Anthropogenic_Secondary_Organic"
}

def get_aero_fields(year, timesel = None, latmin = None, latmax = None) -> xr.Dataset:
    """
    Load aerosol mass concentration fields with optimizations for large zarr archives.
    
    Automatically handles both
    netcdf4 and zarr formats based on configuration.
    
    Args:
        year: int - Year of data to load
        timesel: slice or array - Time selection (pandas-compatible)
        latmin: float - Minimum latitude (default: from CONFIGDICT)
        latmax: float - Maximum latitude (default: from CONFIGDICT)
    
    Returns:
        xr.Dataset: Aerosol pressure-level and surface fields for specified domain/time
    
    Note:
        Uses lazy evaluation and early spatial selection before chunking to
        reduce memory requirements for large datasets.
    """
    if latmin is None:
        latmin = min(CONFIGDICT["latitudes_minmax"])
    if latmax is None:
        latmax = max(CONFIGDICT["latitudes_minmax"])
 
    opends_kwargs = OPENDS_ZARR_KWARGS if CONFIGDICT["use_zarr"] else {"engine": "netcdf4"}
    dataext = "_zarr" if CONFIGDICT["use_zarr"] else ".nc"

    cams_prog_files = glob(
        os.path.join(TMPFLDDIR, aero_pl_namelike(year, CONFIGDICT["gridspec"])+dataext)
        )

    # Neclect surface fields
    #cams_prog_files = cams_prog_files+glob(os.path.join(os.environ["TMPDIR"], "fields", aero_sfc_namelike(year, CONFIGDICT["gridspec"]+dataext)))

    this_prog = xr.open_dataset(cams_prog_files[0], **opends_kwargs) # type: ignore

    # Neglect surface fields
    # this_prog_sfc_tmp = xr.open_dataset(cams_prog_files[1])
    # for var in this_prog_sfc_tmp.variables:
    #     if var not in this_prog:
    #         this_prog[var] = this_prog_sfc_tmp[var]
    # del this_prog_sfc_tmp

    this_prog = this_prog.rename(AERORENAMEDIC)
    if timesel is not None:
        this_prog = this_prog.sel(time=timesel)
    #this_prog = this_prog.chunk({"time":np.ceil(len(this_prog.time)/(32*CONFIGDICT["nprocs"]))})
    print("Aerosol fields chunked successfully.")

    flds_dtype = this_prog["t"].dtype
    print(f"Fields type: {flds_dtype}", flush=True)

    this_prog = this_prog.assign_coords(
        lev=xr.DataArray(data=np.arange(1,len(this_prog["plev"])+1), dims=["plev"])
    ).swap_dims(plev="lev").transpose(..., "lat", "lon").sortby("lat", ascending=False)

    latslice = slice(latmax,latmin)

    this_prog = this_prog.sel(lat=latslice)

    tmp_plev = this_prog["plev"]
    this_prog = this_prog.drop_vars("plev")
    this_prog["plev"] = tmp_plev

    this_prog = this_prog.rename(plev="pressure")

    return this_prog

def get_aero_fromclim(year, latmin = None, latmax = None) -> xr.Dataset:
    """
    Load aerosol mass concentration fields from climatology. 
    Args:
        year: int - Year of data to load
        timesel: slice or array - Time selection (pandas-compatible)
        latmin: float - Minimum latitude (default: from CONFIGDICT)
        latmax: float - Maximum latitude (default: from CONFIGDICT)
    
    Returns:
        xr.Dataset: Aerosol pressure-level and surface fields for specified domain/time
    
    """
    if latmin is None:
        latmin = min(CONFIGDICT["latitudes_minmax"])
    if latmax is None:
        latmax = max(CONFIGDICT["latitudes_minmax"])

    clim_dset = xr.open_dataset(f"{os.environ['TMPDIR']}/fields/aerosol_cams_climatology.nc").transpose(..., "lat", "lon").sortby("lat", ascending=False)

    clim_dset = clim_dset.sel(lat=slice(latmax,latmin))

    if "pressure" not in clim_dset:
        raise ValueError("Climatology file does not contain mandatory field pressure!")
    if "lev" not in clim_dset:
        raise ValueError("Vertical dimension must be 'lev', but name not found.")

    if len(clim_dset["lev"] > 30):
        clim_dset = clim_dset.sel(lev=clim_dset["lev"][-30:])

    if "epoch" in clim_dset:
        min_epoch = clim_dset["epoch"].min().values
        max_epoch = clim_dset["epoch"].max().values

        if year <= min_epoch:
            clim_dset = clim_dset.sel(epoch=min_epoch).assign_coords(epoch=year)
        elif year >= max_epoch:
            clim_dset = clim_dset.sel(epoch=max_epoch).assign_coords(epoch=year)
        else:
            clim_dset = clim_dset.interp(
                epoch=year,
                method="linear").assign_coords(epoch=year)

    return clim_dset