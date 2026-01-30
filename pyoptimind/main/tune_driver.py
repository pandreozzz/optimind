"""Main orchestration logic for LUT tuning workflow."""
import os
import logging
import time
import gc
import numpy as np
import xarray as xr

import pickle

from . import config

from ..launchers.launch_tuning import LOGDIC
from .config import CONFIGDICT
from ..utils.memory import print_memory_status, trim_memory
from ..utils.spillover import print_spill_status
from ..lut.setup import setup_pyrcel_lut, get_actual_lut_recipes
from ..fields.stage import copy_all_files

from ..fields.clouds import get_meteo_cloudy_slices
from ..fields.aerosols import get_aero_fields, get_aero_fromclim
from ..fields.ccn import compute_ccn_species

from ..tools.aerosol import compute_ccn_ifs, get_nccn_over_mcon_from_speclist
from ..tools.aerinterp import get_interpolated_ccn, interpolate_aero

from ..fields.modis import get_modis_data, get_modis_errors

from ..tools.tuner import compute_err_func, tuning_loop

from ..tools.lut import get_lutaero_from_r0

logger = logging.getLogger(__name__)


def get_tuning_params(actual_recipe):
    print("Getting tuning params...")

    species_to_tune = []
    ini_radii = []
    print(config.THISLUTAERO)
    for aero in config.THISLUTAERO:
        print(aero)
        if (aero.name in actual_recipe) and \
        not (CONFIGDICT["bindseasalt"] and aero.name == "seasalt2"):
            print(f"adding {aero} to species_to_tune")
            species_to_tune.append(aero.name)
            ini_radii.append(aero.median.item())
    if CONFIGDICT["bindseasalt"]:
        if CONFIGDICT["ss_coarsetofine_ratio"] is None:
            print("Setting sea salt coarse-to-fine ratio to 10")
            CONFIGDICT["ss_coarsetofine_ratio"] = 10
    else:
        CONFIGDICT["ss_coarsetofine_ratio"] = None

    if CONFIGDICT["firstguess_radii"]:
        firstguess_radii = CONFIGDICT["firstguess_radii"]
        if (len(firstguess_radii) != len(ini_radii)):
            raise ValueError("Invalid length for firstguess_radii vector!! "+\
                             f"Expected type {ini_radii}, got {firstguess_radii}")
        else:
            firstguess_radii = [f if f is not None else i
                                for f,i in zip(firstguess_radii, ini_radii)]
    else:
        print("setting first-guess radii same as initial radii")
        firstguess_radii = ini_radii

    return species_to_tune, ini_radii, firstguess_radii

def run_tuning_year(year: int, config_path: str, logdir_path: str):
    """
    Execute full year tuning workflow.
    
    Args:
        year: Year to process
        config_path: Path to configuration JSON
        logdir_path: Path to logs directory
    """
    logger.info(f"Starting LUT tuning for year {year}")
    
    # Initialize

    logger.info(f"Setting up pyrcel LUT from {CONFIGDICT['pyrcellutpath']}")
    setup_pyrcel_lut(CONFIGDICT["pyrcellutpath"])
    
    print_memory_status("After LUT setup")
    print_spill_status("After LUT setup")
    
    # Stage data files
    logger.info(f"Staging data files for {year}")
    copy_all_files(year)
    
    print("Loading MODIS Nd data...", flush=True)
    this_modis_nd13, this_modis_nd13_errors = get_modis_errors(get_modis_data(year))
    this_modis_nd13 = this_modis_nd13.rename("modis_nd13")
    this_modis_nd13_errors = this_modis_nd13_errors.rename("modis_nd13_errors")
    print(this_modis_nd13.__str__())
    print(this_modis_nd13_errors.__str__(), flush=True)
    
    start_time = time.time()
    print("Getting era5 cloudy points...", flush=True)
    this_ifs, this_ifs_fixedlevel = get_meteo_cloudy_slices(
        year,
        cc_thresh=CONFIGDICT["cldetect_cc_threshold"],
        t_thresh=CONFIGDICT["cldetect_t_threshold"],
        iwr_thresh=CONFIGDICT["cldetect_iwr_threshold"],
        use_grosvenor_tau_c=CONFIGDICT["grosvenor_tau_c_correction"],
        thresh_valid_monthly=CONFIGDICT["cldetect_thresh_valid_monthly"],
        latmin =min(CONFIGDICT["latitudes_minmax"]),
        latmax=max(CONFIGDICT["latitudes_minmax"])
    )
    frac_cloudy_pts = (~np.isnan(this_ifs['p'])).mean().values
    print(f"Fraction of cloudy points: {frac_cloudy_pts*100:.02f}%")
    print(this_ifs.__str__(), flush=True)

    if (this_ifs_fixedlevel is None) and config.SOME_AEROS_OUT_OF_CLOUD:
        raise ValueError("Got no fixedlevel fields while some aeros should be picked out of cloud")
    print("Getting aerosol-related fields...", flush=True)
    gc.collect()
    trim_memory()
    if CONFIGDICT["aerofromclimatology"]:
        this_aero = get_aero_fromclim(year=year)
    else:
        this_aero = get_aero_fields(year, timesel=this_ifs.time.values)

    # Align recipes and aerosol fields
    actual_pyrcellut, actual_recipe, needed_aeros = get_actual_lut_recipes(list(this_aero.variables))
    
    #list(actual_recipe.keys())
    species_to_tune, ini_radii, firstguess_radii = get_tuning_params(actual_recipe)

    print(f"Pyrcel LUT: {CONFIGDICT['pyrcellutpath']}")
    print(f"actual_recipe: {actual_recipe}")
    print(f"species_to_tune: {species_to_tune}")
    print(f"ini_radii = {ini_radii}")
    print(f"firstguess_radii = {firstguess_radii}")

    # Load necessary fields
    print("Loading needed aero fields on memory...", flush=True)
    this_aero = this_aero[needed_aeros]
    print("done!", flush=True)

    if CONFIGDICT["global_mass_scaler"]:
        for spec, factor in CONFIGDICT["global_mass_scaler"].items():
            if spec in this_aero:
                print(f"Scaling {spec} by factor {factor}")
                this_aero[spec] = this_aero[spec]*factor
            else:
                print(f"WARNING!! Could not scale {spec}: not found in aerosol fields!")
    if config.SSRH80:
        for spec in needed_aeros:
            if spec.startswith("Sea_Salt"):
                print(f"Scaling {spec} by 1/4.3 to dry mass")
                this_aero[spec] = this_aero[spec]/4.3

    print(this_aero.__str__())
    print("Interpolating CCN species to era5 level selection", flush=True)
    tgt_rhoa = this_ifs["rho_air"] #if this_ifs_fixedlevel is None else this_ifs_fixedlevel["rho_air"] PICK AERO MMR NOT ACTUAL MASS!!
    this_ccn_mmr = compute_ccn_species(this_aero, actual_recipe).compute()
    #print(this_ccn_mmr.__str__(), flush=True)
    this_ccn_mmr["pressure"] = this_aero["pressure"].compute()

    this_ccn_mcon = get_interpolated_ccn(this_ccn_mmr, this_ifs,
                                         this_ifs_fixedlevel, actual_recipe)*tgt_rhoa

    if CONFIGDICT["tune_rain_dispersion"]:
        rainratio_tmp = this_ifs["crwc"]/this_ifs["clwc"] # At cloud "top"
        this_ccn_mcon["rainratio"] = rainratio_tmp.transpose(*[d for d in this_ccn_mcon.dims if d in rainratio_tmp])

    print(this_ccn_mcon.__str__())

    # Interpolated aerosols
    print("Computing monthly mean aerosols for diagnostics...", flush=True)
    if this_ifs_fixedlevel is not None:
        tgt_pres = this_ifs_fixedlevel["p"]
    else:
        tgt_pres = this_ifs["p"]

    this_aero_mcon_monthly = xr.Dataset(
        data_vars={v: interpolate_aero(
            this_aero[["pressure", v]], tgt_pres,
            aero_timeinterp=CONFIGDICT["aerofromclimatology"]
        )[v]*tgt_rhoa for v in needed_aeros if v != "pressure"})


    this_aero_mcon_monthly = this_aero_mcon_monthly.groupby(
        this_aero_mcon_monthly.time.dt.month
    ).mean()
    print("done!", flush=True)

    del this_ccn_mmr
    del tgt_pres



    monthly_weights = this_ifs["is_warmliquid_cloud_2d"].groupby(this_ifs.time.dt.month).sum() if CONFIGDICT["weightbycloudpresence"] else None

    # Performance of wind-parametrized ccns
    this_ifs_nd13 = (compute_ccn_ifs(
        ws=xr.DataArray(np.sqrt(this_ifs["10u"]**2+this_ifs["10v"]**2)),
        lsm=this_ifs["lsm"])**0.333).groupby(this_ifs["time"].dt.month).mean() # type: ignore

    this_ifs_nd13_pp = this_ifs_nd13/(1+this_ccn_mcon["rainratio"]) if CONFIGDICT["tune_rain_dispersion"] else this_ifs_nd13

    ifs_err = compute_err_func(this_ifs_nd13_pp,
                               this_modis_nd13,
                               this_modis_nd13_errors,
                               monthly_weights)
    print(f"IFS (wind-dependent ccns) errfun: {ifs_err:.3f}")
    # Tuning loop
    print("Launching tuning loop...", flush=True)

    # Populate w_mean and w_prime if used
    for var in ["w_mean", "w_prime"]:
        if var in this_ifs:
            print(f"Populating {var} for LUT extraction")
            this_ccn_mcon[var] = this_ifs[var]

    tuning_res, ini_err, delta_err, \
    ini_nd13, ini_mcon_scaler_fields, tun_nd13 =\
    tuning_loop(this_ccn_mcon, ini_radii, firstguess_radii,
                this_modis_nd13, this_modis_nd13_errors,
                actual_pyrcellut, actual_recipe,
                species_to_tune=species_to_tune,
                bind_seasalt_ratio=CONFIGDICT["ss_coarsetofine_ratio"],
                wspeed_type=CONFIGDICT["wspeed_type"],
                monthly_weights=monthly_weights,
                tune_rain_dispersion=CONFIGDICT["tune_rain_dispersion"]
               )

    tuning_res_radii = tuning_res.x[:-1] if CONFIGDICT["tune_rain_dispersion"] else tuning_res.x
    tuning_rain_disp_factor = tuning_res.x[-1] if CONFIGDICT["tune_rain_dispersion"] else None

    from datetime import datetime as dt
    date = f"{dt.now():%Y%m%d_%H%M%S}"

    logfile = os.path.join(logdir_path, f'tuning_res_{year}_{date}.pkl')
    global LOGDIC
    LOGDIC = {
        **LOGDIC,
        **{
            "config_path" : config_path,
            "this_ifs"  : this_ifs.groupby(this_ifs.time.dt.month).mean(),
            "this_ifs_fixedlevel": this_ifs_fixedlevel.groupby(this_ifs.time.dt.month).mean() if this_ifs_fixedlevel is not None else None,
            "this_aero_mcon_monthly" : this_aero_mcon_monthly,
            "this_dates" : this_ifs.time,
            "this_ccn_mcon_monthly" : this_ccn_mcon.groupby(this_ccn_mcon.time.dt.month).mean(),
            "prior_nccn_over_mcon" : get_nccn_over_mcon_from_speclist(
                get_lutaero_from_r0({name:r for name,r in zip(species_to_tune, ini_radii)},
                                    config.THISLUTAERO,
                                    CONFIGDICT["ss_coarsetofine_ratio"])),
            "tuned_nccn_over_mcon" : get_nccn_over_mcon_from_speclist(
                get_lutaero_from_r0({name:r for name,r in zip(species_to_tune, tuning_res_radii)},
                                    config.THISLUTAERO,
                                    CONFIGDICT["ss_coarsetofine_ratio"])),
            "species_to_tune" : species_to_tune,
            "prior_monthly_nd13" : ini_nd13,
            "tuned_monthly_nd13" : tun_nd13,
            "this_ifs_nd13" : this_ifs_nd13,
            "this_modis_nd13" : this_modis_nd13,
            "this_modis_nd13_errors" : this_modis_nd13_errors,
            "grosvenor_tau_c_correction": CONFIGDICT["grosvenor_tau_c_correction"],
            "modisndrefsample" : CONFIGDICT["modisndrefsample"],
            "modisndvalidthr" : CONFIGDICT["modisndvalidthr"],
            "samplespreads" : CONFIGDICT["samplespreads"],
            "scale_mcon" : CONFIGDICT["scalemcon"],
            "scale_recipe_ingredient": CONFIGDICT["scale_recipe_ingredient"],
            "ini_mcon_scaler_fields" : ini_mcon_scaler_fields,
            "weightbycloudpresence" : CONFIGDICT["weightbycloudpresence"],
            "monthly_weights" : monthly_weights,
            "bind_seasalt" : CONFIGDICT["bindseasalt"],
            "ss_coarsetofine_ratio" : CONFIGDICT["ss_coarsetofine_ratio"],
            "ini_radii" : ini_radii,
            "firstguess_radii" : firstguess_radii,
            "Pyrcel_LUT" : CONFIGDICT["pyrcellutpath"],
            "actual_pyrcel_lut" : actual_pyrcellut,
            "w_speed" : CONFIGDICT["wspeed"],
            "deardorff_scale" : CONFIGDICT["deardorff_scale"],
            "AEROSPECS" : config.THISLUTAERO,
            "ccn_recipe" : actual_recipe,
            "date" : date,
            "running_time" : f"{time.time()-start_time:.1f}s",
            "ifs_err" : ifs_err,
            "ini_err" : ini_err,
            "delta_err_relative" : delta_err,
            "tuning_res" : tuning_res,
            "logfile" : logfile
    }}

    with open(logfile, 'wb') as fopen:
        print(f"Saving logs to {logfile}...", flush=True)
        pickle.dump(LOGDIC, fopen)

    logger.info(f"Completed tuning for year {year}")
