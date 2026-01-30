
"""Main orchestration logic for the LUT tuning workflow."""

from __future__ import annotations

import gc
import logging
import os
import pickle
import time
from datetime import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from . import config
from .config import CONFIGDICT
from ..fields.aerosols import get_aero_fields, get_aero_fromclim
from ..fields.ccn import compute_ccn_species
from ..fields.clouds import get_meteo_cloudy_slices
from ..fields.modis import get_modis_data, get_modis_errors
from ..fields.stage import copy_all_files
from ..lut.setup import get_actual_lut_recipes, setup_pyrcel_lut
from ..tools.aerinterp import get_interpolated_ccn, interpolate_aero
from ..tools.aerosol import compute_ccn_ifs, get_nccn_over_mcon_from_speclist
from ..tools.lut import get_lutaero_from_r0
from ..tools.tuner import compute_err_func, tuning_loop
from ..utils.memory import print_memory_status, trim_memory
from ..utils.spillover import print_spill_status
from ..launchers import launch_tuning  # to access launch_tuning.LOGDIC safely

LOGGER = logging.getLogger(__name__)


def get_tuning_params(actual_recipe: Dict[str, dict]) -> Tuple[List[str], List[float], List[float]]:
    """
    Build the list of species to tune and their initial/first-guess radii.

    Parameters
    ----------
    actual_recipe : dict
        The CCN recipe actually used (filtered for available variables).

    Returns
    -------
    (species_to_tune, ini_radii, firstguess_radii)
    """
    print("Getting tuning params...")  # kept for parity with existing logs

    species_to_tune: List[str] = []
    ini_radii: List[float] = []

    print(config.THISLUTAERO)
    for aero in config.THISLUTAERO:
        print(aero)
        if (aero.name in actual_recipe) and not (
            CONFIGDICT["bindseasalt"] and aero.name == "seasalt2"
        ):
            print(f"adding {aero} to species_to_tune")
            species_to_tune.append(aero.name)
            ini_radii.append(aero.median.item())  # median is a 0-d xarray variable

    # Handle coarse-to-fine constraint for sea salt
    if CONFIGDICT["bindseasalt"]:
        if CONFIGDICT["ss_coarsetofine_ratio"] is None:
            print("Setting sea salt coarse-to-fine ratio to 10")
            CONFIGDICT["ss_coarsetofine_ratio"] = 10
    else:
        CONFIGDICT["ss_coarsetofine_ratio"] = None

    # First guess radii
    if CONFIGDICT["firstguess_radii"]:
        firstguess_radii: List[Optional[float]] = CONFIGDICT["firstguess_radii"]
        if len(firstguess_radii) != len(ini_radii):
            raise ValueError(
                "Invalid length for firstguess_radii vector!! "
                f"Expected length {len(ini_radii)}, got {len(firstguess_radii)}"
            )
        firstguess_radii = [
            f if f is not None else i for f, i in zip(firstguess_radii, ini_radii)
        ]
        firstguess_radii_f: List[float] = [float(x) for x in firstguess_radii]
    else:
        print("setting first-guess radii same as initial radii")
        firstguess_radii_f = ini_radii

    return species_to_tune, ini_radii, firstguess_radii_f


def run_tuning_year(year: int, config_path: str, logdir_path: str) -> None:
    """
    Execute the full-year tuning workflow.

    Parameters
    ----------
    year : int
        Year to process.
    config_path : str
        Path to configuration JSON (for logging reference).
    logdir_path : str
        Directory where results/log pickles will be saved.
    """
    LOGGER.info("Starting LUT tuning for year %s", year)

    # Initialize LUT
    LOGGER.info("Setting up pyrcel LUT from %s", CONFIGDICT["pyrcellutpath"])
    setup_pyrcel_lut(CONFIGDICT["pyrcellutpath"])
    print_memory_status("After LUT setup")
    print_spill_status("After LUT setup")

    # Stage data files
    LOGGER.info("Staging data files for %s", year)
    copy_all_files(year)

    # MODIS data
    print("Loading MODIS Nd data...", flush=True)
    modis_nd_raw = get_modis_data(year)
    this_modis_nd13, this_modis_nd13_errors = get_modis_errors(modis_nd_raw)
    this_modis_nd13 = this_modis_nd13.rename("modis_nd13")
    this_modis_nd13_errors = this_modis_nd13_errors.rename("modis_nd13_errors")
    print(str(this_modis_nd13))
    print(str(this_modis_nd13_errors), flush=True)

    start_time = time.time()

    # Cloudy points
    print("Getting era5 cloudy points...", flush=True)
    this_ifs, this_ifs_fixedlevel = get_meteo_cloudy_slices(
        year,
        cc_thresh=CONFIGDICT["cldetect_cc_threshold"],
        t_thresh=CONFIGDICT["cldetect_t_threshold"],
        iwr_thresh=CONFIGDICT["cldetect_iwr_threshold"],
        use_grosvenor_tau_c=CONFIGDICT["grosvenor_tau_c_correction"],
        thresh_valid_monthly=CONFIGDICT["cldetect_thresh_valid_monthly"],
        latmin=min(CONFIGDICT["latitudes_minmax"]),
        latmax=max(CONFIGDICT["latitudes_minmax"]),
    )
    frac_cloudy_pts = (~np.isnan(this_ifs["p"])).mean().values
    print(f"Fraction of cloudy points: {float(frac_cloudy_pts) * 100:.02f}%")
    print(str(this_ifs), flush=True)

    if (this_ifs_fixedlevel is None) and config.SOME_AEROS_OUT_OF_CLOUD:
        raise ValueError(
            "Got no fixedlevel fields while some aeros should be picked out of cloud"
        )

    # Aerosols
    print("Getting aerosol-related fields...", flush=True)
    gc.collect()
    trim_memory()

    if CONFIGDICT["aerofromclimatology"]:
        this_aero = get_aero_fromclim(year=year)
    else:
        this_aero = get_aero_fields(year, timesel=this_ifs.time.values)

    # Align recipes and aerosol fields
    (
        actual_pyrcellut,
        actual_recipe,
        needed_aeros,
    ) = get_actual_lut_recipes(list(this_aero.variables))

    species_to_tune, ini_radii, firstguess_radii = get_tuning_params(actual_recipe)

    print(f"Pyrcel LUT: {CONFIGDICT['pyrcellutpath']}")
    print(f"actual_recipe: {actual_recipe}")
    print(f"species_to_tune: {species_to_tune}")
    print(f"ini_radii = {ini_radii}")
    print(f"firstguess_radii = {firstguess_radii}")

    # Load necessary fields into memory
    print("Loading needed aero fields in memory...", flush=True)
    this_aero = this_aero[needed_aeros]
    print("done!", flush=True)

    # Global mass scaler (if provided)
    if CONFIGDICT["global_mass_scaler"]:
        for spec, factor in CONFIGDICT["global_mass_scaler"].items():
            if spec in this_aero:
                print(f"Scaling {spec} by factor {factor}")
                this_aero[spec] = this_aero[spec] * factor
            else:
                print(f"WARNING!! Could not scale {spec}: not found in aerosol fields!")

    # SSRH80 fix for sea salt (convert to dry mass)
    if config.SSRH80:
        for spec in needed_aeros:
            if spec.startswith("Sea_Salt"):
                print(f"Scaling {spec} by 1/4.3 to dry mass")
                this_aero[spec] = this_aero[spec] / 4.3

    print(str(this_aero))

    # Interpolate CCN species to ERA5 selection
    print("Interpolating CCN species to ERA5 level selection", flush=True)
    tgt_rhoa = this_ifs["rho_air"]  # pick AERO MMR, not actual mass
    this_ccn_mmr = compute_ccn_species(this_aero, actual_recipe).compute()
    this_ccn_mmr["pressure"] = this_aero["pressure"].compute()

    this_ccn_mcon = get_interpolated_ccn(
        this_ccn_mmr, this_ifs, this_ifs_fixedlevel, actual_recipe
    ) * tgt_rhoa

    if CONFIGDICT["tune_rain_dispersion"]:
        rainratio_tmp = this_ifs["crwc"] / this_ifs["clwc"]  # cloud "top"
        # Align dims where needed
        aligned_dims = [d for d in this_ccn_mcon.dims if d in rainratio_tmp]
        this_ccn_mcon["rainratio"] = rainratio_tmp.transpose(*aligned_dims)

    print(str(this_ccn_mcon))

    # Interpolated aerosols for diagnostics (monthly means)
    print("Computing monthly mean aerosols for diagnostics...", flush=True)
    tgt_pres = this_ifs_fixedlevel["p"] if this_ifs_fixedlevel is not None else this_ifs["p"]

    aero_interp_vars = {}
    for v in needed_aeros:
        if v == "pressure":
            continue
        ds_interp = interpolate_aero(
            this_aero[["pressure", v]], tgt_pres,
            aero_timeinterp=CONFIGDICT["aerofromclimatology"]
        )
        aero_interp_vars[v] = ds_interp[v] * tgt_rhoa

    this_aero_mcon_monthly = xr.Dataset(data_vars=aero_interp_vars)
    this_aero_mcon_monthly = this_aero_mcon_monthly.groupby(
        this_aero_mcon_monthly.time.dt.month
        ).mean()
    print("done!", flush=True)

    # Reduce memory
    del this_ccn_mmr
    del tgt_pres

    monthly_weights = (
        this_ifs["is_warmliquid_cloud_2d"].groupby(this_ifs.time.dt.month).sum()
        if CONFIGDICT["weightbycloudpresence"]
        else None
    )

    # Wind-parametrized CCNs baseline performance
    this_ifs_nd13 = (
        compute_ccn_ifs(
            ws=xr.DataArray(np.sqrt(this_ifs["10u"] ** 2 + this_ifs["10v"] ** 2)),
            lsm=this_ifs["lsm"],
        )
        ** 0.333
    ).groupby(this_ifs["time"].dt.month).mean()  # type: ignore

    this_ifs_nd13_pp = (
        this_ifs_nd13 / (1 + this_ccn_mcon["rainratio"])
        if CONFIGDICT["tune_rain_dispersion"]
        else this_ifs_nd13
    )

    ifs_err = compute_err_func(
        this_ifs_nd13_pp,
        this_modis_nd13,
        this_modis_nd13_errors,
        monthly_weights,
    )
    print(f"IFS (wind-dependent ccns) errfun: {ifs_err:.3f}")

    # Tuning loop
    print("Launching tuning loop...", flush=True)

    # Populate w_mean and w_prime if present in IFS (for LUT extraction)
    for var in ["w_mean", "w_prime"]:
        if var in this_ifs:
            print(f"Populating {var} for LUT extraction")
            this_ccn_mcon[var] = this_ifs[var]

    tuning_res, ini_err, delta_err, ini_nd13, ini_mcon_scaler_fields, tun_nd13 = tuning_loop(
        this_ccn_mcon,
        ini_radii,
        firstguess_radii,
        this_modis_nd13,
        this_modis_nd13_errors,
        actual_pyrcellut,
        species_to_tune=species_to_tune,
        bind_seasalt_ratio=CONFIGDICT["ss_coarsetofine_ratio"],
        wspeed_type=CONFIGDICT["wspeed_type"],
        monthly_weights=monthly_weights,
        tune_rain_dispersion=CONFIGDICT["tune_rain_dispersion"],
    )

    tuning_res_radii = (
        tuning_res.x[:-1] if CONFIGDICT["tune_rain_dispersion"] else tuning_res.x
    )
    tuning_rain_disp_factor: Optional[float] = (
        float(tuning_res.x[-1]) if CONFIGDICT["tune_rain_dispersion"] else None
    )
    _ = tuning_rain_disp_factor  # retained for completeness; not logged directly

    # Compose log record (use module accessor, avoid "global LOGDIC" pattern)
    date_str = f"{dt.now():%Y%m%d_%H%M%S}"
    logfile = os.path.join(logdir_path, f"tuning_res_{year}_{date_str}.pkl")

    prior_ratio = get_nccn_over_mcon_from_speclist(
        get_lutaero_from_r0(
            dict(zip(species_to_tune, ini_radii)),
            config.THISLUTAERO, CONFIGDICT["ss_coarsetofine_ratio"]
        )
    )
    tuned_ratio = get_nccn_over_mcon_from_speclist(
        get_lutaero_from_r0(
            dict(zip(species_to_tune, tuning_res_radii)),
            config.THISLUTAERO,
            CONFIGDICT["ss_coarsetofine_ratio"],
        )
    )

    # Update the central log dictionary
    launch_tuning.LOGDIC = {
        **launch_tuning.LOGDIC,
        **{
            "config_path": config_path,
            "this_ifs": this_ifs.groupby(this_ifs.time.dt.month).mean(),
            "this_ifs_fixedlevel": (
                this_ifs_fixedlevel.groupby(this_ifs.time.dt.month).mean()
                if this_ifs_fixedlevel is not None
                else None
            ),
            "this_aero_mcon_monthly": this_aero_mcon_monthly,
            "this_dates": this_ifs.time,
            "this_ccn_mcon_monthly": this_ccn_mcon.groupby(this_ccn_mcon.time.dt.month).mean(),
            "prior_nccn_over_mcon": prior_ratio,
            "tuned_nccn_over_mcon": tuned_ratio,
            "species_to_tune": species_to_tune,
            "prior_monthly_nd13": ini_nd13,
            "tuned_monthly_nd13": tun_nd13,
            "this_ifs_nd13": this_ifs_nd13,
            "this_modis_nd13": this_modis_nd13,
            "this_modis_nd13_errors": this_modis_nd13_errors,
            "grosvenor_tau_c_correction": CONFIGDICT["grosvenor_tau_c_correction"],
            "modisndrefsample": CONFIGDICT["modisndrefsample"],
            "modisndvalidthr": CONFIGDICT["modisndvalidthr"],
            "samplespreads": CONFIGDICT["samplespreads"],
            "scale_mcon": CONFIGDICT["scalemcon"],
            "scale_recipe_ingredient": CONFIGDICT["scale_recipe_ingredient"],
            "ini_mcon_scaler_fields": ini_mcon_scaler_fields,
            "weightbycloudpresence": CONFIGDICT["weightbycloudpresence"],
            "monthly_weights": monthly_weights,
            "bind_seasalt": CONFIGDICT["bindseasalt"],
            "ss_coarsetofine_ratio": CONFIGDICT["ss_coarsetofine_ratio"],
            "ini_radii": ini_radii,
            "firstguess_radii": firstguess_radii,
            "Pyrcel_LUT": CONFIGDICT["pyrcellutpath"],
            "actual_pyrcel_lut": actual_pyrcellut,
            "w_speed": CONFIGDICT["wspeed"],
            "deardorff_scale": CONFIGDICT["deardorff_scale"],
            "AEROSPECS": config.THISLUTAERO,
            "ccn_recipe": actual_recipe,
            "date": date_str,
            "running_time": f"{time.time() - start_time:.1f}s",
            "ifs_err": ifs_err,
            "ini_err": ini_err,
            "delta_err_relative": delta_err,
            "tuning_res": tuning_res,
            "logfile": logfile,
        },
    }

    # Persist to disk
    os.makedirs(logdir_path, exist_ok=True)
    with open(logfile, "wb") as fopen:
        print(f"Saving logs to {logfile}...", flush=True)
        pickle.dump(launch_tuning.LOGDIC, fopen)

    LOGGER.info("Completed tuning for year %s", year)

