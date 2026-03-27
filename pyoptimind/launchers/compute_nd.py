"""Compute (diagnostic) Nd fields for a given year using tuned/prior parameters."""

import argparse
import glob
import os
import pickle
from typing import Optional, Any, Dict, List

import numpy as np
import xarray as xr
from dask.distributed import Client

from ..main import config
from ..main.config import CONFIGDICT, digest_config
from ..fields.aerosols import get_aero_fields, get_aero_fromclim
from ..fields.ccn import compute_ccn_species
from ..fields.clouds import get_meteo_cloudy_slices
from ..fields.stage import copy_all_files
from ..lut.setup import get_actual_lut_recipes, setup_pyrcel_lut
from ..tools.aerinterp import get_interpolated_ccn, interpolate_aero
from ..tools.lut import compute_nd, _select_include_w_list
from ..tools.stack import get_stacked_aero, get_stacked_lut
from ..tools.aerosol import compute_ccn_ifs

from ..utils.memory import get_available_memory
from ..utils.daskctrl import optimize_dask_for_memory

# Include Dust diagnostics
ADDIT_AEROVAR = ["Mineral_Dust_bin1", "Mineral_Dust_bin2"]


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for compute_nd."""
    parser = argparse.ArgumentParser(
        description="Compute Nd fields for a given year using tuned/prior parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Examples\ntbdone",
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Year to process"
        )
    # Optional argument
    parser.add_argument(
        "--meteo-year",
        type=int,
        default=None,
        help="Meteorology comes from a different year"
        )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration JSON"
        )
    parser.add_argument(
        "--logdir",
        type=str,
        help="Path to output/log directory"
        )
    parser.add_argument(
        "--stats-file",
        type=str,
        default=None,
        help="Path to stats file instead of fetching latest from logdir"
        )
    parser.add_argument(
        "--num-procs",
        type=int,
        default=4,
        help="Number of workers for the Dask local cluster")
    parser.add_argument(
        "--save-alltimes",
        action="store_true",
        default=False,
        help="Whether to save nd for all times in addition to monthly means")
    return parser


def _load_latest_tune_stats(logdir: str, statsfile: Optional[str] = None) -> Dict[str, Any]:
    """Load the most recent tuning stats pickle from a directory or file if specified."""

    if statsfile is None:
        tune_stats_files = glob.glob(os.path.join(logdir, "tune_stats_*.pkl"))
        if len(tune_stats_files) < 1:
            raise ValueError(f"Could not find any tune stats under {logdir}!!")
        tune_stats_files.sort()
        this_statsfile = tune_stats_files[-1]
    else:
        this_statsfile = statsfile

    this_statsfile = os.path.realpath(this_statsfile)
    print(f"Fetching tune stats from {this_statsfile}...")
    with open(this_statsfile, "rb") as fopen:
        return pickle.load(fopen)


def _save_results(ds: xr.Dataset, outdir: str, year: int, meteo_year = None) -> None:
    """Persist results to either Zarr or NetCDF based on config."""
    outext = "_zarr" if CONFIGDICT["use_zarr"] else ".nc"
    filename = f"tuned_nd_{year}" if meteo_year is None else f"tuned_nd_{year}_meteo{meteo_year}"
    results_file = os.path.join(outdir, f"{filename}{outext}")
    print(f"Saving results to {results_file}...")
    if os.path.exists(results_file):
        backup_dest = f"{results_file}_BACKUP"
        print(f"Moving previously-existing {results_file} to {backup_dest}")
        os.rename(results_file, backup_dest)
    if CONFIGDICT["use_zarr"]:
        ds.to_zarr(results_file)
    else:
        ds.to_netcdf(results_file)

def run_compute_nd_year(year, logdir : str, statsfile : Optional[str] = None,
                        meteo_year : Optional[int] = None,
                        save_alltimes : Optional[bool] = False) -> None:
    """"Fetch tuning stats and compute nd for the year"""
    # Stage I/O
    copy_all_files(year, copy_modis_nd=False, meteo_year=meteo_year)

    tune_stats = _load_latest_tune_stats(logdir=logdir, statsfile=statsfile)

    if save_alltimes:
        print("Will save nd for alltimes", flush=True)

    # Meteorology (cloudy slices)
    print("Getting era5 cloudy points...", flush=True)
    cos_sza_minmax = CONFIGDICT["cos_sza_minmax_nd"] or CONFIGDICT["cos_sza_minmax"]
    localhour_minmax = CONFIGDICT["localhour_minmax_nd"] or CONFIGDICT["localhour_minmax"]

    print(f"cos_sza_minmax: {cos_sza_minmax}", flush=True)
    print(f"Localhour minmax: {localhour_minmax}", flush=True)

    this_ifs, this_ifs_fixedlevel = get_meteo_cloudy_slices(
        meteo_year if meteo_year is not None else year,
        cc_thresh=CONFIGDICT["cldetect_cc_threshold"],
        hcc_max=CONFIGDICT["cldetect_hcc_threshold"],
        t_thresh=CONFIGDICT["cldetect_t_threshold"],
        iwr_thresh=CONFIGDICT["cldetect_iwr_threshold"],
        prior_ws=tune_stats["prior_ws"],
        min_tot_tau_c=tune_stats["min_tot_tau_c"],
        min_top_tau_c=tune_stats["min_top_tau_c"],
        use_grosvenor_tau_c=CONFIGDICT["grosvenor_tau_c_correction"],
        thresh_valid_monthly=CONFIGDICT["cldetect_thresh_valid_monthly"],
        latmin=-90, latmax=90, lonwest=0, loneast=360,
        cos_sza_minmax=cos_sza_minmax, localhour_minmax=localhour_minmax
    )

    # Change year in time dimension for this_ifs, this_ifs_fixedlevel
    # to match the aerosol year
    if meteo_year is not None:
        new_time = this_ifs["time"].to_index().map(lambda t: t.replace(year=year))
        this_ifs = this_ifs.assign_coords(time=new_time)
        if this_ifs_fixedlevel is not None:
            this_ifs_fixedlevel = this_ifs_fixedlevel.assign_coords(time=new_time)

    frac_cloudy_pts = (~np.isnan(this_ifs["p"])).mean().values
    print(f"Fraction of cloudy points: {float(frac_cloudy_pts) * 100:.02f}%")
    print(str(this_ifs), flush=True)

    # Rain dispersion factor(s)
    alpha_tuned = {
        stat_typ: tune_stats[f"rtuned_{stat_typ}"][-1] if CONFIGDICT["tune_rain_dispersion"] else 0
        for stat_typ in ["mean", "median"]
    }
    rainratio = (
        this_ifs["crwc"] / this_ifs["clwc"] if CONFIGDICT["tune_rain_dispersion"] else None
    )

    # Aerosols (either from climatology or online fields)
    if CONFIGDICT["aerofromclimatology"]:
        this_aero = get_aero_fromclim(year=year, latmin=-90, latmax=90, lonwest=0, loneast=360)
    else:
        this_aero = get_aero_fields(year, timesel=this_ifs.time.values, latmin=-90, latmax=90, lonwest=0, loneast=360)

    # LUT setup / selection
    setup_pyrcel_lut(CONFIGDICT["pyrcellutpath"])
    actual_lut, actual_recipe, needed_aeros = get_actual_lut_recipes(this_aero)

    all_aero_vars = needed_aeros + ADDIT_AEROVAR
    this_aero = this_aero[all_aero_vars]

    # Optional global mass scaling
    if CONFIGDICT["global_mass_scaler"]:
        for spec, factor in CONFIGDICT["global_mass_scaler"].items():
            if spec in this_aero:
                print(f"Scaling {spec} by factor {factor}")
                this_aero[spec] = this_aero[spec] * factor
            else:
                print(f"WARNING!! Could not scale {spec}: not found in aerosol fields!")

    # SSRH80 adjustment for sea-salt (dry mass)
    if config.SSRH80:
        for spec in all_aero_vars:
            if spec.startswith("Sea_Salt"):
                print(f"Scaling {spec} by 1/4.3 to dry mass")
                this_aero[spec] = this_aero[spec] / 4.3

    # CCN species and interpolation to ERA5 levels
    print("Compute CCN species and interpolate to ERA5 level selection", flush=True)
    tgt_rhoa = this_ifs["rho_air"]  # PICK AERO MMR from fixed level, NOT ACTUAL MASS!!
    this_ccn_mmr = compute_ccn_species(this_aero, actual_recipe).compute()
    this_ccn_mmr["pressure"] = this_aero["pressure"].compute()

    this_ccn_mcon = get_interpolated_ccn(this_ccn_mmr, this_ifs, this_ifs_fixedlevel, actual_recipe) * tgt_rhoa

    # No‑sea‑salt variant
    this_ccn_mcon_noss = this_ccn_mcon.copy()
    for var in list(this_ccn_mcon_noss.variables):
        if str(var).lower().startswith("seasalt"):
            print(f"Setting {var} to 0 for no‑seasalt computations")
            this_ccn_mcon_noss[var] = this_ccn_mcon_noss[var] * 0


    # Interpolated aerosols for diagnostics (monthly means)
    print("Computing monthly mean aerosols for diagnostics...", flush=True)
    tgt_pres = this_ifs_fixedlevel["p"] if this_ifs_fixedlevel is not None else this_ifs["p"]

    aero_interp_vars = {}
    for aerovar in all_aero_vars:
        if aerovar == "pressure":
            continue
        aero_interp_vars[aerovar] = interpolate_aero(
            this_aero[["pressure", aerovar]], tgt_pres,
            aero_timeinterp=CONFIGDICT["aerofromclimatology"]
        )[aerovar]* tgt_rhoa

    this_aero_mcon_monthly = xr.Dataset(data_vars=aero_interp_vars)
    this_aero_mcon_monthly = this_aero_mcon_monthly.groupby(
        this_aero_mcon_monthly.time.dt.month
        ).mean()
    print("done!", flush=True)

    print("Computing ifs nd...", flush=True)
    this_ifs_nd = compute_ccn_ifs(
        ws=xr.DataArray(np.sqrt(this_ifs["10u"] ** 2 + this_ifs["10v"] ** 2)),
        lsm=this_ifs["lsm"],
    )
    this_ifs_nd13_mm = (this_ifs_nd**(0.333)).groupby(this_ifs_nd.time.dt.month).mean()
    this_ifs_nd_mm = this_ifs_nd.groupby(this_ifs_nd.time.dt.month).mean()
    if not save_alltimes:
        del this_ifs_nd
        this_ifs_nd = None
    print("Done", flush=True)

    include_w_list = _select_include_w_list(CONFIGDICT["wspeed_type"])
    this_months = this_ifs.time.dt.month

    activ_results: Dict[str, Any] = {}

    for ccn_data, ccn_descr in zip((this_ccn_mcon, this_ccn_mcon_noss), ("_", "_noss_")):
        for stat_typ in ("prior", "tuned_median"):
            # Per‑case factors
            this_nccn_over_mcon = {
                var: float(tune_stats[f"{stat_typ}_nccn_over_mcon"][var]) for var in actual_recipe
            }

            # Populate w variables if present
            for var in ["w_mean", "w_prime"]:
                if var in this_ifs:
                    print(f"Populating {var} for LUT extraction")
                    ccn_data[var] = this_ifs[var]

            num_act_var = "num_act_kn" if CONFIGDICT["kinetically_limited"] else "num_act"

            ccn_mcon_stack = get_stacked_aero(ccn_data, config.THISLUTAERO, include_w=include_w_list)
            pyrcel_lut_stack = get_stacked_lut(
                actual_lut, wspeed_type=CONFIGDICT["wspeed_type"], lutkin=CONFIGDICT["kinetically_limited"]
            )

            this_activ_results = compute_nd(
                lut_species_mcon_stack=ccn_mcon_stack,
                pyrcel_lut_stack=pyrcel_lut_stack,
                nccn_over_mcon=this_nccn_over_mcon,
                lutkin=CONFIGDICT["kinetically_limited"],
            )
            #print(str(this_activ_results))

            if rainratio is not None:
                this_activ_results["tot_nd"] = this_activ_results["tot_nd"] / (
                    1 + alpha_tuned[stat_typ] * rainratio
                )

            activ_results = {
                **activ_results,
                **{
                    f"{stat_typ}{ccn_descr}{spec}_nd": (
                        this_activ_results[f"{config.PYRCNAMEMAP[spec]}_{num_act_var}"]
                        * ccn_data[spec]
                        * tune_stats[f"{stat_typ}_nccn_over_mcon"][spec]
                    ).groupby(this_months).mean()
                    for spec in actual_recipe
                },
                **{
                    f"{stat_typ}{ccn_descr}tot_nd": this_activ_results["tot_nd"].groupby(this_months).mean(),
                    f"{stat_typ}{ccn_descr}tot_nd13": (this_activ_results["tot_nd"] ** 0.333).groupby(this_months).mean(),
                },
                **{
                    f"{stat_typ}{ccn_descr}tot_nd_allt": this_activ_results["tot_nd"] if save_alltimes else None
                }
            }

    print("Gathering all results...", flush=True)
    # Meteorological context and config echoes
    activ_results = {
        **activ_results,
        **{
            "ifs_nd" : this_ifs_nd_mm,
            "ifs_nd13" :this_ifs_nd13_mm,
            "ifs_nd_alltimes" : this_ifs_nd,
            "cloud_top_p": this_ifs["p"].groupby(this_months).mean(),
            "fixed_lev_p": (
                this_ifs_fixedlevel["p"].groupby(this_months).mean() if this_ifs_fixedlevel is not None else None
            ),
            "cloud_top_t": this_ifs["t"].groupby(this_months).mean(),
            "cloud_tau": this_ifs["tot_tau_c"].groupby(this_months).mean(),
            "fixed_lev_t": (
                this_ifs_fixedlevel["t"].groupby(this_months).mean() if this_ifs_fixedlevel is not None else None
            ),
            "fixed_lev_r": (
                this_ifs_fixedlevel["r"].groupby(this_months).mean() if this_ifs_fixedlevel is not None else None
            ),
            "sp": this_ifs["sp"].groupby(this_months).mean(),
            "wspeed_type": CONFIGDICT["wspeed_type"],
            "wspeed": CONFIGDICT["wspeed"],
            "deardorff_scale": CONFIGDICT["deardorff_scale"],
            "w_mean_min": CONFIGDICT["w_mean_min"],
            "w_prime_min": CONFIGDICT["w_prime_min"],
        },
        **{
            var: this_ifs[var].groupby(this_months).mean()
            for var in ["w", "w_clbase", "wstar", "w_mean_raw", "w_prime_raw", "w_mean", "w_prime"]
            if var in this_ifs
        },
        **{
            f"{mask}_sum": this_ifs[mask].groupby(this_months).sum()
            for mask in ["is_warmliquid_cloud_2d", "cos_sza_mask", "localtime_mask", "total_mask"]
        },
        **{
            f"{mask}_mean": this_ifs[mask].groupby(this_months).mean()
            for mask in ["is_warmliquid_cloud_2d", "cos_sza_mask", "localtime_mask", "total_mask"]
        },
        **{
            f"alpha_{stat_typ}": alpha_tuned[stat_typ] if CONFIGDICT["tune_rain_dispersion"] else None
            for stat_typ in ["mean", "median"]
        },
        **{
            f"{v}_mcon" : this_aero_mcon_monthly[v]
            for v in all_aero_vars
            if v != "pressure"
        }
    }

    activ_results_ds = xr.Dataset(data_vars=activ_results).expand_dims(year=[year])
    _save_results(ds=activ_results_ds, outdir=logdir, year=year, meteo_year=meteo_year)

def main() -> int:
    """CLI entrypoint to compute Nd fields given tuning stats."""
    parser = build_parser()
    args = parser.parse_args()


    # Read configuration
    digest_config(args.config)

    with optimize_dask_for_memory():
        totmem_mbytes = get_available_memory()
        print(f"Total memory {totmem_mbytes:.2f}MB")

        mem_per_worker_mb = max(totmem_mbytes * 0.98 / max(args.num_procs, 1), 256.0)

        with Client(n_workers=args.num_procs, memory_limit=f"{mem_per_worker_mb:.2f}MB"):
            run_compute_nd_year(year=args.year,
                                logdir=args.logdir, statsfile=args.stats_file,
                                meteo_year=args.meteo_year, save_alltimes=args.save_alltimes)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
