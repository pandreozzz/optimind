"""Aggregate tuning results across years and compute summary statistics."""

import argparse
import glob
import logging
import os
import pickle
from typing import Any, Dict, Mapping

import numpy as np
import xarray as xr
from datetime import datetime

from ..tools.aerosol import get_nccn_over_mcon_from_speclist
from ..tools.lut import get_lutaero_from_r0

LOGGER = logging.getLogger(__name__)

# Vars to load from pickle
LOADVARS = ["tuning_res", "delta_err_relative", "ifs_err"]

# Additional vars to load from first year of pickles
ADDLOADVARS = [
    "AEROSPECS",
    "ss_coarsetofine_ratio",
    "cc_thresh",
    "t_thresh",
    "iwr_thresh",
    "prior_ws",
    "min_tot_tau_c",
    "min_top_tau_c",
    "modisndvalidthr",
    "modisndrefsample",
    "firstguess_radii",
    "ccn_recipe",
    "Pyrcel_LUT",
    "prior_nccn_over_mcon",
]


def get_tune_logs(logdir: str, ini_year: int = 2003,
                  end_year: int = 2020) -> Dict[int, Dict[str, Any]]:
    """Collect per‑year job output and pickle logs from a tuning log directory."""
    tune_logs: Dict[int, Dict[str, Any]] = {}
    for year in range(ini_year, end_year + 1):
        tune_logs[year] = {}

        job_files = glob.glob(os.path.join(logdir, f"job_output_*{year-2000}.txt"))
        pkl_files = glob.glob(os.path.join(logdir, f"tuning_res_{year}_*.pkl"))
        job_files.sort()
        pkl_files.sort()

        # Job output (optional)
        if job_files:
            with open(job_files[-1], "r", encoding="utf-8") as fopen:
                tune_logs[year]["job_output"] = fopen.readlines()
            print(f"Loaded job output for year {year}", flush=True)
        else:
            print(f"Job output year {year} not found.", flush=True)

        # Pickled results (mandatory)
        if not pkl_files:
            raise ValueError(f"Job logs year {year} not found.")

        try:
            with open(pkl_files[-1], "rb") as fopen:
                loaded = pickle.load(fopen)
        except Exception as exc:  # pylint: disable=broad-except
            raise ValueError(f"Error while opening logs for year {year}: {exc}") from exc

        if not isinstance(loaded, dict):
            raise ValueError("Pickle must be a dictionary!")

        # Re‑post‑init any dataclass-like objects if present in pickles
        if "AEROSPECS" in loaded:
            for spec in loaded["AEROSPECS"]:
                # Defensive: ensure NumPy types are cast properly after unpickle
                if hasattr(spec, "__post_init__"):
                    spec.__post_init__()  # type: ignore[attr-defined]

        tune_logs[year]["job_logs"] = {}
        for var in LOADVARS:
            tune_logs[year]["job_logs"][var] = loaded[var]

        if year == ini_year:
            for var in ADDLOADVARS:
                tune_logs[year]["job_logs"][var] = loaded[var]

    return tune_logs


def get_tune_stats(
    tune_logs: Mapping[int, Any], ini_year: int = 2003, end_year: int = 2020
) -> Dict[str, Any]:
    """Compute multi‑year statistics from the collected tuning logs."""
    years = np.arange(ini_year, end_year + 1)

    all_radii = np.array([tune_logs[y]["job_logs"]["tuning_res"].x for y in years])
    tun_results = np.array([tune_logs[y]["job_logs"]["tuning_res"].fun for y in years])
    reduction = np.array([tune_logs[y]["job_logs"]["delta_err_relative"] for y in years])

    try:
        all_ifs_errfun = np.array([tune_logs[y]["job_logs"]["ifs_err"] for y in years])
    except Exception as exc:  # pylint: disable=broad-except
        print(exc)
        all_ifs_errfun = np.array([np.nan for _ in years])

    tuning_res_mean = all_radii.mean(axis=0)
    tuning_res_median = np.median(all_radii, axis=0)

    aerospecs = tune_logs[ini_year]["job_logs"]["AEROSPECS"]
    bind_ss_ratio = tune_logs[ini_year]["job_logs"]["ss_coarsetofine_ratio"]

    if bind_ss_ratio is not None:
        species_to_tune = [sp.name for sp in aerospecs if sp.name != "seasalt2"]
    else:
        # If aerospecs is a mapping (older pickle), fallback to its keys.
        species_to_tune = list(getattr(aerospecs, "keys")) or [sp.name for sp in aerospecs]

    tuned_mean_nccn = xr.Dataset(
        get_nccn_over_mcon_from_speclist(
            get_lutaero_from_r0(
                dict(zip(species_to_tune, tuning_res_mean)),
                aerospecs, bind_ss_ratio)
        )
    )
    tuned_median_nccn = xr.Dataset(
        get_nccn_over_mcon_from_speclist(
            get_lutaero_from_r0(
                dict(zip(species_to_tune, tuning_res_median)),
                aerospecs, bind_ss_ratio)
        )
    )

    tune_stats: Dict[str, Any] = {
        **{var: tune_logs[ini_year]["job_logs"][var] for var in ADDLOADVARS},
        **{
            "jobs_output": {int(y): tune_logs[int(y)]["job_output"]
                            for y in years if "job_output" in tune_logs[int(y)]},
            "tuned_specs": species_to_tune,
            "rtuned": all_radii,
            "rtuned_mean": tuning_res_mean,
            "rtuned_median": tuning_res_median,
            "rtuned_std": all_radii.std(axis=0),
            "mean_reduction": reduction.mean(axis=0),
            "tun_res_mean": tun_results.mean(axis=0),
            "tun_res_std": tun_results.std(axis=0),
            "ifs_err_mean": all_ifs_errfun.mean(axis=0),
            "ifs_err_std": all_ifs_errfun.std(axis=0),
            "tuned_mean_nccn_over_mcon": tuned_mean_nccn,
            "tuned_median_nccn_over_mcon": tuned_median_nccn,
        },
    }
    return tune_stats


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for compute_stats."""
    parser = argparse.ArgumentParser(
        description="Compute tuning statistics from a tuning log directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: python compute_stats.py /path/to/logs --ini-year 2005 --end-year 2015",
    )
    parser.add_argument("logdir", help="Path to the tuning log directory")
    parser.add_argument("--ini-year", type=int, default=2003, help="Initial year of tuning")
    parser.add_argument("--end-year", type=int, default=2020, help="Final year of tuning")
    return parser


def main() -> int:
    """CLI entrypoint for computing and saving stats."""
    parser = build_parser()
    args = parser.parse_args()

    logdir = str(os.path.abspath(args.logdir))
    print(f"Getting tune logs from {logdir} from {args.ini_year} to {args.end_year}...")

    tune_stats = get_tune_stats(
        get_tune_logs(logdir, ini_year=args.ini_year, end_year=args.end_year),
        ini_year=args.ini_year,
        end_year=args.end_year,
    )

    date = f"{datetime.now():%Y%m%d_%H%M%S}"
    logfile = os.path.join(logdir, f"tune_stats_{date}.pkl")
    print(f"Saving stats to {logfile}...", flush=True)
    with open(logfile, "wb") as fopen:
        pickle.dump(tune_stats, fopen)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
