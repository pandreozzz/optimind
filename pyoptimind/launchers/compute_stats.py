import argparse
import logging

import os
import sys
import glob
import numpy as np
import xarray as xr
import pickle

from ..tools.lut import get_lutaero_from_r0
from ..tools.aerosol import get_nccn_over_mcon_from_speclist

# Vars to load from pickle
LOADVARS = ["tuning_res", "delta_err_relative", "ifs_err"]
# Additional vars to load from first year of pickles
ADDLOADVARS = ["AEROSPECS", "ss_coarsetofine_ratio",
 "cc_thresh", "t_thresh", "iwr_thresh", "prior_ws",
 "min_tot_tau_c", "min_top_tau_c", "modisndvalidthr", "modisndrefsample",
 "firstguess_radii", "ccn_recipe", "Pyrcel_LUT", "prior_nccn_over_mcon"]

def get_tune_logs(logdir, ini_year: int = 2003, end_year: int = 2020):

    tune_logs = {}
    for year in range(ini_year, end_year+1):
        tune_logs[year] = {}
        this_job = glob.glob(os.path.join(logdir, f"job_output_*{year-2000}.txt"))
        this_log = glob.glob(os.path.join(logdir, f"tuning_res_{year}_*.pkl"))
        this_job.sort()
        this_log.sort()
        try:
            if len(this_job) > 0:
                with open(this_job[-1], "r") as fopen:
                    tune_logs[year]["job_output"] = fopen.readlines()
                print(f"Loaded job output for year {year}", flush=True)
            else:
                print(f"Job output year {year} not found.", flush=True)

            if len(this_log) > 0:
                with open(this_log[-1], "rb") as fopen:
                    this_pickle = pickle.load(fopen)

                if not isinstance(this_pickle, dict):
                    raise ValueError("Pickle must be a dictionary!")

                if "AEROSPECS" in this_pickle:
                    for spec in this_pickle["AEROSPECS"]:
                        spec.__post_init__()

                tune_logs[year]["job_logs"] = {}
                for var in LOADVARS:
                    tune_logs[year]["job_logs"][var] = this_pickle[var]

                if year == ini_year:
                    for var in ADDLOADVARS:
                        tune_logs[year]["job_logs"][var] = this_pickle[var]

                del this_pickle


            else:
                raise ValueError(f"Job logs year {year} not found.")

        except Exception as exc:
            raise ValueError(f"Error while opening logs for year {year}: {exc}")

    return tune_logs

def get_tune_stats(tune_logs, ini_year: int = 2003, end_year: int = 2020):


    years = np.arange(ini_year, end_year+1)
    all_radii = np.array([tune_logs[y]["job_logs"]["tuning_res"].x for y in years])
    tun_results = np.array([tune_logs[y]["job_logs"]["tuning_res"].fun for y in years])
    reduction = np.array([tune_logs[y]["job_logs"]["delta_err_relative"] for y in years])
    # If estimate of error function from previous IFS scheme available
    try:
        all_ifs_errfun = np.array([tune_logs[y]["job_logs"]["ifs_err"] for y in years])
    except Exception as exc:
        print(exc)
        all_ifs_errfun = np.array([np.nan for y in years])

    tuning_res_mean = all_radii.mean(axis=0)
    tuning_res_median = np.median(all_radii, axis=0)

    aerospecs  = tune_logs[ini_year]["job_logs"]["AEROSPECS"]
    bind_ss_ratio = tune_logs[ini_year]["job_logs"]["ss_coarsetofine_ratio"]
    if bind_ss_ratio is not None:
        species_to_tune= [sp.name
                          for sp in aerospecs
                         if sp.name != "seasalt2"]
    else:
        species_to_tune = list(aerospecs.keys())

    tune_stats = {
            **{var: tune_logs[ini_year]["job_logs"][var]
               for var in ADDLOADVARS},
            **{
                "jobs_output": {y: tune_logs[y]["job_output"] for y in years},
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
                "tuned_mean_nccn_over_mcon": xr.Dataset(
                    get_nccn_over_mcon_from_speclist(
                        get_lutaero_from_r0(
                            {name:r for name,r in zip(species_to_tune, tuning_res_mean)},
                            aerospecs, bind_ss_ratio)
                        )
                ),
                "tuned_median_nccn_over_mcon": xr.Dataset(
                    get_nccn_over_mcon_from_speclist(
                        get_lutaero_from_r0(
                            {name:r for name,r in zip(species_to_tune, tuning_res_median)},
                            aerospecs, bind_ss_ratio)
                        )
                ),
                }
            }

    return tune_stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute tuning statistics from a tuning log directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example usage: python compute_stats.py /path/to/tune/logs --ini-year 2005 --end-year 2015"
    )
    parser.add_argument("logdir", help="Path to the tuning log directory")
    parser.add_argument("--ini-year", type=int, default=2003, help="Initial year of tuning")
    parser.add_argument("--end-year", type=int, default=2020, help="Final year of tuning")

    args = parser.parse_args()

    logdir  = str(os.path.abspath(args.logdir))

    print(f"Getting tune logs from {logdir} from {args.ini_year} to {args.end_year}...")
    tune_stats = get_tune_stats(get_tune_logs(logdir, ini_year=args.ini_year, end_year=args.end_year),
                                ini_year=args.ini_year, end_year=args.end_year)

    from datetime import datetime as dt
    date = f"{dt.now():%Y%m%d_%H%M%S}"

    logfile = os.path.join(logdir, f"tune_stats_{date}.pkl")
    print(f"Saving stats to {logfile}...", flush=True)
    with open(logfile, 'wb') as fopen:
        pickle.dump(tune_stats, fopen)
