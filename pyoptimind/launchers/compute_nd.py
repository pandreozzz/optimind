import os
import sys
import glob
import dask
from dask.distributed import Client, LocalCluster

import numpy as np
import xarray as xr

import pickle

from ..main import config
from ..main.config import CONFIGDICT

from ..main.config import digest_config
from ..fields.stage import copy_all_files
from ..fields.clouds import get_meteo_cloudy_slices
from ..fields.aerosols import get_aero_fields, get_aero_fromclim
from ..lut.setup import setup_pyrcel_lut, get_actual_lut_recipes
from ..fields.ccn import compute_ccn_species
from ..tools.aerinterp import get_interpolated_ccn
from ..tools.stack import get_stacked_aero, get_stacked_lut
from ..tools.lut import compute_nd

# Include Dust diagnostics
ADDITAEROVAR=["Mineral_Dust_bin1", "Mineral_Dust_bin2"]


def main():
    import time

    year = int(sys.argv[1])
    config_path = str(os.path.abspath(sys.argv[2]))
    logdir  = str(os.path.abspath(sys.argv[3]))
    totmem_gb = float(sys.argv[4])
    CONFIGDICT["nprocs"] = int(sys.argv[5])
    outdir = logdir

    tune_stats_files = glob.glob(os.path.join(logdir, "tune_stats_*.pkl"))
    if len(tune_stats_files) < 1:
        raise ValueError(f"Could not find any tune stats under {logdir}!!")
    print(f"Fetchig tune stats from {tune_stats_files[-1]}")
    with open(tune_stats_files[-1], "rb") as fopen:
        tune_stats = pickle.load(fopen)

    # Read configuration
    digest_config(config_path)

    print("Using dask...")
    #os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"
    global CLIENT, CLUSTER
    CLUSTER = LocalCluster(n_workers=CONFIGDICT["nprocs"],
                          memory_limit=f"{(totmem_gb*0.95)/CONFIGDICT['nprocs']:.2f}GB")
    CLIENT = Client(CLUSTER)
    print("Client ready.")

    if "MALLOC_TRIM_THRESHOLD_" in os.environ:
        print(f"MALLOC_TRIM_THRESHOLD_: {os.environ['MALLOC_TRIM_THRESHOLD_']}")
    else:
        print("MALLOC_TRIM_THRESHOLD_ NOT SET AS ENVIRONMENT VARIABLE")
    print(CLUSTER, flush=True)

    # Catch Exceptions?
    copy_all_files(year)

    start_time = time.time()
    print("Getting era5 cloudy points...", flush=True)
    this_ifs, this_ifs_fixedlevel = get_meteo_cloudy_slices(
        year,
        cc_thresh=CONFIGDICT["cldetect_cc_threshold"],
        t_thresh=CONFIGDICT["cldetect_t_threshold"],
        iwr_thresh=CONFIGDICT["cldetect_iwr_threshold"],
        prior_ws=tune_stats["prior_ws"],
        min_tot_tau_c=tune_stats["min_tot_tau_c"],
        min_top_tau_c=tune_stats["min_top_tau_c"],
        use_grosvenor_tau_c=CONFIGDICT["grosvenor_tau_c_correction"],
        thresh_valid_monthly=CONFIGDICT["cldetect_thresh_valid_monthly"],
        latmin=-90, latmax=90
    )
    frac_cloudy_pts = (~np.isnan(this_ifs['p'])).mean().values
    print(f"Fraction of cloudy points: {frac_cloudy_pts*100:.02f}%")
    print(this_ifs.__str__(), flush=True)

    alpha_tuned = {
        stat_typ: tune_stats[f"rtuned_{stat_typ}"][-1] if CONFIGDICT["tune_rain_dispersion"] else 0
        for stat_typ in ["mean", "median"]
        }
    rainratio = this_ifs["crwc"]/this_ifs["clwc"] if CONFIGDICT["tune_rain_dispersion"] else None

    if CONFIGDICT["aerofromclimatology"]:
        this_aero = get_aero_fromclim(year=year, latmin=-90, latmax=90)
    else:
        this_aero = get_aero_fields(year, timesel=this_ifs.time.values,
                                        latmin=-90, latmax=90)

    setup_pyrcel_lut(CONFIGDICT["pyrcellutpath"])
    actual_lut, actual_recipe, needed_aeros = get_actual_lut_recipes(this_aero)

    all_aero_vars = needed_aeros+ADDITAEROVAR
    this_aero = this_aero[all_aero_vars]

    if CONFIGDICT["global_mass_scaler"]:
        for spec, factor in CONFIGDICT["global_mass_scaler"].items():
            if spec in this_aero:
                print(f"Scaling {spec} by factor {factor}")
                this_aero[spec] = this_aero[spec]*factor
            else:
                print(f"WARNING!! Could not scale {spec}: not found in aerosol fields!")
    if config.SSRH80:
        for spec in all_aero_vars:
            if spec.startswith("Sea_Salt"):
                print(f"Scaling {spec} by 1/4.3 to dry mass")
                this_aero[spec] = this_aero[spec]/4.3

    print("Compute CCN species and interpolate to era5 level selection", flush=True)
    tgt_rhoa = this_ifs["rho_air"] # eventually PICK AERO MMR from fixed level, NOT ACTUAL MASS!!
    this_ccn_mmr = compute_ccn_species(this_aero, actual_recipe).compute()
    this_ccn_mmr["pressure"] = this_aero["pressure"].compute()

    this_ccn_mcon = get_interpolated_ccn(
            this_ccn_mmr, this_ifs,
            this_ifs_fixedlevel, actual_recipe
        )*tgt_rhoa

    this_ccn_mcon_noss = this_ccn_mcon.copy()
    for var in this_ccn_mcon_noss.variables:
        if str(var).lower().startswith("seasalt"):
            print(f"Setting {var} to 0 for no-seasalt computations")
            this_ccn_mcon_noss[var] = this_ccn_mcon_noss[var]*0

    if CONFIGDICT["wspeed_type"] == 0:
        include_w_list = ["w"]
    elif CONFIGDICT["wspeed_type"] < 3:
        include_w_list = ["w_mean"]
    else:
        include_w_list = ["w_mean", "w_prime"]


    this_months = this_ifs.time.dt.month
    activ_results = {}
    for ccn_data, ccn_descr in zip((this_ccn_mcon, this_ccn_mcon_noss), ("_","_noss_")):
        for stat_typ in ("prior", "tuned_median"):
            ### Iteration over cases
            this_nccn_over_mcon = {var: float(tune_stats[f"{stat_typ}_nccn_over_mcon"][var])
                                   for var in actual_recipe}
                # Populate w_mean and w_prime if used
            for var in ["w_mean", "w_prime"]:
                if var in this_ifs:
                    print(f"Populating {var} for LUT extraction")
                    ccn_data[var] = this_ifs[var]

            num_act_var = "num_act_kn" if CONFIGDICT["kinetically_limited"] else "num_act"

            this_ccn_mcon_stack = get_stacked_aero(ccn_data, config.THISLUTAERO, include_w=include_w_list)

            pyrcel_lut_stack = get_stacked_lut(actual_lut, wspeed_type=CONFIGDICT["wspeed_type"],
                                               lutkin=CONFIGDICT["kinetically_limited"])

            this_activ_results = compute_nd(lut_species_mcon_stack=this_ccn_mcon_stack,
                                     pyrcel_lut_stack=pyrcel_lut_stack,
                                     nccn_over_mcon=this_nccn_over_mcon,
                                     lutkin=CONFIGDICT["kinetically_limited"])
            print(this_activ_results.__str__())
            #print(rainratio.__str__(), flush=True)
            #print(alpha_median)
            if rainratio is not None:
                this_activ_results["tot_nd"] = this_activ_results["tot_nd"]/(1 + alpha_tuned[stat_typ]*rainratio)
            activ_results = {
                **activ_results,
                **{
                    f"{stat_typ}{ccn_descr}{spec}_nd": (this_activ_results[f"{config.PYRCNAMEMAP[spec]}_{num_act_var}"]*\
                                                        ccn_data[spec]*tune_stats[f"{stat_typ}_nccn_over_mcon"][spec]
                                                       ).groupby(this_months).mean()
                    for spec in actual_recipe
                },
                **{
                    f"{stat_typ}{ccn_descr}tot_nd" : this_activ_results["tot_nd"].groupby(this_months).mean(),
                    f"{stat_typ}{ccn_descr}tot_nd13" : (this_activ_results["tot_nd"]**0.333).groupby(this_months).mean(),
                },
            }
            del this_activ_results
    activ_results = {
        **activ_results,
        **{
            "cloud_top_p" : this_ifs["p"].groupby(this_months).mean(),
            "fixed_lev_p" : this_ifs_fixedlevel["p"].groupby(this_months).mean() if this_ifs_fixedlevel is not None else None,
            "cloud_top_t" : this_ifs["t"].groupby(this_months).mean(),
            "cloud_tau" : this_ifs["tot_tau_c"].groupby(this_months).mean(),
            "fixed_lev_t" : this_ifs_fixedlevel["t"].groupby(this_months).mean() if this_ifs_fixedlevel is not None else None,
            "fixed_lev_r" : this_ifs_fixedlevel["r"].groupby(this_months).mean() if this_ifs_fixedlevel is not None else None,
            "sp" : this_ifs["sp"].groupby(this_months).mean(),
            "wspeed_type" :  CONFIGDICT["wspeed_type"],
            "wspeed" :  CONFIGDICT["wspeed"],
            "deardorff_scale" : CONFIGDICT["deardorff_scale"],
            "w_mean_min" : CONFIGDICT["w_mean_min"],
            "w_prime" : CONFIGDICT["w_prime_min"],
            "w_prime_min" : CONFIGDICT["w_prime_min"],
        },
        **{
            var : this_ifs[var].groupby(this_months).mean()
            for var in ["w", "w_clbase", "wstar",
                        "w_mean_raw", "w_prime_raw",
                        "w_mean", "w_prime"]
            if var in this_ifs
        },
        **{
            mask: this_ifs[mask].groupby(this_months).mean()
            for mask in ["is_warmliquid_cloud_2d", "cos_sza_mask",
                        "localtime_mask", "total_mask"]
        },
        **{
            f"alpha_{stat_typ}" : alpha_tuned[stat_typ] if CONFIGDICT["tune_rain_dispersion"] else None
            for stat_typ in ["mean", "median"]
        }
    }
    activ_results = xr.Dataset(data_vars=activ_results).expand_dims(year=[year])

    outext = "_zarr" if CONFIGDICT["use_zarr"] else ".nc"
    results_file = os.path.join(outdir, f"tuned_nd_{year}{outext}")
    print(f"Saving results to {results_file}...")

    if os.path.exists(results_file):
        backup_dest = f"{results_file}_BACKUP"
        print(f"Moving previously-existing {results_file} to {backup_dest}")
        os.rename(results_file, backup_dest)

    if CONFIGDICT["use_zarr"]:
        activ_results.to_zarr(results_file)
    else:
        activ_results.to_netcdf(results_file)
