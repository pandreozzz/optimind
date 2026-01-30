
import numpy as np
import xarray as xr
from scipy.optimize import minimize
#from scipy.optimize import Bounds

from ..main import config
from ..main.config import CONFIGDICT

from .stack import get_stacked_aero, get_stacked_lut
from .aerosol import get_nccn_over_mcon_from_speclist
from .lut import get_lutaero_from_r0, compute_nd

def nd_err_func(inpars, lut_species_stack, modis_nd13, pyrcel_lut_stack,
                scale_mcon, species_to_tune, bind_seasalt_ratio,
                modis_errors, timedim="time", reduce_to_monthly=True, return_data_and_scalers=False,
                monthly_weights=None, tune_rain_dispersion=False, rainratio = None):
    """
    su, ni, ss, om, bc
    valid_nd : fraction of valid retrievals at each point to use as weight
    lut_species are in mass concentrations.
    """
    def alpha_disp_bound(x):
        if x < 0:
            return 1.e12
        return x

    if tune_rain_dispersion and (rainratio is None):
        raise ValueError("Rain ratio required if tuning also rain dispersion!!!!!")
    else:
        rainratio = None

    r0 = inpars[:-1] if tune_rain_dispersion else inpars

    if (len(r0) != len(species_to_tune)):
        raise ValueError(f"Error! r0({len(r0)}) and species_to_tune("+\
                         f"{len(species_to_tune)}) should be ordered 1:1 map!")
    this_lutaero = get_lutaero_from_r0(
        {name:r for name,r in zip(species_to_tune, r0)},
        config.THISLUTAERO,
        bind_seasalt_ratio=bind_seasalt_ratio)
    this_nccn_over_mcon = get_nccn_over_mcon_from_speclist(this_lutaero)

    if scale_mcon:
        prior_mass_fr = {aeroname : aerospec.m_act
                         for aeroname,aerospec in this_lutaero.items()}
    else:
        prior_mass_fr = {}

    all_lut_species = list(species_to_tune).copy()
    if bind_seasalt_ratio is not None:
        all_lut_species = all_lut_species+["seasalt2"]

    # Compute Nd using lookup table

    val_out = compute_nd(lut_species_mcon_stack=lut_species_stack,
                         pyrcel_lut_stack=pyrcel_lut_stack,
                         nccn_over_mcon=this_nccn_over_mcon,
                         lutkin=CONFIGDICT["kinetically_limited"],
                         prior_mass_fr=prior_mass_fr)

    if rainratio is not None:
        val_out = val_out/(1 + alpha_disp_bound(inpars[-1])*rainratio)


    nd13_data = val_out["tot_nd"]**0.333


    mcon_scaler_fields={}
    # Need to if modis_nd**1/3 is monthly data!
    if reduce_to_monthly:
        nd13_data = nd13_data.groupby(val_out[timedim].dt.month).mean()
        if scale_mcon:
            for var in prior_mass_fr.keys():
                mcon_scaler_fields[var] = val_out[f"{var}_mass_scaler"].groupby(val_out[timedim].dt.month).mean()
    else:
        nd13_data = nd13_data.squeeze()
        if scale_mcon:
            for var in prior_mass_fr.keys():
                mcon_scaler_fields[var] = val_out[f"{var}_mass_scaler"].squeeze()


    err = compute_err_func(nd13_data, modis_nd13,
                           modis_errors, monthly_weights)

    print(".", end="", flush=True)

    if not (return_data_and_scalers):
        return err

    return err, nd13_data, mcon_scaler_fields

def compute_err_func(nd13_data : xr.DataArray,
                     modis_nd13 : xr.DataArray,
                     modis_errors : xr.DataArray,
                     monthly_weights):

    diff = ((nd13_data - modis_nd13)/modis_errors)**2

    if monthly_weights is not None:
        diff = diff*monthly_weights**2

    diff_weighted = diff.weighted(np.cos(np.deg2rad(diff.lat)))


    return np.sqrt(diff_weighted.mean(skipna=True).values)

def tuning_loop(this_ccn_mcon, ini_radii, firstguess_radii,
                this_modis_nd13, this_modis_errors,
                pyrcel_lut, actual_recipe,
                species_to_tune,
                bind_seasalt_ratio=None, wspeed_type=0, #use_wstar=False,
                monthly_weights=None, tune_rain_dispersion=False
                ):

    if tune_rain_dispersion and ("rainratio" not in this_ccn_mcon):
        raise ValueError("variable rainratio required if tuning also rain dispersion!!!!!")

    if wspeed_type > 0 and ("w_mean" not in this_ccn_mcon):
        raise ValueError("variable w_mean required in this_ccn_mcon" +\
                         f"if using wspeed_type {wspeed_type}")
    if wspeed_type > 2 and ("w_prime" not in this_ccn_mcon):
        raise ValueError("variable w_prime required in this_ccn_mcon" +\
                         f"if using wspeed_type {wspeed_type}")

    if wspeed_type == 0:
        include_w_list = ["w"]
    elif wspeed_type in [1,2]:
        include_w_list = ["w_mean"]
    elif wspeed_type in [3,4]:
        include_w_list = ["w_mean", "w_prime"]
    else:
        raise ValueError("wspeed_type {wspeed_type} not supported. Only values 0 to 4 currently supported.")
    this_ccn_mcon_stack = get_stacked_aero(this_ccn_mcon, config.THISLUTAERO, include_w=include_w_list)

    if tune_rain_dispersion:
        rainratio_stack = this_ccn_mcon["rainratio"].transpose(
            *this_ccn_mcon_stack.dimorder
        ).values.flatten(order='C')
    else:
        rainratio_stack = None

    pyrcel_lut_stack = get_stacked_lut(pyrcel_lut,
                                       wspeed_type=CONFIGDICT["wspeed_type"],
                                       lutkin=CONFIGDICT["kinetically_limited"])

    # None for free bounds
    # else a scipy.optimize.Bounds object
    # Bounds([0.02,0.02,0.06,0.07,0.007],[0.07,0.08,0.13, 0.12, 0.02])
    r0_bounds = None

    timedim="time"

    ini_radii_pp = ini_radii+[0] if CONFIGDICT["tune_rain_dispersion"] else ini_radii
    firstguess_radii_pp = firstguess_radii+[1] if CONFIGDICT["tune_rain_dispersion"] else firstguess_radii

    inierr,ini_nd13,ini_mcon_scaler_fields = \
    nd_err_func(ini_radii_pp, this_ccn_mcon_stack,
                this_modis_nd13, pyrcel_lut_stack,
                scale_mcon=CONFIGDICT["scalemcon"], species_to_tune=species_to_tune,
                bind_seasalt_ratio=bind_seasalt_ratio, modis_errors=this_modis_errors,
                timedim=timedim, reduce_to_monthly=True,
                return_data_and_scalers=True,
                monthly_weights=monthly_weights, tune_rain_dispersion=CONFIGDICT["tune_rain_dispersion"],
                rainratio=rainratio_stack
               )

    print(f"Initial errfun: {inierr:.3f}")

    res = minimize(nd_err_func, firstguess_radii_pp,
                   method="Nelder-Mead", bounds=r0_bounds,
                   args=(this_ccn_mcon_stack,
                         this_modis_nd13, pyrcel_lut_stack, CONFIGDICT["scalemcon"],
                         species_to_tune, bind_seasalt_ratio, this_modis_errors,
                         timedim, True, False,
                         monthly_weights, CONFIGDICT["tune_rain_dispersion"],
                         rainratio_stack))

    _,tun_nd13,_ = nd_err_func(res.x, this_ccn_mcon_stack,
                               this_modis_nd13, pyrcel_lut_stack,
                               scale_mcon=CONFIGDICT["scalemcon"],
                               species_to_tune=species_to_tune, bind_seasalt_ratio=bind_seasalt_ratio,
                               modis_errors=this_modis_errors, timedim=timedim,
                               reduce_to_monthly=True, return_data_and_scalers=True,
                               monthly_weights=monthly_weights,
                               tune_rain_dispersion=CONFIGDICT["tune_rain_dispersion"],
                               rainratio=rainratio_stack
                              )

    finerr = res.fun
    delterr = (finerr-inierr)/inierr
    print(f"Final errfun: {finerr:.3f}")
    print(f"Variation of err (fractional): {delterr:+.3f}")
    print("\n---------\n\n\n")
    print(res)

    return res, inierr, delterr, ini_nd13, ini_mcon_scaler_fields, tun_nd13
