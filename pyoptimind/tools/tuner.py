"""Objective function and driver for the LUT-based tuning."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
from scipy.optimize import minimize

from ..main import config
from ..main.config import CONFIGDICT
from .aerosol import get_nccn_over_mcon_from_speclist
from .lut import compute_nd, get_lutaero_from_r0, _select_include_w_list
from .stack import get_stacked_aero, get_stacked_lut


def nd_err_func(
    inpars: np.ndarray,
    lut_species_stack: Any,
    modis_nd13: xr.DataArray,
    pyrcel_lut_stack: Any,
    scale_mcon: bool,
    species_to_tune: List[str],
    bind_seasalt_ratio: Optional[float],
    modis_errors: xr.DataArray,
    timedim: str = "time",
    reduce_to_monthly: bool = True,
    return_data_and_scalers: bool = False,
    monthly_weights: Optional[xr.DataArray] = None,
    tune_rain_dispersion: bool = False,
    rainratio: Optional[np.ndarray] = None,
) -> Union[float, Tuple[float, xr.DataArray, Dict[str, xr.DataArray]]]:
    """
    Compute the error function (and optionally intermediate data) for optimizer.

    Parameters
    ----------
    inpars : np.ndarray
        Radii parameters (and optionally rain dispersion factor as the last element).
    lut_species_stack : StackAeroTuple-like
        Stacked aerosol fields.
    modis_nd13 : xr.DataArray
        Target MODIS Nd^(1/3) (or monthly mean of it).
    pyrcel_lut_stack : StackLutTuple-like
        Stacked LUT.
    scale_mcon : bool
        If True, return and use mass scaler fields.
    species_to_tune : list[str]
        Names of aerosol species tuned.
    bind_seasalt_ratio : float or None
        Ratio seasalt2 := seasalt1 * ratio (if binding enabled).
    modis_errors : xr.DataArray
        Retrieval uncertainty used for normalization.
    timedim : str
        Time dimension name.
    reduce_to_monthly : bool
        If True, average to monthly values.
    return_data_and_scalers : bool
        If True, return (err, monthly Nd^(1/3), mass scaler fields).
    monthly_weights : xr.DataArray or None
        Optional monthly weights (e.g., cloud occurrence).
    tune_rain_dispersion : bool
        If True, include rain dispersion parameter.
    rainratio : np.ndarray or None
        Flattened rain ratio (aligned with stacked fields).

    Returns
    -------
    float
        Error metric if `return_data_and_scalers` is False.
    tuple
        (err, Nd^(1/3), {var: mass_scaler}) if `return_data_and_scalers` is True.
    """

    def alpha_disp_bound(x_val: float) -> float:
        # Penalize negative dispersion parameter
        return 1.0e12 if x_val < 0 else x_val

    if tune_rain_dispersion and (rainratio is None):
        raise ValueError("Rain ratio required if tuning also rain dispersion!")

    if not tune_rain_dispersion:
        rainratio = None

    r0 = inpars[:-1] if tune_rain_dispersion else inpars
    if len(r0) != len(species_to_tune):
        raise ValueError(
            f"Error! r0({len(r0)}) and species_to_tune({len(species_to_tune)}) must map 1:1!"
        )

    this_lutaero = get_lutaero_from_r0(
        dict(zip(species_to_tune, r0)),
        config.THISLUTAERO,
        bind_seasalt_ratio=bind_seasalt_ratio,
    )
    this_nccn_over_mcon = get_nccn_over_mcon_from_speclist(this_lutaero)

    prior_mass_fr: Dict[str, Any] = (
        {aeroname: aerospec.m_act for aeroname, aerospec in this_lutaero.items()}
        if scale_mcon
        else {}
    )

    # Compute Nd using the lookup table
    val_out = compute_nd(
        lut_species_mcon_stack=lut_species_stack,
        pyrcel_lut_stack=pyrcel_lut_stack,
        nccn_over_mcon=this_nccn_over_mcon,
        lutkin=CONFIGDICT["kinetically_limited"],
        prior_mass_fr=prior_mass_fr,
    )

    if rainratio is not None:
        # Apply dispersion factor
        val_out["tot_nd"] = val_out["tot_nd"] / (1.0 + alpha_disp_bound(inpars[-1]) * rainratio)

    nd13_data: xr.DataArray = val_out["tot_nd"] ** 0.333

    mcon_scaler_fields: Dict[str, xr.DataArray] = {}
    if reduce_to_monthly:
        nd13_data = nd13_data.groupby(val_out[timedim].dt.month).mean()
        if scale_mcon:
            for var in prior_mass_fr:
                mcon_scaler_fields[var] = (
                    val_out[f"{var}_mass_scaler"].groupby(val_out[timedim].dt.month).mean()
                )
    else:
        nd13_data = nd13_data.squeeze()
        if scale_mcon:
            for var in prior_mass_fr:
                mcon_scaler_fields[var] = val_out[f"{var}_mass_scaler"].squeeze()

    # Error function is evaluated on monthly means
    err = compute_err_func(nd13_data, modis_nd13, modis_errors, monthly_weights)
    print(".", end="", flush=True)

    if not return_data_and_scalers:
        return float(err)
    return float(err), nd13_data, mcon_scaler_fields


def compute_err_func(
    nd13_data: xr.DataArray,
    modis_nd13: xr.DataArray,
    modis_errors: xr.DataArray,
    monthly_weights: Optional[xr.DataArray],
) -> float:
    """
    Compute a latitude-weighted RMSE of Nd^(1/3), with optional monthly weights.
    """
    diff = ((nd13_data - modis_nd13) / modis_errors) ** 2
    if monthly_weights is not None:
        diff = diff * (monthly_weights ** 2)
    diff_weighted = diff.weighted(np.cos(np.deg2rad(diff.lat)))
    return float(np.sqrt(diff_weighted.mean(skipna=True).values))


def tuning_loop(  # pylint: disable=too-many-arguments, too-many-locals
    this_ccn_mcon: xr.Dataset,
    ini_radii: List[float],
    firstguess_radii: List[float],
    this_modis_nd13: xr.DataArray,
    this_modis_errors: xr.DataArray,
    pyrcel_lut: xr.Dataset,
    species_to_tune: List[str],
    bind_seasalt_ratio: Optional[float] = None,
    wspeed_type: int = 0,
    monthly_weights: Optional[xr.DataArray] = None,
    tune_rain_dispersion: bool = False,
):
    """
    Run the Nelder-Mead tuning over initial/first-guess radii (and optional dispersion).
    """
    if tune_rain_dispersion and ("rainratio" not in this_ccn_mcon):
        raise ValueError("variable rainratio required if tuning also rain dispersion!")
    if wspeed_type > 0 and ("w_mean" not in this_ccn_mcon):
        raise ValueError("variable w_mean required in this_ccn_mcon " +\
                         f"if using wspeed_type {wspeed_type}")
    if wspeed_type > 2 and ("w_prime" not in this_ccn_mcon):
        raise ValueError("variable w_prime required in this_ccn_mcon " +\
                         f"if using wspeed_type {wspeed_type}")

    include_w_list = _select_include_w_list(wspeed_type)

    ccn_mcon_stack = get_stacked_aero(this_ccn_mcon, config.THISLUTAERO, include_w=include_w_list)
    if tune_rain_dispersion:
        rainratio_stack = (
            this_ccn_mcon["rainratio"].transpose(*ccn_mcon_stack.dimorder).values.flatten(order="C")
        )
    else:
        rainratio_stack = None

    pyrcel_lut_stack = get_stacked_lut(
        pyrcel_lut, wspeed_type=CONFIGDICT["wspeed_type"], lutkin=CONFIGDICT["kinetically_limited"]
    )

    r0_bounds = None  # kept for API completeness

    timedim = "time"
    ini_radii_pp = ini_radii + [0] if CONFIGDICT["tune_rain_dispersion"] else ini_radii
    firstguess_radii_pp = (
        firstguess_radii + [1] if CONFIGDICT["tune_rain_dispersion"] else firstguess_radii
    )

    # Initial error
    inierr, ini_nd13, ini_mcon_scaler_fields = nd_err_func(
        np.asarray(ini_radii_pp, dtype=float),
        ccn_mcon_stack,
        this_modis_nd13,
        pyrcel_lut_stack,
        scale_mcon=CONFIGDICT["scalemcon"],
        species_to_tune=species_to_tune,
        bind_seasalt_ratio=bind_seasalt_ratio,
        modis_errors=this_modis_errors,
        timedim=timedim,
        reduce_to_monthly=True,
        return_data_and_scalers=True,
        monthly_weights=monthly_weights,
        tune_rain_dispersion=CONFIGDICT["tune_rain_dispersion"],
        rainratio=rainratio_stack,
    )
    print(f"Initial errfun: {inierr:.3f}")

    # Optimization
    res = minimize(
        nd_err_func,
        np.asarray(firstguess_radii_pp, dtype=float),
        method="Nelder-Mead",
        bounds=r0_bounds,
        args=(
            ccn_mcon_stack,
            this_modis_nd13,
            pyrcel_lut_stack,
            CONFIGDICT["scalemcon"],
            species_to_tune,
            bind_seasalt_ratio,
            this_modis_errors,
            timedim,
            True,
            False,
            monthly_weights,
            CONFIGDICT["tune_rain_dispersion"],
            rainratio_stack,
        ),
    )

    _, tun_nd13, _ = nd_err_func(
        res.x,
        ccn_mcon_stack,
        this_modis_nd13,
        pyrcel_lut_stack,
        scale_mcon=CONFIGDICT["scalemcon"],
        species_to_tune=species_to_tune,
        bind_seasalt_ratio=bind_seasalt_ratio,
        modis_errors=this_modis_errors,
        timedim=timedim,
        reduce_to_monthly=True,
        return_data_and_scalers=True,
        monthly_weights=monthly_weights,
        tune_rain_dispersion=CONFIGDICT["tune_rain_dispersion"],
        rainratio=rainratio_stack,
    )

    finerr = float(res.fun)
    delterr = (finerr - inierr) / inierr
    print(f"Final  errfun: {finerr:.3f}")
    print(f"Variation of err (fractional): {delterr:+.3f}")
    print("\n---------\n\n")
    print(res)
    return res, inierr, delterr, ini_nd13, ini_mcon_scaler_fields, tun_nd13
