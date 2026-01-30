"""Aerosol spec utilities: dataclass, lognormal helpers, and CCN parameterizations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np
import xarray as xr

from scipy.integrate import quad
from scipy.special import erf

__all__ = [
    "IFSAeroSpecs",
    "preprocess_params",
    "get_nccn_over_mcon",
    "lognorm_pdf",
    "lognorm_x3_pdf",
    "lognorm_cdf",
    "lognorm_x3_cdf",
    "get_nccn_over_mcon_from_specs",
    "get_nccn_over_mcon_from_speclist",
    "compute_ccn_ifs",
]


@dataclass
class IFSAeroSpecs:
    """IFS aerosol specification.

    Parameters
    ----------
    name : str
        Species name.
    median : np.ndarray
        Median (geometric-mean) radius for each mode.
    shape : np.ndarray
        Geometric standard deviation for each mode.
    density : float
        Species density [kg m^-3].
    rmin, rmax : float
        Min/max radius bounds.
    fact : np.ndarray
        Fraction per mode (multimodal only).
    m_act : float
        Activated mass fraction (hydrophilic species).
    nmodes : int
        Number of modes.
    """

    name: str
    median: np.ndarray = field(default_factory=lambda: np.array(np.nan))
    shape: np.ndarray = field(default_factory=lambda: np.array(np.nan))
    density: float = 1.0
    rmin: float = 0.0
    rmax: float = float(np.inf)
    fact: np.ndarray = field(default_factory=lambda: np.array(np.nan))
    m_act: float = 0.0
    nmodes: int = 1

    def __post_init__(self) -> None:
        # Make any provided lists immutable and coerce to expected types
        for fld, fld_type in self.__annotations__.items():
            val = getattr(self, fld)
            if isinstance(val, list):
                setattr(self, fld, tuple(val))
                val = getattr(self, fld)

            if fld in ["median", "shape", "fact"]:
                setattr(self, fld, np.array(val, dtype=float))
            if fld in ["density", "rmin", "rmax", "m_act"]:
                setattr(self, fld, float(val))

            # Best-effort type assertion (np.ndarray works with isinstance)
            if fld_type is np.ndarray and not isinstance(getattr(self, fld), np.ndarray):
                raise TypeError(
                    f"The field `{fld}` must be np.ndarray, not {type(getattr(self, fld))}"
                )


def preprocess_params(
    r_med: Union[float, np.ndarray],
    shape: Union[float, np.ndarray],
    mode_factors: Optional[Union[float, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize input parameters to 1-D arrays with consistent length."""
    r_med_out = np.atleast_1d(np.array(r_med, dtype=float)).copy()
    s_out = np.atleast_1d(np.array(shape, dtype=float)).copy()

    n_modes = len(r_med_out)
    if len(s_out) != n_modes:
        raise ValueError(f"r_med({len(r_med_out)}) and shape({len(s_out)}) must match!")

    if mode_factors is None:
        mode_factors_out = np.ones((n_modes,), dtype=float) / n_modes
    else:
        mode_factors_out = np.atleast_1d(np.array(mode_factors, dtype=float)).copy()
        if len(mode_factors_out) != n_modes:
            raise ValueError(f"mode_factors({len(mode_factors_out)}) must have length {n_modes}!")

    if not np.isclose(mode_factors_out.sum(), 1.0):
        print(f"Warning!! mode_factors sums to {mode_factors_out.sum()} != 1")

    return r_med_out, s_out, mode_factors_out


def get_nccn_over_mcon(  # pylint: disable=too-many-arguments
    dens: float,
    rmed_in: Union[float, np.ndarray],
    shape_in: Union[float, np.ndarray],
    mode_factors_in: Optional[Union[float, np.ndarray]] = None,
    rmin_n: float = 0.0,
    rmax_n: float = float(np.inf),
    rmin_m: float = 0.0,
    rmax_m: float = float(np.inf),
) -> Union[float, np.ndarray]:
    """Compute number-per-mass conversion factor for a (multi)lognormal distribution.

    Returns
    -------
    float
        `nd` with units cm^-3 kg^-1 m^3 (consistent with the project’s usage).
    """
    r_med, shape, mode_factors = preprocess_params(rmed_in, shape_in, mode_factors_in)
    dens_arr = np.array(dens, dtype=float)

    # Fraction of number between rmin_n and rmax_n (normalized if full range)
    if rmin_n > 0 or rmax_n < np.inf:
        numer = lognorm_cdf(rmax_n, r_med, shape, mode_factors) - lognorm_cdf(
            rmin_n, r_med, shape, mode_factors
        )
    else:
        numer = 1.0

    # 3rd moment between rmin_m and rmax_m (analytical if full range)
    if rmin_m > 0 or rmax_m < np.inf:
        integrals = lognorm_x3_cdf(rmax_m, r_med, shape, mode_factors) -\
            lognorm_x3_cdf(rmin_m, r_med, shape, mode_factors)
    else:
        integrals = np.array(
            [np.exp(3 * np.log(r) + 9 / 2 * (np.log(s) ** 2)) * f
             for r, s, f in zip(r_med, shape, mode_factors)]
        ).sum()

    denom = (4 / 3 * np.pi * dens_arr) * integrals
    return 1.0e12 * numer / denom


def lognorm_pdf(
    x_in: Union[float, np.ndarray],
    r_med_in: Union[float, np.ndarray],
    shape_in: Union[float, np.ndarray],
    mode_factors_in: Optional[Union[float, np.ndarray]] = None,
) -> np.ndarray:
    """Normalized lognormal PDF."""
    r_med, shape, mode_factors = preprocess_params(r_med_in, shape_in, mode_factors_in)

    x = np.array(x_in, dtype=float)
    si = np.log(shape)  # sigma

    if x.shape and x.size > 1:
        x = x[None, :]
        r_med = r_med[:, None]
        si = si[:, None]
        mode_factors = mode_factors[:, None]

    with np.errstate(divide="ignore"):
        x_arg = x / r_med
        norm_fact = 1.0 / (np.sqrt(2 * np.pi) * si * x)
        return (norm_fact * np.exp(-(np.log(x_arg) ** 2) / (2 * (si ** 2))) * mode_factors).sum(axis=0)


def lognorm_x3_pdf(
    x: Union[float, np.ndarray], r_med: np.ndarray, shape: np.ndarray,
    mode_factors: Optional[np.ndarray] = None
    ) -> np.ndarray:
    """Return x^3 * n(x), where n is the normalized lognormal PDF."""
    return np.asarray(x) ** 3 * lognorm_pdf(x, r_med, shape, mode_factors)


def lognorm_cdf(
    x_in: Union[float, np.ndarray],
    r_med_in: Union[float, np.ndarray],
    shape_in: Union[float, np.ndarray],
    mode_factors_in: Optional[Union[float, np.ndarray]] = None
    ) -> np.ndarray:
    """CDF of the normalized lognormal PDF."""

    r_med, shape, mode_factors = preprocess_params(r_med_in, shape_in, mode_factors_in)

    x = np.array(x_in, dtype=float)
    si = np.log(shape)

    if x.shape and x.size > 1:
        x = x[None, :]
        r_med = r_med[:, None]
        si = si[:, None]
        mode_factors = mode_factors[:, None]

    with np.errstate(divide="ignore"):
        x_arg = x / r_med
        return (0.5 * (1 + erf(np.log(x_arg) / (si * np.sqrt(2)))) * mode_factors).sum(axis=0)


def lognorm_x3_cdf(
    x_in: Union[float, np.ndarray], r_med: np.ndarray, shape: np.ndarray,
    mode_factors: Optional[np.ndarray] = None
    ) -> np.ndarray:
    """Integral of x^3 * n(x) from 0 to x, where n(x) is the normalized lognormal PDF."""

    def _integral(xsup: float) -> float:
        return quad(lognorm_x3_pdf, 0.0, xsup, args=(r_med, shape, mode_factors))[0]

    x = np.array(x_in, dtype=float)
    if x.shape and x.size > 1:
        return np.array([_integral(float(xsup)) for xsup in x], dtype=float)
    return np.array(_integral(float(x)), dtype=float)


def get_nccn_over_mcon_from_specs(
    species: IFSAeroSpecs,
    rmin_n: float = 0.0,
    rmax_n: float = float(np.inf),
    rmin_m: float = 0.0,
    rmax_m: float = float(np.inf),
) -> float:
    """Convenience wrapper using an `IFSAeroSpecs` instance."""
    rmin_n = float(np.clip(rmin_n, species.rmin, species.rmax))
    rmax_n = float(np.clip(rmax_n, species.rmin, species.rmax))
    rmin_m = float(np.clip(rmin_m, species.rmin, species.rmax))
    rmax_m = float(np.clip(rmax_m, species.rmin, species.rmax))

    if rmin_n == rmax_n:
        return 0.0
    if rmin_m == rmax_m:
        return float(np.inf)

    return np.sum(get_nccn_over_mcon(species.density, species.median, species.shape,
                              species.fact, rmin_n, rmax_n, rmin_m, rmax_m))


def get_nccn_over_mcon_from_speclist(specs: Dict[str, IFSAeroSpecs]) -> Dict[str, float]:
    """Vectorized wrapper for a dictionary of species specs."""
    return {name: get_nccn_over_mcon_from_specs(aerospec) for name, aerospec in specs.items()}


def compute_ccn_ifs(ws: xr.DataArray, lsm: xr.DataArray) -> xr.DataArray:
    """IFS-style CCN parameterization as a function of 10 m wind speed and land/sea mask.

    Parameters
    ----------
    ws : xr.DataArray
        Absolute 10 m wind speed.
    lsm : xr.DataArray
        Land-sea mask.

    Returns
    -------
    xr.DataArray
        Estimated Nd.
    """
    landmask = lsm > 0.5
    wind_lt15 = ws < 15

    a_par = xr.where(wind_lt15, 0.16, 0.13)
    b_par = xr.where(wind_lt15, 1.45, 1.89)
    qa = np.exp(a_par * ws + b_par).clip(-np.inf, 327)

    c_par = xr.where(landmask, 2.21, 1.2)
    d_par = xr.where(landmask, 0.3, 0.5)

    na = 10 ** (c_par + d_par * np.log10(qa))
    nd = xr.where(
        landmask,
        -2.10e-4 * (na ** 2) + 0.568 * na - 27.9,
        -1.15e-3 * (na ** 2) + 0.963 * na + 5.30,
    )
    return nd