"""Astronomical (IFS-like) utilities for solar geometry and irradiance."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr

# IFS-style constants and defaults
DAYSECS: int = 86_400
YEADAYS: float = 365.25
REA: float = 1.0
REPSM: float = 0.409093  # polar axis tilting (radians)

__all__ = ["Irradiance"]


NumberOrArray = Union[float, np.ndarray, xr.DataArray]


class Irradiance:
    """Reproduce IFS astronomical computations (solar time, declination, SZA, etc.).

    Parameters
    ----------
    date : np.datetime64 | np.ndarray | xr.DataArray
        Scalar or array-like date(s).
    delay_s : float, optional
        Time delay (seconds) applied to the input date.
    ifs_like_2pi : bool, optional
        Use IFS numerical constants to approximate 2*pi.
    ignore_eot : bool, optional
        If True, ignore the Equation Of Time (EOT).
    skip_irr : bool, optional
        Reserved (no-op): keeps parity with legacy code.
    solar_irr_path : str or None, optional
        Path to a CMIP6 *total solar irradiance* NetCDF (`tsi`) to sample from.

    Notes
    -----
    This class stores intermediate astronomical quantities as attributes for reuse.
    """

    # pylint: disable=too-many-arguments, too-many-instance-attributes

    def __init__(
        self,
        date: NumberOrArray,
        delay_s: float = 0.0,
        ifs_like_2pi: bool = True,
        ignore_eot: bool = False,
        skip_irr: bool = False,  # kept for API parity
        solar_irr_path: Optional[str] = None,
    ) -> None:
        # Validate type
        if not isinstance(date, (np.datetime64, np.ndarray, xr.DataArray)):
            raise ValueError("date must be np.datetime64, np.ndarray, or xr.DataArray")

        self.date_type = type(date)
        self.ifs_like_2pi = bool(ifs_like_2pi)
        self.ignore_eot = bool(ignore_eot)

        # Apply delay
        self.date = date - np.timedelta64(1, "s") * int(delay_s)

        # Core quantities
        self.year_fraction = self._year_fraction()
        self.day_fraction = self._day_fraction()
        self.orbit_theta = self._orbit_theta()
        self.earth_sun_dist_frac = self._earth_sun_dist_frac()
        self.earth_sun_dist_m = REA * self.earth_sun_dist_frac
        self.sun_declination_rad = self._sun_declination()
        self.sun_declination_deg = np.rad2deg(self.sun_declination_rad)
        self.eq_of_time_s = 0.0 if self.ignore_eot else self._equation_of_time_s()

        # Solar irradiance (optional)
        self.solar_irr = self._get_solar_irr(solar_irr_path) if solar_irr_path else None

    # -----------------------------
    # Private helpers (former nested defs)
    # -----------------------------

    def _two_pi(self, for_rel: str) -> float:
        """IFS-like 2*pi constants vary slightly by formula in the legacy code."""
        if not self.ifs_like_2pi:
            return 2 * np.pi
        if for_rel == "theta":
            return 6.283076
        if for_rel == "theta_rem":
            return 6.283020
        if for_rel == "decl":
            return 6.283320
        # Default
        return 6.283185

    def _year_fraction(self) -> NumberOrArray:
        """Fraction of the year elapsed at `date`."""
        year_type = "datetime64[Y]"
        if isinstance(self.date, xr.DataArray):
            this_year = self.date.values.astype(year_type)
        else:
            this_year = np.asarray(self.date).astype(year_type)
        return (self.date - this_year) / (np.timedelta64(24, "h") * YEADAYS)

    def _day_fraction(self) -> NumberOrArray:
        """Fraction of the day elapsed at `date`."""
        day_type = "datetime64[D]"
        if isinstance(self.date, xr.DataArray):
            today = self.date.values.astype(day_type)
        else:
            today = np.asarray(self.date).astype(day_type)
        return (self.date - today) / np.timedelta64(24, "h")

    def _get_solar_irr(self, solar_irr_path: str) -> NumberOrArray:
        """Sample total solar irradiance (CMIP6 `tsi`) and scale by distance."""
        sun_irr_dset = xr.open_dataset(solar_irr_path, decode_times=False)["tsi"]

        if isinstance(self.date, xr.DataArray):
            this_yr = self.date.dt.year.values
        else:
            this_yr = np.asarray(self.date).astype("datetime64[Y]").astype(int) + 1970

        now_yr = this_yr + self.year_fraction

        indexer_shape = now_yr.shape
        if isinstance(self.date, np.ndarray):
            now_yr = now_yr.flatten(order="C")

        min_yr = sun_irr_dset.time.min().values
        max_yr = sun_irr_dset.time.max().values
        clipped = np.clip(now_yr, min_yr, max_yr)
        if np.any(now_yr != clipped):
            print(
                "Warning! some dates are outside the solar irradiance dataset "
                f"range: {min_yr:.1f}->{max_yr:.1f}"
            )

        sun_irr = sun_irr_dset.interp(time=clipped, method="linear").astype(np.float32)
        if isinstance(self.date, np.ndarray):
            sun_irr = sun_irr.values.reshape(indexer_shape, order="C")

        return sun_irr * (1.0 / (self.earth_sun_dist_frac ** 2))

    def _orbit_theta(self) -> NumberOrArray:
        """Earth's orbital angle (REL reference)."""
        return 1.7535 + self._two_pi("theta") * self.year_fraction

    def _orbit_theta_rem(self) -> NumberOrArray:
        """Earth's orbital angle (REM reference)."""
        return 6.240075 + self._two_pi("theta_rem") * self.year_fraction

    def _earth_sun_dist_frac(self) -> NumberOrArray:
        """Earth–Sun distance as fraction of mean distance REA (RRS reference)."""
        return 1.0001 - 0.0163 * np.sin(self.orbit_theta) + 0.0037 * np.cos(self.orbit_theta)

    def _sun_declination(self) -> NumberOrArray:
        """Solar declination (radians)."""
        rel = (
            4.8952
            + self._two_pi("decl") * self.year_fraction
            - 0.0075 * np.sin(self.orbit_theta)
            - 0.0326 * np.cos(self.orbit_theta)
            - 0.0003 * np.sin(2 * self.orbit_theta)
            + 0.0002 * np.cos(2 * self.orbit_theta)
        )
        return np.arcsin(np.sin(REPSM) * np.sin(rel))

    def _equation_of_time_s(self) -> NumberOrArray:
        """Equation of time (seconds)."""
        rel = 4.8951 + self._two_pi("theta") * self.year_fraction
        theta_rem = self._orbit_theta_rem()
        sin_rem = np.sin(theta_rem)
        return (
            591.8 * np.sin(2 * rel)
            - 459.4 * sin_rem
            + 39.5 * sin_rem * np.cos(2 * rel)
            - 12.7 * np.sin(4 * rel)
            - 4.8 * np.sin(2 * theta_rem)
        )

    # -----------------------------
    # Public API
    # -----------------------------

    def solar_time(self) -> NumberOrArray:
        """Solar time (radians)."""
        return 2 * np.pi * (self.eq_of_time_s / DAYSECS + self.day_fraction)

    def solar_coords_rad(self) -> Tuple[NumberOrArray, NumberOrArray]:
        """Sun position in Earth coordinates (lat, lon) in radians."""
        return self.sun_declination_rad, self.solar_time()

    def solar_coords_deg(self) -> Tuple[NumberOrArray, NumberOrArray]:
        """Sun position in Earth coordinates (lat, lon) in degrees."""
        lat_rad, lon_rad = self.solar_coords_rad()
        return np.rad2deg(lat_rad), np.rad2deg(lon_rad)

    def solar_angles(self, phi: NumberOrArray,
                     lam: NumberOrArray) -> Tuple[NumberOrArray, NumberOrArray]:
        """Solar zenith and azimuth angles from latitude/longitude (radians).

        Parameters
        ----------
        phi : float | np.ndarray
            Latitude (radians).
        lam : float | np.ndarray
            Longitude (radians).

        Returns
        -------
        (sza, saa) : tuple
            Solar zenith angle and solar azimuth angle (radians).
        """
        lams = np.mod(-2 * np.pi * (self.eq_of_time_s / \
                                    DAYSECS + self.day_fraction + 0.5), 2 * np.pi)
        phis = self.sun_declination_rad

        sx = np.cos(phis) * np.sin(lams - lam)
        sy = np.cos(phi) * np.sin(phis) - np.sin(phi) * np.cos(phis) * np.cos(lams - lam)
        sz = np.sin(phi) * np.sin(phis) + np.cos(phi) * np.cos(phis) * np.cos(lams - lam)

        sza = np.arccos(sz)
        saa = np.arctan2(sx, sy)
        return sza, saa

    def mu0_cos_sza_rad(self, phi: NumberOrArray, lam: NumberOrArray,
                        zamu0: bool = False) -> NumberOrArray:
        """Cosine of the solar zenith angle (SZA) at the current date (radians input)."""
        decl = self.sun_declination_rad
        h_angle = self.solar_time() + lam + np.pi
        mu0 = np.sin(decl) * np.sin(phi) + np.cos(decl) * np.cos(phi) * np.cos(h_angle)

        if zamu0:
            rrae = 0.1277 * 1.0e-2
            zcrae = rrae * (rrae + 2)
            # Support both scalar and array
            if np.isscalar(mu0):
                mu0 = rrae / (np.sqrt(mu0 ** 2 + zcrae) - mu0) \
                    if mu0 > 1.0e-10 else rrae / np.sqrt(zcrae)
            else:
                mu0 = np.where(
                    mu0 > 1.0e-10,
                    rrae / (np.sqrt(mu0 ** 2 + zcrae) - mu0),
                    rrae / np.sqrt(zcrae)
                    )
        return np.clip(mu0, 0.0, 1.0)

    def mu0_cos_sza_deg(self, phi: NumberOrArray, lam: NumberOrArray,
                        zamu0: bool = False) -> NumberOrArray:
        """Cosine of SZA (degrees input)."""
        return self.mu0_cos_sza_rad(np.deg2rad(phi), np.deg2rad(lam), zamu0=zamu0)

    def zenith_rad(self, phi: NumberOrArray, lam: NumberOrArray) -> NumberOrArray:
        """Zenith angle θ (radians input)."""
        return np.arccos(self.mu0_cos_sza_rad(phi, lam))

    def zenith_deg(self, phi: NumberOrArray, lam: NumberOrArray) -> NumberOrArray:
        """Zenith angle θ (degrees input)."""
        return self.zenith_rad(np.deg2rad(phi), np.deg2rad(lam))

    def azimuth_rad(self, phi: NumberOrArray, lam: NumberOrArray) -> NumberOrArray:
        """Azimuth angle φ (radians input)."""
        _, saa = self.solar_angles(phi, lam)
        return saa
