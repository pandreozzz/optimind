import os
import numpy  as np
import xarray as xr
# IFS data

# RDAY
DAYSECS = 86400
# days in 1 year
YEADAYS = 365.25

# mean earth-sun distance
REA     = 1.

# polar axis tilting
REPSM   = 0.409093

# solar irradiance dset
#IFSIRRVERS = "47r1" # or "49r1"
#solar_irr_path  ~ "<path_to_ifs_files>/total_solar_irradiance_CMIP6_{IFSIRRVERS}.nc"


# Class Irradiance
class Irradiance:
    def __init__(self, date : np.ndarray,
                 delay_s : float = 0., ifs_like_2pi : bool = True,
                 ignore_eot : bool = False, skip_irr : bool = False,
                 solar_irr_path = None):
        """
        Class to reproduce astronomical computations in the IFS
        """
        # Ensure date has legal type
        _date_is_xarray = isinstance(date, xr.DataArray)
        _date_is_ndarray = isinstance(date, np.ndarray)
        _date_is_scalar = isinstance(date, np.datetime64)

        if not (_date_is_xarray or _date_is_ndarray or _date_is_scalar):
            raise ValueError("date must be of type np.datetime64 or np.ndarray or xr.DataArray!")
        self.date_type = type(date)

        _year_type = "datetime64[Y]"
        _day_type = "datetime64[D]"
        _float_type = np.float32

        def _year_fraction(self) -> float:
            """ Computes the fraction of the yearly Earth's orbit
            around the sun from the start of the year to date
            RTETA
            """
            if _date_is_xarray:
                this_yr = self.date.values.astype(_year_type)
            else:
                this_yr        = self.date.astype(_year_type)

            return (self.date - this_yr)/(np.timedelta64(24, "h")*YEADAYS)

        def _day_fraction(self) -> float:
            """ Computes the fraction of the day at the current date
            """

            if _date_is_xarray:
                today = self.date.values.astype(_day_type)
            else:
                today = self.date.astype(_day_type)
            return (self.date - today)/np.timedelta64(24, "h")

        def _get_solar_irr(self, solar_irr_path) -> float:
            sun_irr_dset = xr.open_dataset(solar_irr_path, decode_times=False)["tsi"]

            if _date_is_xarray:
                this_yr = self.date.dt.year.values
            else:
                this_yr      = self.date.astype(_year_type).astype(int)+1970

            now_yr       = this_yr + self.year_fraction
            indexer_shape = now_yr.shape

            if _date_is_ndarray:
                now_yr = now_yr.flatten(order='C')

            # Cannot get values out of dataset bound
            min_dset_yr = sun_irr_dset.time.min().values
            max_dset_yr = sun_irr_dset.time.max().values
            now_yr_clipped = np.clip(now_yr, min_dset_yr, max_dset_yr)
            if np.any(now_yr != now_yr_clipped):
                print("Warning! some dates are outside of the domain" \
                      "covered by the solar irradiance dataset!" \
                      f"year span: {min_dset_yr:.1f}->{max_dset_yr:.1f}")

            sun_irr      = sun_irr_dset.interp(time=now_yr_clipped,
                                               method="linear").astype(_float_type)

            if _date_is_ndarray:
                sun_irr = sun_irr.values.reshape(indexer_shape, order='C')

            return sun_irr*(1/self.earth_sun_dist_frac**2)


        def _orbit_theta(self) -> float:
            """ Computes the angle of the yearly
            Earth's orbit around the sun to date (no precession of equinoxes)
            REL reference
            """
            two_pi = 6.283076 if ifs_like_2pi else 2*np.pi

            return 1.7535 + two_pi*self.year_fraction

        def _orbit_theta_rem(self) -> float:
            """ Computes the  angle of the yearly
            Earth's orbit around the sun to date (no precession of equinoxes)
            REM reference
            """
            two_pi = 6.283020 if ifs_like_2pi else 2*np.pi

            return 6.240075 + two_pi*self.year_fraction

        def _earth_sun_dist_frac(self) -> float:
            """ Computes the distance between the Earth and the sun
            for the current date
            date : np.datetime64 or np.ndarray of np.datetime64
            RRS reference as a fraction of REA (earth-sun distance mean)
            """
            return 1.0001 - 0.0163*np.sin(self.orbit_theta) + 0.0037*np.cos(self.orbit_theta)

        # RDS equivalent to eq. 3.7 below
        def _sun_declination(self) -> float:
            """ Computes the sun declination for the current date
            date : np.datetime64 or np.ndarray of np.datetime64
            """
            two_pi = 6.283320 if ifs_like_2pi else 2*np.pi
            relative_rllls = 4.8952 + two_pi*self.year_fraction - 0.0075*np.sin(self.orbit_theta) +\
                -0.0326*np.cos(self.orbit_theta) - 0.0003*np.sin(2*self.orbit_theta)              +\
                +0.0002*np.cos(2*self.orbit_theta)
            return np.arcsin(np.sin(REPSM)*np.sin(relative_rllls))

        def _equation_of_time_s(self) -> float:
            """ Equation of time for the current date (in seconds)
            date : np.datetime64 or np.ndarray of np.datetime64
            """
            two_pi = 6.283076 if ifs_like_2pi else 2*np.pi
            relative_rlls = 4.8951 + two_pi*self.year_fraction
            orbit_theta_rem = _orbit_theta_rem(self)

            sin_rem       = np.sin(orbit_theta_rem)
            return 591.8*np.sin(2*relative_rlls) - 459.4*sin_rem +\
                +39.5*sin_rem*np.cos(2*relative_rlls)            +\
                -12.7*np.sin(4*relative_rlls) - 4.8*np.sin(2*orbit_theta_rem)

        # Set date
        self.date            = date - np.timedelta64(1,"s")*delay_s

        # Fill astronomical data
        self.year_fraction   = _year_fraction(self)
        self.day_fraction    = _day_fraction(self)

        # Require self.year_fraction
        self.orbit_theta     = _orbit_theta(self)

        # Require self.orbit_theta
        self.earth_sun_dist_frac = _earth_sun_dist_frac(self)
        self.earth_sun_dist_m    = REA*self.earth_sun_dist_frac
        self.sun_declination_rad = _sun_declination(self)
        self.sun_declination_deg = np.rad2deg(self.sun_declination_rad)

        # Require self.orbit_theta_rem (option to ignore equation of time)
        self.eq_of_time_s    = 0. if ignore_eot else _equation_of_time_s(self)

        # Require year_fraction and earth_sun_dist_frac
        if solar_irr_path is not None:
            self.solar_irr = _get_solar_irr(self, solar_irr_path)
        else:
            self.solar_irr = None

    def solar_time(self) -> float:
        """ Computes the solar time at the current date in radians
        date : np.datetime64 or np.ndarray of np.datetime64
        """
        return 2*np.pi*(self.eq_of_time_s/DAYSECS + self.day_fraction)

    def solar_coords_rad(self):
        """ Computes sun position in the Earth's coordinates
        returns lat, lon in radians
        """

        return self.sun_declination_rad, self.solar_time()

    def solar_coords_deg(self):
        """ Computes sun position in the Earth's coordinates
        returns lat, lon in degrees
        """
        lat_rad, lon_rad = self.solar_coords_rad()

        return np.rad2deg(lat_rad), np.rad2deg(lon_rad)


    def solar_angles(self, phi, lam):
        """ After https://doi.org/10.1016/j.renene.2021.03.047
            phi   : float or np.ndarray latitude in radians
            lam   : float or np.ndarray longitude in radians
        """
        # subsolar point longitude
        lams = np.mod(-2*np.pi*(self.eq_of_time_s/86400 + self.day_fraction + 0.5), 2*np.pi)
        # subsolar point latitude = declination
        phis = self.sun_declination_rad
        Sx = np.cos(phis)*np.sin(lams - lam)
        Sy = np.cos(phi)*np.sin(phis) - np.sin(phi)*np.cos(phis)*np.cos(lams - lam)
        Sz = np.sin(phi)*np.sin(phis) + np.cos(phi)*np.cos(phis)*np.cos(lams - lam)

        sza = np.arccos(Sz)
        saa = np.arctan2(Sx, Sy)

        return sza, saa

    def mu0_cos_sza_rad(self, phi : float, lam : float, zamu0 : bool = False) -> float:
        """ Computes the cosine of solar zenith angle at the current date
        date  : np.datetime64 or np.ndarray of np.datetime64
        phi   : float or np.ndarray latitude in radians
        lam   : float or np.ndarray longitude in radians
        zamu0 : bool set to true to simulate IFS input to ecrad
        """

        decl       = self.sun_declination_rad
        h_angle    = self.solar_time() + lam + np.pi
        mu0        = np.sin(decl)*np.sin(phi) + \
                    + np.cos(decl)*np.cos(phi)*np.cos(h_angle)
        if zamu0:
            rrae = 0.1277*1.e-2
            zcrae = rrae*(rrae+2)
            mu0[...] = np.where(mu0>1.e-10, rrae/(np.sqrt(mu0**2+zcrae)-mu0), rrae/np.sqrt(zcrae))

        return np.clip(mu0, 0., 1.)

    def mu0_cos_sza_deg(self, phi : float, lam : float, zamu0 : bool = False) -> float:
        """ Computes the cosine of solar zenith angle at the current date
        date : np.datetime64 or np.ndarray of np.datetime64
        phi  : float or np.ndarray latitude in degrees
        lam  : float or np.ndarray longitude in degrees
        zamu0 : bool set to true to simulate IFS input to ecrad
        """

        return self.mu0_cos_sza_rad(phi=np.deg2rad(phi), lam=np.deg2rad(lam), zamu0=zamu0)

    def zenith_rad(self, phi : float, lam : float) -> float:
        """ Computes the zenith distance theta
        at the current date (in radians)
        date : np.datetime64 or np.ndarray of np.datetime64
        phi  : float or np.ndarray latitude in radians
        lam  : float or np.ndarray longitude in radians
        """

        return np.arccos(self.mu0_cos_sza_deg(phi, lam))


    def zenith_deg(self, phi : float, lam : float) -> float:
        """ Computes the zenith distance theta
        at the current date (in radians)
        date : np.datetime64 or np.ndarray of np.datetime64
        phi  : float or np.ndarray latitude in degrees
        lam  : float or np.ndarray longitude in degres
        """

        return self.zenith_rad(np.deg2rad(phi), np.deg2rad(lam))

    def azimuth_rad(self, phi : float, lam : float) -> float:
        """ Computes the azimuth  phi
        at the current date (in radians)
        date : np.datetime64 or np.ndarray of np.datetime64
        phi  : float or np.ndarray latitude in radians
        lam  : float or np.ndarray longitude in radians
        """

        return np.arccos(self.mu0_cos_sza_deg(phi, lam))
