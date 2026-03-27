"""Collection of masks for selection of cloud fields"""

import numpy as np

from ..main.config import CONFIGDICT
from ..tools.eot import Irradiance

def get_cos_sza_mask(time, lat, lon, cos_sza_minmax):
    """
    Compute solar zenith angle (SZA) mask for daytime filtering.

    Calculates cosine of solar zenith angle and creates a boolean mask
    for observations within configured daytime SZA bounds.

    Args:
        time: datetime or datetime-like - Time for SZA calculation
        lat: array-like - Latitude coordinates in degrees
        lon: array-like - Longitude coordinates in degrees

    Returns:
        array: Boolean mask (True where cos(SZA) is in configured range)

    Note:
        SZA bounds are specified by CONFIGDICT['cos_sza_minmax']
    """
    cos_sza_min = np.float32(max(min(min(cos_sza_minmax), 1), 0))
    cos_sza_max = np.float32(max(min(max(cos_sza_minmax), 1), 0))
    print("Restricting local cosine of SZA between " +
          f"{cos_sza_min:.2f} and {cos_sza_max:.2f}", flush=True)

    MyIrr = Irradiance(date=time, ifs_like_2pi=False)
    this_cos_sza = MyIrr.mu0_cos_sza_deg(phi=lat, lam=lon)
    cos_sza_mask = (this_cos_sza >= cos_sza_min) & (this_cos_sza <= cos_sza_max)

    return cos_sza_mask


def get_localtime_mask(time, lon, localhour_minmax):
    """
    Create local time mask for filtering observations by local solar time.

    Converts UTC time to local solar time and creates a boolean mask for
    observations within configured local hour range.

    Args:
        time: datetime or datetime-like - UTC time coordinate
        lon: array-like - Longitude coordinates in degrees (-180 to 180)

    Returns:
        array: Boolean mask (True where local time is in configured range)

    Note:
        Local hour range is specified by CONFIGDICT['localhour_minmax']
    """
    lo = min(localhour_minmax)
    hi = max(localhour_minmax)
    hour_min = np.float32(min(max(lo, 0), 24))
    hour_max = np.float32(min(max(hi, 0), 24))
    print(f"Restricting local time between {hour_min} and {hour_max}", flush=True)

    exact_local_time = time + ((lon.where(lon <= 180, lon-360) - 180)/360 + 0.5) *\
      np.timedelta64(1, 'D').astype("timedelta64[ns]")
    localtime_h = exact_local_time.dt.hour + exact_local_time.dt.minute / 60
    # semi-closed interval to avoid (honestly small) artifacts in time counting
    localtime_mask = (localtime_h >= hour_min) & (localtime_h < hour_max)

    return localtime_mask
