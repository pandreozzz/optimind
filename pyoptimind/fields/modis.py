"""Processing of MODIS Nd**(1/3) fields"""
import os
import numpy as np
import xarray as xr

from ..main.config import CONFIGDICT, TMPFLDDIR

def get_modis_data(year,
                   latmin : float = -90, latmax : float = 90,
                   lonwest : float = 0, loneast : float = 360) -> xr.Dataset:
    """
    Fetch modis data
    """

    #Ignore "Z18"
    samples = list(set(CONFIGDICT["samplespreads"])|set([CONFIGDICT["modisndrefsample"]]))
    modis_nd13 = xr.open_dataset(
                    os.path.join(
                        TMPFLDDIR,
                        f"modis_nd_nd13.monthlymeans_{year:4d}.AT.v1_{CONFIGDICT['gridspec']}.nc")
    ).transpose(..., "lat", "lon").sortby("lat", ascending=False)

    latslice = slice(latmax,latmin)

    # Load lon coordinate eagerly (tiny) to keep selection graph small.
    all_lons = modis_nd13["lon"].load()
    if lonwest < loneast:
        lonsel = all_lons.sel(lon=slice(lonwest, loneast))
    else:
        lonsel = xr.concat(
            [
                all_lons.where(all_lons >= lonwest, drop=True),
                all_lons.where(all_lons <= loneast, drop=True),
            ],
            dim="lon",
        )

    modis_nd13 = modis_nd13.sel(lat=latslice, lon=lonsel)

    modis_nd13_data = {}
    for sample in samples:
        biascorr = "_bcorr" if CONFIGDICT["modisndbiascorrection"] else ""
        modis_nd13_mean  = modis_nd13[f"Nd13_{sample}{biascorr}"]
        modis_nd13_valid = modis_nd13[f"Valid_{sample}"]

        modis_nd13_mean  = modis_nd13_mean.where(
            modis_nd13_valid > CONFIGDICT["modisndvalidthr"]
            ).sel(lat=latslice).compute()
        modis_nd13_valid = modis_nd13_valid.where(
            modis_nd13_valid > CONFIGDICT["modisndvalidthr"]
            ).sel(lat=latslice).compute()

        modis_nd13_data[f"Nd13_{sample}"]  = modis_nd13_mean
        modis_nd13_data[f"Valid_{sample}"] = modis_nd13_valid

    modis_nd13.close()

    return xr.Dataset(data_vars=modis_nd13_data)

def get_modis_errors(modis_nd13):
    """ Returns the cubic spread
    """

    this_modis_nd13 = modis_nd13[f"Nd13_{CONFIGDICT['modisndrefsample']}"]
    valid_mask = ~np.isnan(this_modis_nd13)

    this_modis_ensemble = xr.concat([modis_nd13[f"Nd13_{s}"].expand_dims("tmp_sampledim", 0)
                                     for s in CONFIGDICT["samplespreads"]], dim="tmp_sampledim")

    this_modis_mean = this_modis_ensemble.mean(dim="tmp_sampledim", skipna=True).where(valid_mask)

    ensemble_is_broken = np.isnan(this_modis_ensemble).sum(dim="tmp_sampledim")

    this_modis_val = this_modis_mean if CONFIGDICT["modisndusemean"] else this_modis_nd13

    this_modis_errorbar = xr.where(
        ensemble_is_broken,
        this_modis_val,
        (
            this_modis_ensemble.max(dim="tmp_sampledim") -\
            this_modis_ensemble.min(dim="tmp_sampledim")
        ).clip(1,1.e4)/2
    ).where(valid_mask)

    return this_modis_val, this_modis_errorbar
