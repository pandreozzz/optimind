import ctypes as ct

import numpy as np
import xarray as xr

from ..main import config
from ..main.config import CONFIGDICT
from .stack import tools_to_stack_xarrays


f_lib = ct.CDLL(config.VERTINTERP_LIB)
c_real = ct.c_double
c_real_ptr = np.ctypeslib.ndpointer(c_real)#ct.POINTER(c_real)
c_int  = ct.c_int32
c_int_ptr = np.ctypeslib.ndpointer(c_int)#ct.POINTER(c_int)

f_lib.interp.argtypes = [c_real_ptr,]*2+[c_int_ptr, c_real_ptr]+[c_int,]*6
f_lib.interp_fld.argtypes = [c_real_ptr,]*2+[c_int_ptr, c_real_ptr]+[c_int,]*6

def interp_vertical(psrc, ptgt, chunk_size_max=1000):
    """
        Interpolates psrc to ptgt
        psrc(ncom,nsrc,nlevsrc)
        ptgt(ncom,ntgt,nlevtgt)

        returns
        tgtlevs(ncom,nsrc,ntgt,nlevtgt), weights(ncom,nsrc,ntgt,nlevtgt)
    """
    ncom,ntgt,nlevtgt  = ptgt.shape
    ncom2,nsrc,nlevsrc = psrc.shape


    if (ncom != ncom2):
        raise ValueError(
                "Different common dimension size between source and target!"+\
                f"({ncom2},{nsrc}) and ({ncom},{nsrc})")


    outshape = (ncom,nsrc,ntgt,nlevtgt)

    tgtlevs = np.zeros(outshape, dtype=c_int)
    weights = np.zeros(outshape, dtype=c_real)

    # Enforce contiguity
    psrc_cont = np.ascontiguousarray(psrc)
    ptgt_cont = np.ascontiguousarray(ptgt)
    tgtlevs = np.ascontiguousarray(tgtlevs)
    weights = np.ascontiguousarray(weights)

    f_lib.interp(
        psrc_cont.astype(c_real), ptgt_cont.astype(c_real),
        tgtlevs, weights,
        c_int(ncom), c_int(nsrc), c_int(ntgt),
        c_int(nlevsrc), c_int(nlevtgt), c_int(chunk_size_max)
        )
    del psrc_cont, ptgt_cont

    return tgtlevs, weights

def interp_fld_vertical(fsrc, tgtlevs, weights, chunk_size_max=1000):
    """
        Interpolates fields according to weights
        fsrc(ncom,nsrc,nlevsrc)
        tgtlevs(ncom,ntgt,nlevtgt)
        weights(ncom,ntgt,nlevtgt)

        returns
        fdst(ncom,nsrc,ntgt,nlevtgt)
    """
    ncom,nsrc,nlevsrc = fsrc.shape

    ncom2,ntgt,nlevtgt   = tgtlevs.shape
    ncom3,ntgt2,nlevtgt2 = weights.shape

    if (ncom != ncom2) or (ncom2 != ncom3) or (ntgt != ntgt2) or (nlevtgt != nlevtgt2):
        raise ValueError(
                "Some dimensions are incompatible!!"+\
                f"fsrc(ncom={ncom},nsrc={nsrc},nlevsrc{nlevsrc})"+\
                f"tgtlevs(ncom={ncom2},ntgt={ntgt},nlevtgt{nlevtgt})"+\
                f"tgtlevs(ncom={ncom3},ntgt={ntgt2},nlevtgt{nlevtgt2})")

    fdstshape = (ncom,nsrc,ntgt,nlevtgt)
    fdst = np.zeros(fdstshape, dtype=c_real)

    # Enforce contiguity
    fsrc_cont = np.ascontiguousarray(fsrc)
    fdst = np.ascontiguousarray(fdst)
    tgtlevs_cont = np.ascontiguousarray(tgtlevs)
    weights_cont = np.ascontiguousarray(weights)

    f_lib.interp_fld(
            fsrc_cont.astype(c_real),
            fdst,
            tgtlevs_cont.astype(c_int),
            weights_cont.astype(c_real),
            c_int(ncom), c_int(nsrc), c_int(ntgt),
            c_int(nlevsrc), c_int(nlevtgt), c_int(chunk_size_max)
            )
    del fsrc_cont, tgtlevs_cont, weights_cont

    return fdst

#############################################
# INTERPOLATION OF MONTHLY CLIMATOLOGIES

def interpolate_monthly_clim(dset,
                             dates : xr.DataArray):
    """
    Perform time interpolation for monthly climatology

    Args:
        dset: xr.Dataset or xr.DataArray - containing "month" dimension
        dates: xr.DataArray - dates to which interpolate
    Returns:
        interpolated: same type as dset
    """

    prev_month     = (dates.values - np.timedelta64(14,'D')).astype("datetime64[M]")+ np.timedelta64(14,'D')
    foll_month     = (prev_month + np.timedelta64(18,'D')).astype("datetime64[M]") + np.timedelta64(14,'D')

    monthdelta     = foll_month - prev_month
    thisdelta      = dates  - prev_month.astype("datetime64[ns]")

    timeweight     = thisdelta/monthdelta # in [0,1]

    intmonths_bot  = xr.DataArray(data=prev_month.astype("datetime64[ns]"), coords={"time":dates}).dt.month
    intmonths_top  = xr.DataArray(data=foll_month.astype("datetime64[ns]"), coords={"time":dates}).dt.month

    return  (1-timeweight) * dset.sel(month=intmonths_bot).drop_vars("month") +\
        timeweight*dset.sel(month=intmonths_top).drop_vars("month")

def interpolate_aero(aero_fields, dst_pres, aero_timeinterp=True,
                     intp_dim_name="lev") -> xr.Dataset:
    def find_dim_name(basename:str, current_dims : list):
        for i in range(len(current_dims)+2):
            dim_name = f"{basename}_{i}"
            if dim_name not in current_dims:
                return dim_name

    print("Computing aero vertical interpolation weights...")
    aux_dimname2 = find_dim_name("tmp_aux", current_dims = list(dst_pres.dims)+list(aero_fields.dims))

    tmp_src = aero_fields["pressure"]
    tmp_dst = dst_pres

    if aero_timeinterp:
        unique_dates_arr = np.unique(tmp_dst.time.dt.date)
        unique_dates = xr.DataArray(data=unique_dates_arr,dims=["time"])
        tmp_src = interpolate_monthly_clim(tmp_src, unique_dates.astype("datetime64[ns]")).compute()
        aux_datedim = find_dim_name("tmp_aux_date", current_dims = list(dst_pres.dims)+list(aero_fields.dims))
    else:
        aux_datedim = ""
        unique_dates_arr = [None]

    tmp_tgtlevs_list = []
    tmp_weights_list = []

    tmp_stacktools = None

    for date in unique_dates_arr:

        tmp_timeslice = slice(str(date),str(date)) if date is not None else tmp_dst.time.values

        if "time" in tmp_src.dims:
            this_tmp_src = tmp_src.sel(time=tmp_timeslice)
            if aero_timeinterp:
                this_tmp_src = this_tmp_src.rename(time=aux_datedim)
        else:
            this_tmp_src = tmp_src

        this_tmp_dst = tmp_dst.sel(time=tmp_timeslice)
        #print(this_tmp_src.__str__())
        #print(this_tmp_dst.__str__(), flush=True)

        tmp_stacktools = tools_to_stack_xarrays(src_arr=this_tmp_src,
                                                dst_arr=this_tmp_dst,
                                                intp_dim_name=intp_dim_name)

        tgtlevs_aero, weights_aero = interp_vertical(
            psrc=this_tmp_src.transpose(*tmp_stacktools.src_dim_order).values.reshape(tmp_stacktools.src_stackshape),
            ptgt=this_tmp_dst.sel(time=tmp_timeslice).transpose(*tmp_stacktools.dst_dim_order).values.reshape(tmp_stacktools.dst_stackshape)
        )
        del this_tmp_src, this_tmp_dst

        tgtlevs_aero = xr.DataArray(data=tgtlevs_aero.reshape(tmp_stacktools.out_shape),
                               dims=tmp_stacktools.out_dim_order, coords=tmp_stacktools.out_coords)
        weights_aero = xr.DataArray(data=weights_aero.reshape(tmp_stacktools.out_shape),
                               dims=tmp_stacktools.out_dim_order, coords=tmp_stacktools.out_coords)
        if aero_timeinterp:
            tgtlevs_aero = tgtlevs_aero.squeeze(aux_datedim, drop=True)
            weights_aero = weights_aero.squeeze(aux_datedim, drop=True)

        tmp_tgtlevs_list.append(tgtlevs_aero)
        tmp_weights_list.append(weights_aero)

        del tgtlevs_aero, weights_aero

    tgtlevs_aero = xr.concat(tmp_tgtlevs_list, dim="time")
    weights_aero = xr.concat(tmp_weights_list, dim="time")

    print("Done!")

    del tmp_src, tmp_dst, tmp_stacktools

    specs = [f for f in aero_fields if f != "pressure"]
    print(f"The following species will be interpolated: {specs}")
    interpolaeros = []
    for spec in specs:
        print(f"Interpolating {spec}...", flush=True)
        interpospecs = []

        # Aero timeinterp
        if aero_timeinterp:
            tmp_src = interpolate_monthly_clim(
                aero_fields[spec], unique_dates.astype("datetime64[ns]")
            ).compute()
        else:
            tmp_src = aero_fields[spec]

        tmp_dst = tgtlevs_aero

        for date in unique_dates_arr:
            tmp_timeslice = slice(str(date),str(date)) if date is not None else tmp_dst.time.values

            if "time" in tmp_src.dims:
                this_tmp_src = tmp_src.sel(time=tmp_timeslice)
                if aero_timeinterp:
                    this_tmp_src = this_tmp_src.rename(time=aux_datedim)
            else:
                this_tmp_src = tmp_src

            this_tgtlevs_aero = tgtlevs_aero.sel(time=tmp_timeslice)
            this_weights_aero = weights_aero.sel(time=tmp_timeslice)

            tmp_stacktools = tools_to_stack_xarrays(src_arr=this_tmp_src, dst_arr=this_tgtlevs_aero,
                                                 intp_dim_name=intp_dim_name)

            this_aero_field_intp = interp_fld_vertical(
                fsrc=this_tmp_src.transpose(*tmp_stacktools.src_dim_order).values.reshape(tmp_stacktools.src_stackshape),
                tgtlevs=this_tgtlevs_aero.transpose(*tmp_stacktools.dst_dim_order).values.reshape(tmp_stacktools.dst_stackshape),
                weights=this_weights_aero.transpose(*tmp_stacktools.dst_dim_order).values.reshape(tmp_stacktools.dst_stackshape)
            )
            del this_tmp_src, this_tgtlevs_aero, this_weights_aero,

            this_aero_field_intp = xr.DataArray(data=this_aero_field_intp.reshape(tmp_stacktools.out_shape).astype(np.float32),
                             dims=tmp_stacktools.out_dim_order, coords=tmp_stacktools.out_coords
                            )
            if aero_timeinterp:
                this_aero_field_intp = this_aero_field_intp.squeeze(aux_datedim, drop=True)

            interpospecs.append(this_aero_field_intp)

            del this_aero_field_intp, tmp_stacktools

        del tmp_src

        interpolaeros.append(xr.concat(interpospecs, dim="time").rename(spec))
        del interpospecs

    return xr.merge(interpolaeros)

def get_interpolated_ccn(this_ccn_mmr, this_ifs, this_ifs_fixedlevel, actual_recipe):

    if (CONFIGDICT["aeros_out_of_cloud"] is None) and config.SOME_AEROS_OUT_OF_CLOUD:
        aeros_out_of_cloud = [a for a in actual_recipe]
    else:
        aeros_out_of_cloud = CONFIGDICT["aeros_out_of_cloud"]

    if config.SOME_AEROS_OUT_OF_CLOUD:
        print("Picking aerosols at out-of-cloud level")
        if aeros_out_of_cloud is None:
            print("All aerosols are picked at the fixed or below-cloud level!")
            aeros_out_of_cloud = [a for a in actual_recipe]

        out_of_cloud_ccn = [a for a in actual_recipe if a in aeros_out_of_cloud]
        in_cloud_ccn = [a for a in actual_recipe if a not in aeros_out_of_cloud]
        if len(out_of_cloud_ccn) > 0:
            this_ccn_mcon = interpolate_aero(
                this_ccn_mmr[["pressure"]+out_of_cloud_ccn],
                this_ifs_fixedlevel["p"],
                aero_timeinterp=CONFIGDICT["aerofromclimatology"]
            )
            dimorder = [d for d in this_ifs_fixedlevel["p"].dims
                        if d in this_ccn_mcon.dims]
            this_ccn_mcon = this_ccn_mcon.transpose(*dimorder)
        else:
            this_ccn_mcon = xr.Dataset(None)
    else:
        out_of_cloud_ccn = []
        in_cloud_ccn = list(actual_recipe.keys())
        this_ccn_mcon = xr.Dataset(None)

    # Aerosols at cloud level
    if len(in_cloud_ccn) > 0:
        print(f"Picking aerosols at cloud level: {in_cloud_ccn}")
        print(this_ccn_mmr.__str__(), flush=True)
        this_ccn_mcon.update(interpolate_aero(
            this_ccn_mmr[["pressure"]+in_cloud_ccn],
            this_ifs["p"],
            aero_timeinterp=CONFIGDICT["aerofromclimatology"]
        ))
        dimorder = [d for d in this_ifs["p"].dims
                    if d in this_ccn_mcon.dims]
        this_ccn_mcon = this_ccn_mcon.transpose(*dimorder)

    return this_ccn_mcon[in_cloud_ccn+out_of_cloud_ccn]
