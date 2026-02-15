"""Vertical interpolation helpers for aerosol fields and CCN mass mixing ratios.

This module provides:
- Thin ctypes wrappers over the compiled vertical interpolation routines
  (weights/indices and field application).
- High-level xarray helpers to compute weights once and apply them to multiple fields.
- Monthly climatology time interpolation for aerosols.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import ctypes as ct
import numpy as np
import xarray as xr

from ..main import config
from ..main.config import CONFIGDICT
from .stack import tools_to_stack_xarrays


__all__ = [
    "interp_vertical",
    "interp_fld_vertical",
    "interpolate_monthly_clim",
    "interpolate_aero",
    "get_interpolated_ccn",
]

# -----------------------------------------------------------------------------
# ctypes setup (the compiled library expects float64 / int32)
# -----------------------------------------------------------------------------
F_LIB = ct.CDLL(config.VERTINTERP_LIB)

C_REAL = ct.c_double
C_INT = ct.c_int32
C_REAL_PTR = np.ctypeslib.ndpointer(C_REAL)
C_INT_PTR = np.ctypeslib.ndpointer(C_INT)

# Signatures
F_LIB.interp.argtypes = [C_REAL_PTR, C_REAL_PTR, C_INT_PTR, C_REAL_PTR] + [C_INT] * 6
F_LIB.interp_fld.argtypes = [C_REAL_PTR, C_REAL_PTR, C_INT_PTR, C_REAL_PTR] + [C_INT] * 6


# -----------------------------------------------------------------------------
# Low-level wrappers
# -----------------------------------------------------------------------------
def interp_vertical(
    psrc: np.ndarray,
    ptgt: np.ndarray,
    chunk_size_max: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute vertical interpolation indices/weights from source to target pressures.

    Parameters
    ----------
    psrc : np.ndarray
        Shape (ncom, nsrc, nlevsrc). Must be float64 (will be cast) and C-contiguous.
    ptgt : np.ndarray
        Shape (ncom, ntgt, nlevtgt). Must be float64 (will be cast).
    chunk_size_max : int
        Chunk size hint for the compiled routine.

    Returns
    -------
    (tgtlevs, weights) : tuple[np.ndarray, np.ndarray]
        Both with shape (ncom, nsrc, ntgt, nlevtgt). `tgtlevs` is int32, `weights` float64.
    """
    ncom, ntgt, nlevtgt = ptgt.shape
    ncom2, nsrc, nlevsrc = psrc.shape
    if ncom != ncom2:
        raise ValueError(
            "Different common dimension size between source and target! "
            f"(src: ncom={ncom2}, dst: ncom={ncom})"
        )

    outshape = (ncom, nsrc, ntgt, nlevtgt)
    tgtlevs = np.zeros(outshape, dtype=np.int32)
    weights = np.zeros(outshape, dtype=np.float64)

    # Enforce contiguity and dtype for ctypes
    psrc_cont = np.ascontiguousarray(psrc, dtype=np.float64)
    ptgt_cont = np.ascontiguousarray(ptgt, dtype=np.float64)
    tgtlevs = np.ascontiguousarray(tgtlevs, dtype=np.int32)
    weights = np.ascontiguousarray(weights, dtype=np.float64)

    F_LIB.interp(
        psrc_cont,
        ptgt_cont,
        tgtlevs,
        weights,
        C_INT(ncom),
        C_INT(nsrc),
        C_INT(ntgt),
        C_INT(nlevsrc),
        C_INT(nlevtgt),
        C_INT(chunk_size_max),
    )
    return tgtlevs, weights


def interp_fld_vertical(
    fsrc: np.ndarray,
    tgtlevs: np.ndarray,
    weights: np.ndarray,
    chunk_size_max: int = 1000,
) -> np.ndarray:
    """Apply precomputed vertical weights/indices to a 3‑D source field.

    Parameters
    ----------
    fsrc : np.ndarray
        Source field, shape (ncom, nsrc, nlevsrc).
    tgtlevs : np.ndarray
        Target indices, shape (ncom, ntgt, nlevtgt), int32.
    weights : np.ndarray
        Target weights, shape (ncom, ntgt, nlevtgt), float64.
    chunk_size_max : int
        Chunk size hint for the compiled routine.

    Returns
    -------
    np.ndarray
        Interpolated field, shape (ncom, nsrc, ntgt, nlevtgt), float64.
    """
    ncom, nsrc, nlevsrc = fsrc.shape
    ncom2, ntgt, nlevtgt = tgtlevs.shape
    ncom3, ntgt2, nlevtgt2 = weights.shape
    if (ncom != ncom2) or (ncom2 != ncom3) or (ntgt != ntgt2) or (nlevtgt != nlevtgt2):
        raise ValueError(
            "Some dimensions are incompatible! "
            f"fsrc (ncom={ncom}, nsrc={nsrc}, nlevsrc={nlevsrc}); "
            f"tgtlevs (ncom={ncom2}, ntgt={ntgt}, nlevtgt={nlevtgt}); "
            f"weights (ncom={ncom3}, ntgt={ntgt2}, nlevtgt={nlevtgt2})"
        )

    fdstshape = (ncom, nsrc, ntgt, nlevtgt)
    fdst = np.zeros(fdstshape, dtype=np.float64)

    # Enforce contiguity and ctypes dtypes
    fsrc_cont = np.ascontiguousarray(fsrc, dtype=np.float64)
    fdst = np.ascontiguousarray(fdst, dtype=np.float64)
    tgtlevs_cont = np.ascontiguousarray(tgtlevs, dtype=np.int32)
    weights_cont = np.ascontiguousarray(weights, dtype=np.float64)

    F_LIB.interp_fld(
        fsrc_cont,
        fdst,
        tgtlevs_cont,
        weights_cont,
        C_INT(ncom),
        C_INT(nsrc),
        C_INT(ntgt),
        C_INT(nlevsrc),
        C_INT(nlevtgt),
        C_INT(chunk_size_max),
    )
    return fdst


# -----------------------------------------------------------------------------
# Monthly climatology interpolation
# -----------------------------------------------------------------------------
def interpolate_monthly_clim(
    dset: xr.Dataset | xr.DataArray,
    dates: xr.DataArray,
) -> xr.Dataset | xr.DataArray:
    """Linearly interpolate a monthly climatology to arbitrary `dates`.

    The input `dset` **must** have a `month` coordinate/dimension.

    Parameters
    ----------
    dset : xr.Dataset | xr.DataArray
        Monthly climatology with a `month` dimension.
    dates : xr.DataArray
        Target dates (datetime64).

    Returns
    -------
    xr.Dataset | xr.DataArray
        The same kind as input, interpolated to `dates` along a new/used `time` coordinate.
    """
    # Previous and following "anchor" mid-months around the target dates
    prev_month = (dates.values - np.timedelta64(14, "D")).astype("datetime64[M]") +\
          np.timedelta64(14, "D")
    foll_month = (prev_month + np.timedelta64(18, "D")).astype("datetime64[M]") +\
          np.timedelta64(14, "D")

    monthdelta = foll_month - prev_month
    thisdelta = dates - prev_month.astype("datetime64[ns]")
    timeweight = thisdelta / monthdelta  # in [0, 1]

    intmonths_bot = xr.DataArray(
        data=prev_month.astype("datetime64[ns]"), 
        coords={"time": dates}
        ).dt.month
    intmonths_top = xr.DataArray(
        data=foll_month.astype("datetime64[ns]"),
        coords={"time": dates}
        ).dt.month

    lower = dset.sel(month=intmonths_bot).drop_vars("month")
    upper = dset.sel(month=intmonths_top).drop_vars("month")
    return (1 - timeweight) * lower + timeweight * upper


# -----------------------------------------------------------------------------
# High-level aerosol interpolation
# -----------------------------------------------------------------------------
def _find_dim_name(basename: str, current_dims: Sequence[str]) -> str:
    """Find a unique auxiliary dimension name given existing dims."""
    for idx in range(len(current_dims) + 2):
        dim_name = f"{basename}_{idx}"
        if dim_name not in current_dims:
            return dim_name
    # Fallback (should never happen in practice)
    return f"{basename}__aux"


def interpolate_aero(
    aero_fields: xr.Dataset,
    dst_pres: xr.DataArray,
    aero_timeinterp: bool = True,
    intp_dim_name: str = "lev",
) -> xr.Dataset:
    """Interpolate aerosol fields to target pressure levels (and optionally in time)."""

    print("Computing aero vertical interpolation weights...")

    tmp_src: xr.DataArray = aero_fields["pressure"]
    tmp_dst: xr.DataArray = dst_pres

    # Optionally time-interpolate climatology to only the unique target dates
    if aero_timeinterp:
        unique_dates_arr = np.unique(tmp_dst.time.dt.date)
        unique_dates = xr.DataArray(data=unique_dates_arr, dims=["time"])
        tmp_src = interpolate_monthly_clim(tmp_src, unique_dates.astype("datetime64[ns]")).compute()
        aux_datedim = _find_dim_name("tmp_aux_date", list(dst_pres.dims) + list(aero_fields.dims))
    else:
        aux_datedim = None
        unique_dates_arr = [None]

    # Compute weights per unique date slice, then concatenate back along time
    tmp_tgtlevs_list: List[xr.DataArray] = []
    tmp_weights_list: List[xr.DataArray] = []
    tmp_stacktools = None

    for date in unique_dates_arr:
        timeslice = slice(str(date), str(date)) if date is not None else tmp_dst.time.values

        if "time" in tmp_src.dims:
            this_tmp_src = tmp_src.sel(time=timeslice)
            if aero_timeinterp and aux_datedim:
                this_tmp_src = this_tmp_src.rename(time=aux_datedim)
        else:
            this_tmp_src = tmp_src

        this_tmp_dst = tmp_dst.sel(time=timeslice)
        if intp_dim_name not in this_tmp_dst.dims:
            this_tmp_dst = this_tmp_dst.expand_dims(intp_dim_name)

        tmp_stacktools = tools_to_stack_xarrays(
            src_arr=this_tmp_src,
            dst_arr=this_tmp_dst,
            intp_dim_name=intp_dim_name,
        )

        tgtlevs_aero_np, weights_aero_np = interp_vertical(
            psrc=this_tmp_src.transpose(*tmp_stacktools.src_dim_order).values.reshape(
                tmp_stacktools.src_stackshape
            ),
            ptgt=this_tmp_dst.transpose(*tmp_stacktools.dst_dim_order).values.reshape(
                tmp_stacktools.dst_stackshape
            ),
        )

        # Wrap as DataArray on original dims
        tgtlevs_aero = xr.DataArray(
            data=tgtlevs_aero_np.reshape(tmp_stacktools.out_shape),
            dims=tmp_stacktools.out_dim_order,
            coords=tmp_stacktools.out_coords,
        )
        weights_aero = xr.DataArray(
            data=weights_aero_np.reshape(tmp_stacktools.out_shape),
            dims=tmp_stacktools.out_dim_order,
            coords=tmp_stacktools.out_coords,
        )

        if aero_timeinterp and aux_datedim:
            tgtlevs_aero = tgtlevs_aero.squeeze(aux_datedim, drop=True)
            weights_aero = weights_aero.squeeze(aux_datedim, drop=True)

        tmp_tgtlevs_list.append(tgtlevs_aero)
        tmp_weights_list.append(weights_aero)

    tgtlevs_aero = xr.concat(tmp_tgtlevs_list, dim="time")
    weights_aero = xr.concat(tmp_weights_list, dim="time")
    print("Done!")

    # Interpolate each aerosol species (all non-pressure variables)
    specs = [v for v in aero_fields if v != "pressure"]
    print(f"The following species will be interpolated: {specs}")

    interpolaeros: List[xr.DataArray] = []
    for spec in specs:
        print(f"Interpolating {spec}...", flush=True)

        # Optional time interpolation on the species itself
        if aero_timeinterp:
            tmp_src_spec = interpolate_monthly_clim(
                aero_fields[spec],
                unique_dates.astype("datetime64[ns]"),  # type: ignore[name-defined]
            ).compute()
        else:
            tmp_src_spec = aero_fields[spec]

        interpospecs: List[xr.DataArray] = []
        for date in unique_dates_arr:
            if date is not None:
                timeslice = slice(str(date), str(date))
            else:
                timeslice = tgtlevs_aero.time.values

            if "time" in tmp_src_spec.dims:
                this_tmp_src = tmp_src_spec.sel(time=timeslice)
                if aero_timeinterp and aux_datedim:
                    this_tmp_src = this_tmp_src.rename(time=aux_datedim)
            else:
                this_tmp_src = tmp_src_spec

            this_tgtlevs_aero = tgtlevs_aero.sel(time=timeslice)
            this_weights_aero = weights_aero.sel(time=timeslice)

            tmp_stacktools = tools_to_stack_xarrays(
                src_arr=this_tmp_src,
                dst_arr=this_tgtlevs_aero,
                intp_dim_name=intp_dim_name,
            )

            this_aero_field_intp_np = interp_fld_vertical(
                fsrc=this_tmp_src.transpose(
                    *tmp_stacktools.src_dim_order
                    ).values.reshape(
                    tmp_stacktools.src_stackshape
                ),
                tgtlevs=this_tgtlevs_aero.transpose(
                    *tmp_stacktools.dst_dim_order
                    ).values.reshape(
                    tmp_stacktools.dst_stackshape
                ),
                weights=this_weights_aero.transpose(
                    *tmp_stacktools.dst_dim_order
                    ).values.reshape(
                    tmp_stacktools.dst_stackshape
                ),
            )

            this_aero_field_intp = xr.DataArray(
                data=this_aero_field_intp_np.reshape(
                    tmp_stacktools.out_shape).astype(np.float32),
                dims=tmp_stacktools.out_dim_order,
                coords=tmp_stacktools.out_coords,
            )
            if aero_timeinterp and aux_datedim:
                this_aero_field_intp = \
                    this_aero_field_intp.squeeze(aux_datedim, drop=True)

            interpospecs.append(this_aero_field_intp)

        interpolaeros.append(
            xr.concat(interpospecs, dim="time").rename(spec).squeeze(dim=intp_dim_name)
            )

    return xr.merge(interpolaeros)


def get_interpolated_ccn(
    this_ccn_mmr: xr.Dataset,
    this_ifs: xr.Dataset,
    this_ifs_fixedlevel: Optional[xr.Dataset],
    actual_recipe: Dict[str, Any],
) -> xr.Dataset:
    """Interpolate CCN (in MMR) to the needed level(s), depending on configuration.

    Two groups may be extracted:
      * Out‑of‑cloud aerosols at fixed/below‑cloud levels (when configured).
      * In‑cloud aerosols at the native cloud level.

    The merged Dataset preserves a deterministic dimension order aligned with the
    target host (either fixed-level or cloud-level IFS slices).
    """
    if (CONFIGDICT["aeros_out_of_cloud"] is None) and config.SOME_AEROS_OUT_OF_CLOUD:
        aeros_out_of_cloud = list(actual_recipe)
    else:
        aeros_out_of_cloud = CONFIGDICT["aeros_out_of_cloud"]

    if config.SOME_AEROS_OUT_OF_CLOUD:
        print("Picking aerosols at out-of-cloud level")
        if aeros_out_of_cloud is None:
            print("All aerosols are picked at the fixed or below-cloud level!")
            aeros_out_of_cloud = list(actual_recipe)

        out_of_cloud_ccn = [a for a in actual_recipe if a in aeros_out_of_cloud]
        in_cloud_ccn = [a for a in actual_recipe if a not in aeros_out_of_cloud]

        if out_of_cloud_ccn:
            ccn_mcon = interpolate_aero(
                this_ccn_mmr[["pressure"] + out_of_cloud_ccn],  # type: ignore[index]
                this_ifs_fixedlevel["p"],  # type: ignore[index]
                aero_timeinterp=CONFIGDICT["aerofromclimatology"],
            )
            # Align to fixed-level dims (only keep dims present in the interpolated result)
            dimorder = [d for d in this_ifs_fixedlevel["p"].dims if d in ccn_mcon.dims]  # type: ignore[index]
            ccn_mcon = ccn_mcon.transpose(*dimorder)
        else:
            ccn_mcon = xr.Dataset(None)
    else:
        out_of_cloud_ccn = []
        in_cloud_ccn = list(actual_recipe.keys())
        ccn_mcon = xr.Dataset(None)

    # In-cloud aerosols at cloud level
    if in_cloud_ccn:
        print(f"Picking aerosols at cloud level: {in_cloud_ccn}")
        print(str(this_ccn_mmr), flush=True)

        ccn_mcon.update(
            interpolate_aero(
                this_ccn_mmr[["pressure"] + in_cloud_ccn],  # type: ignore[index]
                this_ifs["p"],  # type: ignore[index]
                aero_timeinterp=CONFIGDICT["aerofromclimatology"],
            )
        )
        dimorder = [d for d in this_ifs["p"].dims if d in ccn_mcon.dims] # type: ignore[index]
        ccn_mcon = ccn_mcon.transpose(*dimorder)

    # Preserve the final species order (in-cloud first to match call sites)
    return ccn_mcon[in_cloud_ccn + out_of_cloud_ccn]