"""LUT helpers: map r0 to LUT aerosol specs and compute Nd from stacked inputs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import multiprocessing.dummy as mp  # thread-based Pool

import ctypes as cty
import numpy as np
import xarray as xr

from ..main import config
from .aerosol import IFSAeroSpecs
from .stack import StackAeroTuple, StackLutTuple



def _select_include_w_list(wspeed_type: int) -> List[str]:
    """Return the list of w variables to include based on wspeed_type."""
    if wspeed_type == 0:
        return []
    if wspeed_type < 3:
        return ["w_mean"]
    return ["w_mean", "w_prime"]

def get_lutaero_from_r0(
    r0: Dict[str, float],
    lut_aero: List[Any],
    bind_seasalt_ratio: Optional[float],
) -> Dict[str, IFSAeroSpecs]:
    """
    Build a dict of IFSAeroSpecs from r0 values, honoring an optional sea-salt binding ratio.
    """
    full_r0: Dict[str, float] = {}
    for aero in lut_aero:
        if bind_seasalt_ratio is not None and (aero.name == "seasalt2"):
            this_r0 = r0["seasalt1"] * bind_seasalt_ratio
        elif aero.name in r0:
            this_r0 = r0[aero.name]
        else:
            this_r0 = 1.0e20  # No number contribution
        full_r0[aero.name] = this_r0

    return {
        aero.name: IFSAeroSpecs(
            aero.name,
            np.array(full_r0[aero.name]),
            np.array(aero.shape),
            aero.density,
            0.0,
            np.inf,
            np.array(1),
            aero.m_act,
            1,
        )
        for aero in lut_aero
    }


def compute_nd(  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    lut_species_mcon_stack: StackAeroTuple,
    pyrcel_lut_stack: StackLutTuple,
    nccn_over_mcon: Dict[str, float],
    lutkin: bool,
    prior_mass_fr: Optional[Dict[str, float]] = None,
    chunksize: int = 10000
) -> xr.Dataset:
    """
    Evaluate the pyrcel LUT for the given stacked mass concentrations and return Nd fields.

    Notes
    -----
    - Ensures NumPy dtypes match `ctypes` signatures (float32/int32) for the C library.
    - Returns per-species mass scaler fields if `prior_mass_fr` is provided.
    """
    prior_mass_fr = prior_mass_fr or {}

    # ctypes types
    c_real = cty.c_float
    c_real_ptr = np.ctypeslib.ndpointer(c_real)
    c_int = cty.c_int32
    c_int_ptr = np.ctypeslib.ndpointer(c_int)
    c_bool = cty.c_bool

    f_lib = cty.CDLL(config.GETLUTVAL_LIB)

    # Check flattening order consistency
    c_order = lut_species_mcon_stack.c_order
    if c_order != pyrcel_lut_stack.c_order:
        raise ValueError(
            "Pyrcel LUT and fields must follow the same flattening order! "
            f"Got instead c_ordered lut: {pyrcel_lut_stack.c_order}, fields: {c_order}"
        )

    # Species and vertical-velocity metadata
    nspec = len(lut_species_mcon_stack.varlist)
    nwspeed = len(lut_species_mcon_stack.w_varlist)

    # Scale LUT bins to get mass bins
    lut_spec_bins_list: List[np.ndarray] = [
        pyrcel_lut_stack.lut_bins[f"{config.PYRCNAMEMAP[v]}_nccn"].values
        / float(nccn_over_mcon[v])
        for v in lut_species_mcon_stack.varlist
    ]

    # W-speed bins
    lut_wspeed_bins_list: List[np.ndarray] = [
        pyrcel_lut_stack.lut_bins[v].values for v in lut_species_mcon_stack.w_varlist
    ]

    # Determine maximum bin count across all axes
    max_lut_bins = 0
    for wsp in lut_wspeed_bins_list:
        max_lut_bins = max(max_lut_bins, len(wsp))
    for spc in lut_spec_bins_list:
        max_lut_bins = max(max_lut_bins, len(spc))

    # Pack LUT bin matrices
    nspecbins_list: List[int] = []
    lut_spec_bins = np.zeros((nspec, max_lut_bins), dtype=np.float32)
    for s, spc in enumerate(lut_spec_bins_list):
        lut_spec_bins[s, : len(spc)] = spc.astype(np.float32, copy=False)
        nspecbins_list.append(len(spc))
    nspecbins = np.asarray(nspecbins_list, dtype=np.int32)

    aero_species = np.ascontiguousarray(lut_species_mcon_stack.data, dtype=np.float32)

    if nwspeed > 0:
        nwspeedbins_list: List[int] = []
        lut_wspeed_bins = np.zeros((nwspeed, max_lut_bins), dtype=np.float32)
        for w, wsp in enumerate(lut_wspeed_bins_list):
            lut_wspeed_bins[w, : len(wsp)] = wsp.astype(np.float32, copy=False)
            nwspeedbins_list.append(len(wsp))
        nwspeedbins = np.asarray(nwspeedbins_list, dtype=np.int32)
        wspeeds = np.ascontiguousarray(lut_species_mcon_stack.w_data, dtype=np.float32)
    else:
        lut_wspeed_bins = None
        nwspeedbins = None
        wspeeds = None

    lut_maps = pyrcel_lut_stack.lut_maps.astype(np.float32, copy=False)
    map_size, nmaps = lut_maps.shape
    nvals, nspec2 = aero_species.shape

    if nspec2 != nspec:
        raise ValueError(f"Got species with nspec = {nspec2}, but expected nspec = {nspec}!")
    if wspeeds is not None:
        nvals2, nwspeed2 = wspeeds.shape
        if (nvals2 != nvals) or (nwspeed2 != nwspeed):
            raise ValueError(
                f"Got wspeed with shape = ({nvals2}, {nwspeed2}), "
                f"but expected ({nvals}, {nwspeed})!"
            )

    # Output buffer that the C routine will fill
    val_out_data = np.ascontiguousarray(np.zeros((nvals, nmaps), dtype=np.float32))

    # Define signatures & call C
    if nwspeed > 0:
        f_lib.get_flexi_lutvals.argtypes = (
            [c_real_ptr] + [c_int] * 6 + [c_real_ptr] * 2 + [c_int_ptr, c_real_ptr]
            + [c_real_ptr] * 2 + [c_int_ptr] + [c_int, c_bool]
        )
        f_lib.get_flexi_lutvals(
            lut_maps, c_int(nmaps), c_int(map_size),
            c_int(nwspeed), c_int(nspec), c_int(nvals), c_int(max_lut_bins),
            aero_species, lut_spec_bins,
            nspecbins, val_out_data,
            wspeeds, lut_wspeed_bins,  # type: ignore[arg-type]
            nwspeedbins, c_int(chunksize), c_bool(c_order)  # type: ignore[arg-type]
        )
    else:
        f_lib.get_flexi_lutvals_nowspeed.argtypes = (
            [c_real_ptr] + [c_int] * 6 + [c_real_ptr] * 2 +\
                [c_int_ptr, c_real_ptr] + [c_int, c_bool]
        )
        f_lib.get_flexi_lutvals_nowspeed(
            lut_maps, c_int(nmaps), c_int(map_size),
            c_int(nwspeed), c_int(nspec), c_int(nvals), c_int(max_lut_bins),
            aero_species, lut_spec_bins,
            nspecbins, val_out_data,
            c_int(chunksize), c_bool(c_order)
        )

    # Map column indices to names for convenient access
    num_act_var = "num_act_kn" if lutkin else "num_act"
    mass_act_var = "mass_act_kn" if lutkin else "mass_act"
    val_out_data_map: Dict[str, int] = {mapname: m 
                                        for m, mapname in enumerate(pyrcel_lut_stack.maplist)}

    # Compute total Nd as the sum of species contributions
    def _species_nd(v_and_name: tuple[int, str]) -> np.ndarray:
        v, var = v_and_name
        pyrcvar = config.PYRCNAMEMAP[var]
        tmp_nd = (
            val_out_data[:, val_out_data_map[f"{pyrcvar}_{num_act_var}"]]
            * lut_species_mcon_stack.data[:, v]
            * nccn_over_mcon[var]
        )
        if prior_mass_fr:
            mass_scaler_np = (
                float(prior_mass_fr[var]) / \
                    val_out_data[:, val_out_data_map[f"{pyrcvar}_{mass_act_var}"]]
            )
            tmp_nd = tmp_nd * mass_scaler_np
        return tmp_nd

    arglist = [(v, var) for v, var in enumerate(lut_species_mcon_stack.varlist)
               if var not in ["w_mean", "w_prime"]]

    # Use thread pool over species; this is CPU-light compared to the C call but keeps parity
    with mp.Pool(len(arglist)) as pool:
        all_nds = pool.map(_species_nd, arglist)

    tot_nd = np.zeros_like(val_out_data[:, 0], dtype=np.float32)
    for contrib in all_nds:
        tot_nd += contrib.astype(np.float32, copy=False)

    # Choose reshape order and build the output Dataset
    flat_order = "C" if c_order else "F"
    data_vars = {
        mapname: (
            lut_species_mcon_stack.dimorder,
            val_out_data[:, m].reshape(lut_species_mcon_stack.fldshape, order=flat_order),
        )
        for mapname, m in val_out_data_map.items()
    }
    data_vars["tot_nd"] = (
        lut_species_mcon_stack.dimorder,
        tot_nd.reshape(lut_species_mcon_stack.fldshape, order=flat_order),
    )

    # Also export mass scaler fields if provided
    if prior_mass_fr:
        for var in lut_species_mcon_stack.varlist:
            if var in ["w_mean", "w_prime"]:
                continue
            pyrcvar = config.PYRCNAMEMAP[var]
            mass_scaler_np = (
                float(prior_mass_fr[var]) / \
                    val_out_data[:, val_out_data_map[f"{pyrcvar}_{mass_act_var}"]]
            ).astype(np.float32, copy=False)
            data_vars[f"{var}_mass_scaler"] = (
                lut_species_mcon_stack.dimorder,
                mass_scaler_np.reshape(lut_species_mcon_stack.fldshape, order=flat_order),
            )

    val_out = xr.Dataset(
        data_vars=data_vars,
        coords={d: lut_species_mcon_stack.coords[d] for d in lut_species_mcon_stack.coords},
    )
    return val_out
