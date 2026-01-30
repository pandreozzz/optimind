import numpy as np
import xarray as xr

from ..main import config
from  .aerosol import IFSAeroSpecs
from .stack import StackAeroTuple, StackLutTuple

def get_lutaero_from_r0(r0 : dict, lut_aero : list, bind_seasalt_ratio : float) -> dict:
    full_r0 = {}

    for aero in lut_aero:
        if bind_seasalt_ratio is not None and (aero.name == "seasalt2"):
            this_r0 = r0["seasalt1"]*bind_seasalt_ratio
        elif aero.name in r0:
            this_r0 = r0[aero.name]
        else:
            this_r0 = 1.e20 # No number contribution
        full_r0[aero.name] = this_r0
        del this_r0

    return {aero.name:
        IFSAeroSpecs(aero.name,
                  np.array(full_r0[aero.name]),  np.array(aero.shape),
                  aero.density, 0., np.inf, np.array(1), aero.m_act, 1)
        for aero in lut_aero
           }

def compute_nd(lut_species_mcon_stack : StackAeroTuple,
                     pyrcel_lut_stack : StackLutTuple,
                     nccn_over_mcon : dict, lutkin : bool,
                     prior_mass_fr : dict = {}, chunksize=10000,
                     chunk_out=False, verbose=False) -> xr.Dataset:
    """
    lutkin specifies whether the
    """
    import multiprocessing.dummy as mp

    from time import time
    stsec = time()

    import ctypes as cty

    f_lib = cty.CDLL(config.GETLUTVAL_LIB)
    c_real = cty.c_float
    c_real_ptr = np.ctypeslib.ndpointer(c_real)
    c_int  = cty.c_int32
    c_int_ptr = np.ctypeslib.ndpointer(c_int)
    c_bool = cty.c_bool

    f_lib.get_flexi_lutvals.argtypes = [c_real_ptr,]*4+[c_int,]*5

    # Check flattening order
    c_order = lut_species_mcon_stack.c_order
    if c_order ^ pyrcel_lut_stack.c_order:
        raise ValueError(
            "Pyrcel LUT and fields must follow the same flattening order!" +\
            f"Got instead c_ordered lut: {pyrcel_lut_stack.c_order}, fields: {c_order}")


    # [sulfates, seasalt1, seasalt2...]
    nspec = len(lut_species_mcon_stack.varlist)
    nwspeed = len(lut_species_mcon_stack.w_varlist)

    # Scale lut bins to get mass_bins
    lut_spec_bins_list = [
        pyrcel_lut_stack.lut_bins[f"{config.PYRCNAMEMAP[v]}_nccn"].values/float(nccn_over_mcon[v])
        for v in lut_species_mcon_stack.varlist
    ]
    # Vertical velocity dimensions
    lut_wspeed_bins_list = [
        pyrcel_lut_stack.lut_bins[v].values for v in lut_species_mcon_stack.w_varlist
    ]

    # Find max_lut_bins
    max_lut_bins = 0
    for wsp in lut_wspeed_bins_list:
        max_lut_bins = max(max_lut_bins, len(wsp))
    for spc in lut_spec_bins_list:
        max_lut_bins = max(max_lut_bins, len(spc))

    # Populate lut_bins matrices
    nspecbins = []
    lut_spec_bins = np.zeros((nspec, max_lut_bins), dtype=np.float32)
    for s,spc in enumerate(lut_spec_bins_list):
        lut_spec_bins[s, :len(spc)] = spc
        nspecbins.append(len(spc))
    nspecbins = np.array(nspecbins, dtype=np.float32)
    aero_species = np.ascontiguousarray(lut_species_mcon_stack.data)
    del lut_spec_bins_list

    if nwspeed > 0:
        nwspeedbins = []
        lut_wspeed_bins = np.zeros((nwspeed, max_lut_bins), dtype=np.float32)
        for w,wsp in enumerate(lut_wspeed_bins_list):
            lut_wspeed_bins[w, :len(wsp)] = wsp
            nwspeedbins.append(len(wsp))
        nwspeedbins = np.array(nwspeedbins, dtype=np.float32)
        wspeeds = np.ascontiguousarray(lut_species_mcon_stack.w_data)
    else:
        lut_wspeed_bins = None
        nwspeedbins = None
        wspeeds = None
    del lut_wspeed_bins_list

    lut_maps = pyrcel_lut_stack.lut_maps
    map_size, nmaps = lut_maps.shape

    nvals, nspec2 = aero_species.shape
    # Check ntotdims consistent with spec and wspeed dims
    if (nspec2 != nspec):
        raise ValueError(f"Got species with nspec = {nspec2}, but expected \
        nspec={nspec}!")

    if wspeeds is not None:
        nvals2, nwspeed2 = wspeeds.shape
        if (nvals2 != nvals) or (nwspeed2 != nwspeed):
            raise ValueError(f"Got wspeed with shape = ({nvals2}, {nwspeed2}), but expected \
            ({nvals}, {nwspeed})!")

    # Which maps
    num_act_var = "num_act_kn" if lutkin else "num_act"
    mass_act_var = "mass_act_kn" if lutkin else "mass_act"

    val_out_data = np.ascontiguousarray(
        np.zeros((nvals, nmaps)),
        dtype=np.float32)

    if nwspeed > 0:
        f_lib.get_flexi_lutvals.argtypes = [c_real_ptr]+[c_int]*6+\
        [c_real_ptr]*2+[c_int_ptr, c_real_ptr] +\
        [c_real_ptr]*2+[c_int_ptr]+[c_int, c_bool]
        
        f_lib.get_flexi_lutvals(
            lut_maps.astype(c_real),
            c_int(nmaps), c_int(map_size),
            c_int(nwspeed), c_int(nspec), c_int(nvals), c_int(max_lut_bins),
            aero_species.astype(c_real), lut_spec_bins.astype(c_real),
            nspecbins.astype(c_int), val_out_data,
            wspeeds.astype(c_real), lut_wspeed_bins.astype(c_real), # type: ignore
            nwspeedbins.astype(c_int), c_int(chunksize), c_bool(c_order) # type: ignore
        )
    else:
        f_lib.get_flexi_lutvals_nowspeed.argtypes = [c_real_ptr]+[c_int]*6+\
        [c_real_ptr]*2+[c_int_ptr, c_real_ptr] +\
        [c_int, c_bool]

        f_lib.get_flexi_lutvals_nowspeed(
            lut_maps.astype(c_real),
            c_int(nmaps), c_int(map_size),
            c_int(nwspeed), c_int(nspec), c_int(nvals), c_int(max_lut_bins),
            aero_species.astype(c_real), lut_spec_bins.astype(c_real),
            nspecbins.astype(c_int), val_out_data, c_int(chunksize),
            c_bool(c_order))

    from time import time
    stsec = time()
    # Construct a map to val_out_data
    val_out_data_map = {
        mapname: m
        for m,mapname in enumerate(pyrcel_lut_stack.maplist)
    }

    # this takes most of the computation time...
    def get_tot_nd(v_var):
        v, var = v_var
        pyrcvar = config.PYRCNAMEMAP[var]

        #val_out[f"{var}_nd"] don't need to store array
        tmp_var_nd = val_out_data[:,val_out_data_map[f"{pyrcvar}_{num_act_var}"]]*\
                     lut_species_mcon_stack.data[:,v] *\
                     nccn_over_mcon[var]

        if prior_mass_fr != {}:
            mass_scaler = prior_mass_fr[var]/val_out_data[:,val_out_data_map[f"{pyrcvar}_{mass_act_var}"]]
            val_out[f"{var}_mass_scaler"] = mass_scaler.assign_attrs(units="1")
            tmp_var_nd[...] = tmp_var_nd[...] * mass_scaler.values

        return tmp_var_nd

    arglist = [
        (v,var)
        for v,var in enumerate(lut_species_mcon_stack.varlist)
        if var not in ["w_mean", "w_prime"]
    ]
    
    with mp.Pool(len(arglist)) as p:
        all_nds = p.map(get_tot_nd, arglist)
    tot_nd = np.zeros_like(val_out_data[:,0])

    for this_nd in all_nds:
        tot_nd[...] = tot_nd[...] + this_nd[...]

    # Only C and F orders supported
    flat_order = 'C' if c_order else 'F'
    val_out_data_vars = {
        mapname : (
            lut_species_mcon_stack.dimorder,
            val_out_data[:, m].reshape(lut_species_mcon_stack.fldshape, order=flat_order))
        for mapname, m in val_out_data_map.items()
    }
    val_out_data_vars["tot_nd"] = (lut_species_mcon_stack.dimorder,
                                   tot_nd.reshape(lut_species_mcon_stack.fldshape, order=flat_order))

    val_out = xr.Dataset(
        data_vars = val_out_data_vars,
        coords = {d: lut_species_mcon_stack.coords[d]
                  for d in lut_species_mcon_stack.coords}
    )


    return val_out