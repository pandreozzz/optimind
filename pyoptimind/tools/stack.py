import numpy as np

from collections import namedtuple
StackTools = namedtuple('StackTools', ["src_dim_order", "dst_dim_order", "out_dim_order",
                                       "src_stackshape", "dst_stackshape",
                                       "out_coords", "out_shape"
                                      ])
def tools_to_stack_xarrays(src_arr, dst_arr, intp_dim_name):
    """
    Returns all tools to reorder and reshape arrays using numpy
    """

    nonintp_dims_src = [d  for d in src_arr.dims if d != intp_dim_name]
    nonintp_dims_dst = [d  for d in dst_arr.dims if d != intp_dim_name]

    common_dim_names = list(set(nonintp_dims_src) & set(nonintp_dims_dst))
    unique_dim_names = list(set(nonintp_dims_src) ^ set(nonintp_dims_dst))
    onlysrc_dim_names = [d for d in nonintp_dims_src if d in unique_dim_names]
    onlydst_dim_names = [d for d in nonintp_dims_dst if d in unique_dim_names]
    
    src_dim_order = common_dim_names+onlysrc_dim_names
    dst_dim_order = common_dim_names+onlydst_dim_names
    if intp_dim_name:
        src_dim_order = src_dim_order+[intp_dim_name]
        dst_dim_order = dst_dim_order+[intp_dim_name]
    
    com_ndims = [len(src_arr[com_dim]) for com_dim in common_dim_names]
    src_ndims = [len(src_arr[src_dim]) for src_dim in onlysrc_dim_names]
    src_intp_ndims = [len(src_arr[intp_dim_name])] if intp_dim_name in src_arr.dims else []
    
    dst_ndims = [len(dst_arr[dst_dim]) for dst_dim in onlydst_dim_names]
    dst_intp_ndims = [len(dst_arr[intp_dim_name])] if intp_dim_name in dst_arr.dims else []
    
    src_stackshape = tuple([int(np.array(ndims).prod()) if len(ndims) > 0 else 1 for ndims in [com_ndims, src_ndims, src_intp_ndims] ])
    dst_stackshape = tuple([int(np.array(ndims).prod()) if len(ndims) > 0 else 1 for ndims in [com_ndims, dst_ndims, dst_intp_ndims]])

    out_stackshape = tuple([int(np.array(ndims).prod()) if len(ndims) > 0 else 1  for ndims in [com_ndims, src_ndims, dst_ndims, dst_intp_ndims]])
    out_shape = tuple(com_ndims+src_ndims+dst_ndims+dst_intp_ndims)
    if len(out_shape) == 0:
        out_shape = None
        out_dim_order = None
    else:
        out_dim_order =  common_dim_names+onlysrc_dim_names+onlydst_dim_names
        if intp_dim_name:
            out_dim_order = out_dim_order+[intp_dim_name]
    out_coords = {
        **{com_dim: src_arr.coords[com_dim] for com_dim in common_dim_names},
        **{src_dim: src_arr.coords[src_dim] for src_dim in onlysrc_dim_names},
        **{dst_dim: dst_arr.coords[dst_dim] for dst_dim in onlydst_dim_names},
    }
    if intp_dim_name:
        out_coords[intp_dim_name] = dst_arr.coords[intp_dim_name]
    
    arglist = [src_dim_order, dst_dim_order, out_dim_order]+\
    [src_stackshape, dst_stackshape, out_coords, out_shape]

    return StackTools(*arglist)

StackAeroTuple = namedtuple('stackedaero', ["data", "w_data", "varlist", "w_varlist",
                                            "dimorder", "fldshape", "coords", "c_order"])
def get_stacked_aero(fld, this_lutaero, include_w : list = [],
                     include_rainratio=False, flat_order = 'C') -> StackAeroTuple:
    """
    include_w : list = w variables to include
    """

    # ensure dim order
    varlist = [v.name for v in this_lutaero
               if v.name in list(fld.data_vars)]
    
    if varlist == []:
        print(varlist)
        print(this_lutaero)
        print(fld.data_vars)
        import sys
        sys.exit(1)

    # Include vert_speed dims if required
    varlist = varlist

    if include_rainratio:
        varlist = varlist+["rainratio"]

    dimorder = tuple(fld[varlist[0]].dims)
    fldshape = tuple([len(fld[dim]) for dim in dimorder])

    print(f"Enforcing dimension order on aero fields: {dimorder} {fldshape}")
    fld_ordered = np.ascontiguousarray(np.concatenate(
        [fld[var].transpose(*dimorder).values.flatten(order=flat_order)[:,None]
         for var in varlist], axis=-1))
    w_fld_ordered = np.ascontiguousarray(np.concatenate(
        [fld[var].transpose(*dimorder).values.flatten(order=flat_order)[:,None]
         for var in include_w], axis=-1)) if len(include_w) > 0 else None
    
    coords = {d: fld.coords[d] for d in dimorder if d in fld.coords}

    return StackAeroTuple(fld_ordered, w_fld_ordered, varlist, include_w, dimorder,
                          fldshape, coords, flat_order == 'C')

StackLutTuple = namedtuple('stackedlut', ["lut_maps", "maplist", "maptypes", "lut_bins", "c_order"])
def get_stacked_lut(pyrcel_lut, wspeed_type : int, lutkin : bool,
                    include_mass_act_lut : bool = False, flat_order = 'C') -> StackLutTuple:

    num_act_var = "num_act_kn" if lutkin else "num_act"
    mass_act_var = "mass_act_kn" if lutkin else "mass_act"
    maptypes = [num_act_var]
    if include_mass_act_lut:
        maptypes.append(mass_act_var)

    maplist = [f"aero{spn}_{maptype}"
               for maptype in maptypes
               for spn in pyrcel_lut.spec_num.values]


    lut_maps = np.ascontiguousarray(np.concatenate(
        [pyrcel_lut[mapname].values.flatten(order=flat_order)[:,None]
         for mapname in maplist],
        axis=-1))

    lut_bins_vars = [f"aero{spn}_nccn" for spn in pyrcel_lut.spec_num.values]

    if wspeed_type in [1,2]:
        lut_bins_vars = ["w_mean"] + lut_bins_vars
    elif wspeed_type in [3,4]:
        lut_bins_vars = ["w_mean", "w_prime"] + lut_bins_vars

    return StackLutTuple(lut_maps, maplist, maptypes, pyrcel_lut[lut_bins_vars], flat_order == 'C')