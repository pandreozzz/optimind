import json
import numpy as np
import xarray as xr
from ..tools.aerosol import IFSAeroSpecs
from ..main.config import CONFIGDICT

def setup_pyrcel_lut(pyrcel_lut_path = None):
    """
    Load and initialize pyrcel lookup table based on wspeed_type configuration.
    
    Routes to the appropriate setup function (setup_pyrcel_lut_0/1_2/3_4) based on
    CONFIGDICT['wspeed_type'] and sets global LUT variables.
    
    Args:
        pyrcel_lut_path: str - Path to pyrcel LUT netcdf/zarr file
    
    Raises:
        ValueError: If wspeed_type is not in [0,1,2,3,4]
    
    Side Effects:
        - Sets global PYRCELLUT, THISLUTAERO, PYRCNAMEMAP, AERONAMEMAP, THISRECIPE
    """

    if pyrcel_lut_path is None:
        pyrcel_lut_path = CONFIGDICT["pyrce"]
    if CONFIGDICT["wspeed_type"] == 0:
        setup_pyrcel_lut_0(pyrcel_lut_path)
    elif CONFIGDICT["wspeed_type"] in [1, 2]:
        setup_pyrcel_lut_1_2(pyrcel_lut_path)
    elif CONFIGDICT["wspeed_type"] in [3, 4]:
        setup_pyrcel_lut_3_4(pyrcel_lut_path)
    else:
        raise ValueError(f"Could not setup Pyrcel lut. Unrecognized wspeed_type {CONFIGDICT['wspeed_type']}")

def finalize_pyrcel_lut_setup(pyrcellut : xr.Dataset, ccn_densities, ccn_mact, recipe_file_path, include_w : list = []):
    """
    Common finalization for all pyrcel LUT setup variants (wspeed_type 0,1,2,3,4).
    
    Args:
        pyrcellut: xarray Dataset - The loaded pyrcel lookup table
        ccn_densities: list - CCN species densities (kg/m^3)
        ccn_mact: list - CCN mass activation thresholds
        recipe_file_path: str - Path to CCN recipe JSON file
    """
    
    pyrcellut = pyrcellut.sortby("spec_num")
    aeronames = pyrcellut["name"].values
    aeroidxs = pyrcellut.spec_num.values
    aeronamemap = {f"aero{spn}": str(pyrcellut.name.sel(spec_num=spn).item()) for spn in aeroidxs}
    pyrcnamemap = {v: k for k, v in aeronamemap.items()}

    dim_order = ["spec_num"]+include_w+[f"aero{i}" for i in aeroidxs]
    
    if len(aeronames) != len(ccn_densities):
        raise ValueError("ccn_densities should be the same number of pyrcel LUT species!"+
                        f"got instead ccn_densities ({len(ccn_densities)}), "+
                        f"aeronames ({len(aeronames)})")
    if len(aeronames) != len(ccn_mact):
        raise ValueError("ccn_mact should be the same number of pyrcel LUT species!"+
                        f"got instead ccn_mact ({len(ccn_mact)}), "+
                        f"aeronames ({len(aeronames)})")

    thislutaero = [
        IFSAeroSpecs(name,
                  rmed, shape,
                  dens, 0, np.inf, np.array(1.), mact, 1)
        for name, rmed, shape, dens, mact in zip(
            aeronames, pyrcellut["r_median"].values, pyrcellut["shape"].values,
            ccn_densities, ccn_mact, strict=True)]

    pyrcellut = pyrcellut.assign_coords({f"aero{aerox}": pyrcellut[f"aero{aerox}_nccn"] for aerox in aeroidxs})


    with open(recipe_file_path, 'r') as recipe_file:
        ccn_recipes = json.load(recipe_file)
    thisrecipe = {}
    print("Populating recipes...")
    for ccn_idx in aeroidxs:
        ccn_aero = f"aero{ccn_idx}"
        if ccn_aero not in ccn_recipes:
            raise ValueError(
                f"ccn name {ccn_aero} from LUT" +\
                f"was not found in recipe file {recipe_file}"
            )
        print(f"Adding {aeronamemap[ccn_aero]}...")
        thisrecipe[aeronamemap[ccn_aero]] = ccn_recipes[ccn_aero]

    # Set global state
    from ..main import config as cfg
    cfg.PYRCELLUT = pyrcellut
    cfg.THISLUTAERO = thislutaero
    cfg.PYRCNAMEMAP = pyrcnamemap
    cfg.AERONAMEMAP = aeronamemap
    cfg.THISRECIPE = thisrecipe
    


def setup_pyrcel_lut_0(pyrcel_lut_path):
    """
    Setup pyrcel LUT for wspeed_type 0 (monodisperse vertical velocity).
    
    Loads LUT with fixed vertical velocity (no w_prime or w_mean), interpolates
    to configured speed, and prepares for CCN calculations.
    
    Args:
        pyrcel_lut_path: str - Path to pyrcel LUT netcdf/zarr file
    
    Raises:
        ValueError: If wspeed is outside LUT bounds
    
    Side Effects:
        Sets global PYRCELLUT, THISLUTAERO, PYRCNAMEMAP, AERONAMEMAP, THISRECIPE
    """
    
    print(f"Opening LUT from {pyrcel_lut_path}")
    pyrcellut = xr.open_dataset(pyrcel_lut_path)

    if (CONFIGDICT["wspeed"] < pyrcellut.w.values.min()) or (CONFIGDICT["wspeed"] > pyrcellut.w.values.max()):
        raise ValueError(f"vertical speed {CONFIGDICT['wspeed']} m/s out of bounds for Pyrcel LUT!")
    
    print(f"Linearly interpolating Pyrcel LUT for vertical speed {CONFIGDICT['wspeed']} m/s")
    pyrcellut = pyrcellut.interp(w=CONFIGDICT["wspeed"], method="linear").compute()


    include_w = ["w"] if pyrcellut.w.shape != () else []

    finalize_pyrcel_lut_setup(
        pyrcellut,
        CONFIGDICT["ccn_densities"], CONFIGDICT["ccn_mact_def"],
        CONFIGDICT["ccn_recipe_file"], include_w)

def setup_pyrcel_lut_1_2(pyrcel_lut_path):
    """
    Setup pyrcel LUT for wspeed_type 1 or 2 (w_mean and w_prime).
    
    Loads LUT with both mean and turbulent vertical velocity, interpolates
    w_prime to configured value, and prepares for CCN calculations.
    
    Args:
        pyrcel_lut_path: str - Path to pyrcel LUT netcdf/zarr file
    
    Raises:
        ValueError: If w_prime is outside LUT bounds
    
    Side Effects:
        Sets global PYRCELLUT, THISLUTAERO, PYRCNAMEMAP, AERONAMEMAP, THISRECIPE
    """
    
    print(f"Opening LUT from {pyrcel_lut_path}")
    pyrcellut = xr.open_dataset(pyrcel_lut_path)

    if (CONFIGDICT["w_prime"] < pyrcellut.w_prime.values.min()) or (CONFIGDICT["w_prime"] > pyrcellut.w_prime.values.max()):
        raise ValueError(f"w_prime {CONFIGDICT['w_prime']} m/s out of bounds for Pyrcel LUT!")
    
    print(f"Linearly interpolating Pyrcel LUT for fixed w_prime {CONFIGDICT['w_prime']} m/s")
    pyrcellut = pyrcellut.interp(w_prime=CONFIGDICT["w_prime"], method="linear").compute()

    finalize_pyrcel_lut_setup(
        pyrcellut, 
        CONFIGDICT["ccn_densities"], CONFIGDICT["ccn_mact_def"],
        CONFIGDICT["ccn_recipe_file"], ["w_mean"])


def setup_pyrcel_lut_3_4(pyrcel_lut_path):
    """
    Setup pyrcel LUT for wspeed_type 3 or 4 (w_mean and w_prime with Deardorff scaling).
    
    Loads LUT with both mean and turbulent vertical velocity, applies Deardorff scaling
    based on cloud base height, and prepares for CCN calculations.
    
    Args:
        pyrcel_lut_path: str - Path to pyrcel LUT netcdf/zarr file
    
    Side Effects:
        Sets global PYRCELLUT, THISLUTAERO, PYRCNAMEMAP, AERONAMEMAP, THISRECIPE
    """
    
    print(f"Opening LUT from {pyrcel_lut_path}")
    pyrcellut = xr.open_dataset(pyrcel_lut_path)

    print(f"Using deardorff_scale: {CONFIGDICT['deardorff_scale']:.02f},\n" +
          f"LUT w_mean range: {pyrcellut.w_mean.min().values:.02f}m/s-{pyrcellut.w_mean.max().values:.02f}m/s\n" +
          f"LUT w_prime range: {pyrcellut.w_prime.min().values:.02f}m/s-{pyrcellut.w_prime.max().values:.02f}m/s")

    finalize_pyrcel_lut_setup(
        pyrcellut,
        CONFIGDICT["ccn_densities"], CONFIGDICT["ccn_mact_def"],
        CONFIGDICT["ccn_recipe_file"], ["w_mean", "w_prime"])


def get_actual_lut_recipes(represented_aeros):
    """
    Adjust LUT recipe to include only represented aerosol species
    
    Args:
        represented_aeros: - Iterable containing names of represented aerosols

    Returns:
        actual_pyrcellut, actual_recipe, needed_aeros
    """
    from ..main.config import THISRECIPE, PYRCNAMEMAP, PYRCELLUT
    
    var_to_keep = ["pressure"]
    actual_recipe = {}
    print("Computing CCN species:", flush=True)
    lut_dims_to_exclude = []
    for ccn_name,rec in THISRECIPE.items():
        this_ccn_present = False
        for aero in rec:
            if aero in represented_aeros:
                this_ccn_present = True
                var_to_keep.append(aero)
        if this_ccn_present:
            actual_recipe[ccn_name] = {}
            for aero in rec:
                if aero in represented_aeros:
                    actual_recipe[ccn_name][aero] = rec[aero]
            print(f"CCN species {ccn_name}:")
            print("\n".join([f"\t{k} {v:.2f}"
                             for k,v in actual_recipe[ccn_name].items()]))
        else:
            print(f"Excluding PYRCEL LUT CCN species {ccn_name}")
            lut_dims_to_exclude.append(PYRCNAMEMAP[ccn_name])

    if len(lut_dims_to_exclude) > 0:
        actual_pyrcellut = PYRCELLUT.isel({dim: 0 for dim in lut_dims_to_exclude},
                                          drop=True).load()
    else:
        actual_pyrcellut = PYRCELLUT.load()
    print("done!", flush=True)
    return actual_pyrcellut, actual_recipe, list(set(var_to_keep))
