"""Setup and selection utilities for the Pyrcel LUT and CCN recipes."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import json
import numpy as np
import xarray as xr

from ..main.config import CONFIGDICT
# Set global state
from ..main import config  # local import to avoid circulars at import time
from ..tools.aerosol import IFSAeroSpecs


def setup_pyrcel_lut(pyrcel_lut_path: str = "") -> None:
    """
    Load and initialize the Pyrcel lookup table based on `wspeed_type`.

    Dispatches to one of: `setup_pyrcel_lut_0`, `setup_pyrcel_lut_1_2`, or
    `setup_pyrcel_lut_3_4`, then stores global state into `..main.config`.

    Parameters
    ----------
    pyrcel_lut_path : str
        Path to the Pyrcel LUT NetCDF/Zarr. If empty, uses
        `CONFIGDICT["pyrcellutpath"]`.

    Raises
    ------
    ValueError
        If `wspeed_type` is not in {0, 1, 2, 3, 4}.
    """
    if pyrcel_lut_path == "":
        # Original code had CONFIGDICT["pyrce"]; assuming intended key:
        pyrcel_lut_path = CONFIGDICT["pyrcellutpath"]

    wtype = CONFIGDICT["wspeed_type"]
    if wtype == 0:
        setup_pyrcel_lut_0(pyrcel_lut_path)
    elif wtype in [1, 2]:
        setup_pyrcel_lut_1_2(pyrcel_lut_path)
    elif wtype in [3, 4]:
        setup_pyrcel_lut_3_4(pyrcel_lut_path)
    else:
        raise ValueError(f"Unrecognized wspeed_type {wtype}; expected one of [0, 1, 2, 3, 4].")


def finalize_pyrcel_lut_setup(
    pyrcellut: xr.Dataset,
    ccn_densities: List[float],
    ccn_mact: List[float],
    recipe_file_path: str,
    include_w: List[str] | None = None,
) -> None:
    """
    Common finalization for all Pyrcel LUT setup variants.

    Parameters
    ----------
    pyrcellut : xr.Dataset
        Loaded (and potentially speed-interpolated) Pyrcel LUT.
    ccn_densities : list of float
        CCN species densities [kg m^-3], same length/order as LUT species.
    ccn_mact : list of float
        CCN mass activation thresholds (per species).
    recipe_file_path : str
        JSON file path containing CCN recipes keyed by "aero{spec_num}".
    include_w : list[str] | None
        Additional velocity coordinates to keep in dim order (e.g., ["w_mean", "w_prime"]).
    """
    include_w = include_w or []

    # Ensure a stable species order
    pyrcellut = pyrcellut.sortby("spec_num")

    aeronames = pyrcellut["name"].values
    aeroidxs = pyrcellut.spec_num.values

    aeronamemap: Dict[str, str] = {
        f"aero{spn}": str(pyrcellut.name.sel(spec_num=spn).item()) for spn in aeroidxs
    }
    pyrcnamemap: Dict[str, str] = {v: k for k, v in aeronamemap.items()}

    # Validate lengths
    if len(aeronames) != len(ccn_densities):
        raise ValueError(
            "ccn_densities length must match number of Pyrcel LUT species; "
            f"got {len(ccn_densities)} vs {len(aeronames)}."
        )
    if len(aeronames) != len(ccn_mact):
        raise ValueError(
            "ccn_mact length must match number of Pyrcel LUT species; "
            f"got {len(ccn_mact)} vs {len(aeronames)}."
        )

    # Build IFSAeroSpecs list in LUT order
    thislutaero = [
        IFSAeroSpecs(
            name,
            rmed,
            shape,
            dens,
            0.0,
            float(np.inf),
            np.array(1.0),
            mact,
            1,
        )
        for name, rmed, shape, dens, mact in zip(
            aeronames, pyrcellut["r_median"].values,
            pyrcellut["shape"].values, ccn_densities, ccn_mact
        )
    ]

    # Add 1-D coordinate aliases `aero{spec}` pointing to the CCN bins for convenience
    pyrcellut = pyrcellut.assign_coords(
        {f"aero{aerox}": pyrcellut[f"aero{aerox}_nccn"] for aerox in aeroidxs}
    )

    # Load recipes from JSON file
    with open(recipe_file_path, "r", encoding="utf-8") as recipe_file:
        ccn_recipes = json.load(recipe_file)

    thisrecipe: Dict[str, Dict[str, float]] = {}
    print("Populating recipes...")
    for ccn_idx in aeroidxs:
        ccn_aero = f"aero{ccn_idx}"
        if ccn_aero not in ccn_recipes:
            raise ValueError(
                f"CCN name {ccn_aero} from LUT was not found in recipe file {recipe_file_path}"
            )
        print(f"Adding {aeronamemap[ccn_aero]}...")
        thisrecipe[aeronamemap[ccn_aero]] = ccn_recipes[ccn_aero]

    config.PYRCELLUT = pyrcellut
    config.THISLUTAERO = thislutaero
    config.PYRCNAMEMAP = pyrcnamemap
    config.AERONAMEMAP = aeronamemap
    config.THISRECIPE = thisrecipe


def setup_pyrcel_lut_0(pyrcel_lut_path: str) -> None:
    """
    Setup Pyrcel LUT for wspeed_type 0 (monodisperse vertical velocity).

    Loads the LUT, verifies that `CONFIGDICT["wspeed"]` is within bounds,
    linearly interpolates over `w`, and finalizes globals.

    Parameters
    ----------
    pyrcel_lut_path : str
        LUT path (NetCDF/Zarr).

    Raises
    ------
    ValueError
        If `CONFIGDICT["wspeed"]` is out of LUT `w` range.
    """
    print(f"Opening LUT from {pyrcel_lut_path}")
    pyrcellut = xr.open_dataset(pyrcel_lut_path)

    if (CONFIGDICT["wspeed"] < pyrcellut.w.values.min()) or (
        CONFIGDICT["wspeed"] > pyrcellut.w.values.max()
    ):
        raise ValueError(
            f'vertical speed {CONFIGDICT["wspeed"]} m/s out of bounds for Pyrcel LUT!'
        )

    print(f'Linearly interpolating Pyrcel LUT for vertical speed {CONFIGDICT["wspeed"]} m/s')
    pyrcellut = pyrcellut.interp(w=CONFIGDICT["wspeed"], method="linear").compute()

    include_w = ["w"] if pyrcellut.w.shape != () else []
    finalize_pyrcel_lut_setup(
        pyrcellut,
        CONFIGDICT["ccn_densities"],
        CONFIGDICT["ccn_mact_def"],
        CONFIGDICT["ccn_recipe_file"],
        include_w,
    )


def setup_pyrcel_lut_1_2(pyrcel_lut_path: str) -> None:
    """
    Setup Pyrcel LUT for wspeed_type 1 or 2 (w_mean only, or w_mean + tendency).

    Loads the LUT, verifies that `CONFIGDICT["w_prime"]` is within bounds,
    linearly interpolates over `w_prime`, and finalizes globals.
    """
    print(f"Opening LUT from {pyrcel_lut_path}")
    pyrcellut = xr.open_dataset(pyrcel_lut_path)

    if (CONFIGDICT["w_prime"] < pyrcellut.w_prime.values.min()) or (
        CONFIGDICT["w_prime"] > pyrcellut.w_prime.values.max()
    ):
        raise ValueError(
            f'w_prime {CONFIGDICT["w_prime"]} m/s out of bounds for Pyrcel LUT!'
        )

    print(f'Linearly interpolating Pyrcel LUT for fixed w_prime {CONFIGDICT["w_prime"]} m/s')
    pyrcellut = pyrcellut.interp(w_prime=CONFIGDICT["w_prime"], method="linear").compute()

    finalize_pyrcel_lut_setup(
        pyrcellut,
        CONFIGDICT["ccn_densities"],
        CONFIGDICT["ccn_mact_def"],
        CONFIGDICT["ccn_recipe_file"],
        ["w_mean"],
    )


def setup_pyrcel_lut_3_4(pyrcel_lut_path: str) -> None:
    """
    Setup Pyrcel LUT for wspeed_type 3 or 4 (w_mean and w_prime with Deardorff scaling).

    This variant loads the LUT and keeps both `w_mean` and `w_prime` dimensions
    (no interpolation at setup time), relying on downstream logic to select/use
    the appropriate values for each point.
    """
    print(f"Opening LUT from {pyrcel_lut_path}")
    pyrcellut = xr.open_dataset(pyrcel_lut_path)

    print(
        "Using deardorff_scale: "
        f'{CONFIGDICT["deardorff_scale"]:.02f},\n'
        f"LUT w_mean range: {pyrcellut.w_mean.min().values:.02f}m/s-"
        f"{pyrcellut.w_mean.max().values:.02f}m/s\n"
        f"LUT w_prime range: {pyrcellut.w_prime.min().values:.02f}m/s-"
        f"{pyrcellut.w_prime.max().values:.02f}m/s"
    )

    finalize_pyrcel_lut_setup(
        pyrcellut,
        CONFIGDICT["ccn_densities"],
        CONFIGDICT["ccn_mact_def"],
        CONFIGDICT["ccn_recipe_file"],
        ["w_mean", "w_prime"],
    )


def get_actual_lut_recipes(
    represented_aeros: Iterable[str],
) -> Tuple[xr.Dataset, Dict[str, Dict[str, float]], List[str]]:
    """
    Restrict the LUT and recipes to only the aerosol species present in the data.

    Parameters
    ----------
    represented_aeros : iterable of str
        Names of aerosol variables actually present in the fields.

    Returns
    -------
    (actual_pyrcellut, actual_recipe, needed_aeros)
        - actual_pyrcellut : xr.Dataset
            Possibly reduced LUT (species dropped -> first index selected + dropped).
        - actual_recipe : dict
            Recipe only including species present in `represented_aeros`.
        - needed_aeros : list[str]
            Variables to keep from input (includes "pressure" + represented species).
    """

    var_to_keep: List[str] = ["pressure"]
    actual_recipe: Dict[str, Dict[str, float]] = {}

    print("Computing CCN species:", flush=True)
    lut_dims_to_exclude: List[str] = []

    represented_aeros_set = set(represented_aeros)
    for ccn_name, rec in config.THISRECIPE.items():
        this_ccn_present = any(aero in represented_aeros_set for aero in rec)

        if this_ccn_present:
            # Filter the recipe to present species
            actual_recipe[ccn_name] = {a: rec[a] for a in rec if a in represented_aeros_set}
            var_to_keep.extend(actual_recipe[ccn_name].keys())

            print(f"CCN species {ccn_name}:")
            print("\n".join([f"\t{k} {v:.2f}" for k, v in actual_recipe[ccn_name].items()]))
        else:
            print(f"Excluding PYRCEL LUT CCN species {ccn_name}")
            lut_dims_to_exclude.append(config.PYRCNAMEMAP[ccn_name])

    if lut_dims_to_exclude:
        actual_pyrcellut = config.PYRCELLUT.isel({dim: 0 for dim in lut_dims_to_exclude}, drop=True).load()
    else:
        actual_pyrcellut = config.PYRCELLUT.load()

    print("done!", flush=True)
    return actual_pyrcellut, actual_recipe, list(set(var_to_keep))
