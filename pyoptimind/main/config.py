"""Global configuration and helpers for the tuning workflow."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import xarray as xr

# ---------------------------------------------------------------------
# Paths (computed from script location)
# ---------------------------------------------------------------------
SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
GETLUTVAL_LIB = Path(os.path.join(SCRIPTDIR, "../../libs/shared/fget_lutval.so")).resolve()
VERTINTERP_LIB = Path(os.path.join(SCRIPTDIR, "../../libs/shared/fvertinterp.so")).resolve()

ERA5_DATADIR = Path(os.path.join(SCRIPTDIR, "../../data/era5")).resolve()
AERO_DATADIR = Path(os.path.join(SCRIPTDIR, "../../data/cams")).resolve()
MODIS_DATADIR = Path(os.path.join(SCRIPTDIR, "../../data/modis")).resolve()

PYRCELLUT_DATADIR = Path(os.path.join(SCRIPTDIR, "../../data/pyrcellut")).resolve()
RECIPES_DEFDIR = Path(os.path.join(SCRIPTDIR, "../../setup_files/recipes")).resolve()
# Temporary fields directory (ensure trailing separator via os.path.join)
TMPFLDDIR = os.path.join(os.environ.get("TMPDIR", "/tmp"), "fields")
os.makedirs(TMPFLDDIR, exist_ok=True)

# ---------------------------------------------------------------------
# Global configuration dictionary (defaults)
# ---------------------------------------------------------------------
CONFIGDICT: Dict[str, Any] = {
    "gridspec": "r30",
    "latitudes_minmax": [-90, 90],
    "longitudes_minmax": [0, 360],
    "cos_sza_minmax": [0, 1],
    "localhour_minmax": [0, 24],
    "hourly": "3hourly",
    "aerofromclimatology": False,
    "fixedaeromodellevel": None,  # 135 or 129 or None
    "nlevelsbelowcloudbase": None,
    # None means: use aerosol mmr at fixedaeromodellevel or below cloud base
    "aeros_out_of_cloud": None,
    "aerosolclimfile": None,
    "pyrcellutpath": None,  # REQUIRED by calling code
    # wspeed_type:
    # 0: fixed monodisperse speed
    # 1: w_mean = w_ls; w_prime fixed ("w_prime")
    # 2: w_mean = w_ls + g/cp dT/dt (needs ml tendencies); w_prime fixed
    # 3: w_mean = w_ls; w_prime = deardorff_scale * wstar
    # 4: w_mean = w_ls + g/cp dT/dt; w_prime = deardorff_scale * wstar
    "wspeed_type": 3,
    "w_prime": None,  # float for wspeed_type 1 and 2
    "w_prime_min": 0.1,
    "w_mean_min": -10,
    "wspeed": None,  # for gen 0, fixed monodisperse vertical speed
    "deardorff_scale": 0.4,
    "kinetically_limited": False,
    "scalemcon": False,  # Makes sense only for prognostic aerosols
    "scale_recipe_ingredient": None,
    "bindseasalt": True,
    "ss_coarsetofine_ratio": 10,
    "grosvenor_tau_c_correction": False,
    "firstguess_radii": None,
    "global_mass_scaler": None,  # mapping {species: factor}
    "modisndrefsample": "Q06",  # "BR17" or "Q06"
    "modisndusemean": False,
    "modisndvalidthr": 0.01,
    "samplespreads": ["Q06", "G18", "BR17"],
    "modisndbiascorrection": False,
    "cldetect_cc_threshold": 0.8,
    "cldetect_t_threshold": 268,
    "cldetect_iwr_threshold": 0.05,
    "cldetect_thresh_valid_monthly": 0.1,
    "useverheggenactivfrac": False,
    "tune_rain_dispersion": False,
    "weightbycloudpresence": False,
    "ccn_densities": [1760, 1760, 2180, 2180, 1300, 1000],
    "ccn_mact_def": [0.7, 0.8, 0.9, 0.9, 0.7, 0.7],
    "ccn_recipe_file": None,
    "nprocs": 1,
    "use_zarr": True,
}

# Global state variables (kept for compatibility with existing code)
SOME_AEROS_OUT_OF_CLOUD = False
PYRCELLUT = xr.Dataset(None)
THISLUTAERO: List[Any] = []
PYRCNAMEMAP: Dict[str, Any] = {}
AERONAMEMAP: Dict[str, Any] = {}
THISRECIPE: Dict[str, Any] = {}

# Default ERA5 file signatures
ERA5MLFILESIGN = "ml_sel"
ERA5TENDFILESIGN = "tend_ml"
ERA5SFCFILESIGN = "sfc"
COPYFIELDS = False

# IO defaults
OPENDS_ZARR_KWARGS: Dict[str, Any] = {
    "engine": "zarr",
    "chunks": {"time": "auto"},
    "consolidated": False,
}
SSRH80 = True


def digest_config(config_path: str) -> None:
    """
    Load and validate configuration from a JSON file and merge into CONFIGDICT.

    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file.

    Raises
    ------
    ValueError
        If a config key in the file is not present in the default CONFIGDICT.
    """
    # Read configurations
    with open(config_path, "r", encoding="utf-8") as config_file:
        cfg_in: Dict[str, Any] = json.load(config_file)

    # Validate keys: anything not in defaults (and not prefixed with "other_") is an error
    for key in cfg_in:
        if not key.startswith("other_") and key not in CONFIGDICT:
            raise ValueError(
                f"config key {key} unknown. Verify spelling errors."
            )

    # Merge (excluding "other_*" keys which are ignored by design)
    for key, val in cfg_in.items():
        if key.startswith("other_"):
            continue
        CONFIGDICT[key] = val
        print(f"Set {key:>20} to {CONFIGDICT[key]}")

    # Print defaults for keys not present in input
    for key, val in CONFIGDICT.items():
        if key not in cfg_in and not key.startswith("other_"):
            print(f"Using default value for {key}: {val}")

    # Cross-field consistency
    if (
        CONFIGDICT["nlevelsbelowcloudbase"] is not None
        and CONFIGDICT["fixedaeromodellevel"] is not None
    ):
        print(
            "Warning! Ignoring fixedaeromodellevel because nlevelsbelowcloudbase is set!"
        )
        CONFIGDICT["fixedaeromodellevel"] = None

    # Flag for “aeros out of cloud”
    global SOME_AEROS_OUT_OF_CLOUD  # pylint: disable=global-statement
    SOME_AEROS_OUT_OF_CLOUD = (
        CONFIGDICT["nlevelsbelowcloudbase"] is not None
        or CONFIGDICT["fixedaeromodellevel"] is not None
    )

    # Deardorff scale vs wspeed
    if CONFIGDICT["deardorff_scale"] is not None and CONFIGDICT["wspeed"] is not None:
        print(
            f"Warning! Setting deardorff_scale={CONFIGDICT['deardorff_scale']:.2f} "
            f"overrides wspeed={CONFIGDICT['wspeed']}."
        )
        CONFIGDICT["wspeed"] = None

    if not os.path.exists(CONFIGDICT["pyrcellutpath"]):
        corrected_pyrcellutpath = os.path.join(PYRCELLUT_DATADIR, CONFIGDICT["pyrcellutpath"])
        if os.path.exists(corrected_pyrcellutpath):
            print(f"considering pyrcellutpath as relative to {PYRCELLUT_DATADIR}")
            CONFIGDICT["pyrcellutpath"] = corrected_pyrcellutpath
        else:
            raise ValueError(f"Could not find {CONFIGDICT['pyrcellutpath']}")

    if not os.path.exists(CONFIGDICT["ccn_recipe_file"]):
        corrected_ccn_recipe_file = os.path.join(RECIPES_DEFDIR, CONFIGDICT["ccn_recipe_file"])
        if os.path.exists(corrected_pyrcellutpath):
            print(f"considering pyrcellutpath as relative to {RECIPES_DEFDIR}")
            CONFIGDICT["ccn_recipe_file"] = corrected_ccn_recipe_file
        else:
            raise ValueError(f"Could not find {CONFIGDICT['ccn_recipe_file']}")

    print(f"Successfully read configuration from {config_path}")


def get_config() -> Dict[str, Any]:
    """Return the current global configuration dictionary."""
    return CONFIGDICT
