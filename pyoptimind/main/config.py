import json
import os

import xarray as xr

global SCRIPTDIR, GETLUTVAL_LIB, VERTINTERP_LIB

SCRIPTDIR=os.path.dirname(os.path.realpath(__file__))
GETLUTVAL_LIB=os.path.join(SCRIPTDIR, "../../libs/shared/fget_lutval.so")
VERTINTERP_LIB=os.path.join(SCRIPTDIR, "../../libs/shared/fvertinterp.so")

global ERA5_DATADIR, AERO_DATADIR, MODIS_DATADIR
ERA5_DATADIR=os.path.join(SCRIPTDIR, "../../data/era5")
AERO_DATADIR=os.path.join(SCRIPTDIR, "../../data/cams")
MODIS_DATADIR=os.path.join(SCRIPTDIR, "../../data/modis")

global CONFIGDICT
CONFIGDICT = {
    "gridspec": "r30",
    "latitudes_minmax": [-90,90],
    "longitudes_minmax": [0,360],
    "cos_sza_minmax": [0,1],
    "localhour_minmax": [0,24],
    "hourly": "3hourly",
    "aerofromclimatology": False,
    "fixedaeromodellevel": None, #135 #129 # or None

    "nlevelsbelowcloudbase" : None,

    "aeros_out_of_cloud" : None, # None means all - it is the aerosol mmr picked at fixedaeromodellevel or below cloud base

    "aerosolclimfile": None,
    #"/home/papa/data/aerosol_cams_climatology_49r1_v2_4D_no_compression_classic",
    #aerosol_cams_climatology_43r3_v2_3D_no_compression_classic.nc
    #aerosol_cams_climatology_49r2_1951-2019_4D.nc
    #aerosol_cams_climatology_49r1camsround3_3D_no_compression_classic.nc

    "pyrcellutpath": None, #NO DEFAULT  "/home/papa/Documents/parapyrcel/outdir_luts/init_6specs_01su_w10.nc",
    #CONFIGDICT["pyrcellutpath"]="/home/papa/Documents/parapyrcel/lut_sepcarb_sepni_1w_q06withscaling_tunv2.nc"
    # wspeed_type
    # 0: fixed monodisperse speed
    # 1: w_mean=w_ls
    #    w_prime fixed (set "wprime" below)
    # 2: w_mean=w_ls + g/cp dT/dt (requires ml tendencies!),
    #    w_prime fixed (set "wprime" below)
    # 3: w_mean=w_ls,
    #    w_prime=deardorff_scale*wstar
    # 4: w_mean=w_ls + g/cp dT/dt (requires ml tendencies!),
    #    w_prime=deardorff_scale*wstar
    "wspeed_type" : 3,
    "w_prime" : None,# float for wspeed_type 1 and 2
    "w_prime_min" : 0.1,
    "w_mean_min" : -10,
    "wspeed": None, # For gen 0, with fixed monodisperse vertical speed
    "deardorff_scale": 0.4,
    "kinetically_limited": False,
    "scalemcon": False, # Makes sense only for prognostic areosols
    "scale_recipe_ingredient": None,
    "bindseasalt": True,
    "ss_coarsetofine_ratio": 10,

    "grosvenor_tau_c_correction": False,

    "firstguess_radii": None,
    "global_mass_scaler" : None,

    "modisndrefsample": "Q06", #"BR17" or "Q06"
    "modisndusemean" : False,
    "modisndvalidthr": 0.01,
    "samplespreads": ["Q06", "G18", "BR17"],

    "modisndbiascorrection": False,

    "cldetect_cc_threshold": 0.8,
    "cldetect_t_threshold": 268,
    "cldetect_iwr_threshold": 0.05,
    "cldetect_thresh_valid_monthly": 0.1,
    "tune_rain_dispersion": False,

    "weightbycloudpresence": False,

    "ccn_densities" : [1760, 1760, 2180, 2180, 1300, 1000],
    "ccn_mact_def"  : [0.7, 0.8, 0.9, 0.9, 0.7, 0.7],
    "ccn_recipe_file" : None,

    "nprocs" : 1,
    "use_zarr" : True,
}

# Global state variables
global SOME_AEROS_OUT_OF_CLOUD, PYRCELLUT, THISLUTAERO, PYRCNAMEMAP, THISRECIPE
SOME_AEROS_OUT_OF_CLOUD = False
PYRCELLUT = xr.Dataset(None)
THISLUTAERO = []
PYRCNAMEMAP = {}
AERONAMEMAP = {}
THISRECIPE = {}
global TMPFLDDIR
TMPFLDDIR = os.environ.get('TMPDIR', '/tmp') + 'fields'
os.makedirs(TMPFLDDIR, exist_ok=True)

# File signature constants
global ERA5MLFILESIGN, ERA5TENDFILESIGN, ERA5SFCFILESIGN, COPYFIELDS
ERA5MLFILESIGN = "ml_sel"
ERA5TENDFILESIGN = "tend_ml"
ERA5SFCFILESIGN = "sfc"
COPYFIELDS = False

global OPENDS_ZARR_KWARGS, SSRH80
OPENDS_ZARR_KWARGS = {
    "engine": "zarr",
    "chunks": {"time": "auto"},
    "consolidated": False
}

SSRH80 = True

def digest_config(config_path: str):
    """
    Load and validate configuration from JSON file.
    
    Reads a JSON configuration file and merges settings with CONFIGDICT,
    validating that all config keys exist in defaults. Prints status of
    each configuration option.
    
    Args:
        config_path: str - Path to configuration JSON file
    
    Raises:
        ValueError: If a config key is unknown (not in default CONFIGDICT)
    
    Side Effects:
        - Modifies global CONFIGDICT
        - Modifies global SOME_AEROS_OUT_OF_CLOUD
        - Prints configuration status to stdout
    """
    import json

    # Initialize global variables
    with open(config_path) as config_file:
        #config_logs = "\n".join(config_file.readlines())
        config_in = json.load(config_file)

    # Read configurations
    allkeys = list(set(CONFIGDICT.keys()) | config_in.keys())
    for key in allkeys:
        if (key in config_in) and (not key.startswith("other_")):
            CONFIGDICT[key] = config_in[key]
            if key not in CONFIGDICT:
                raise ValueError(f"config key {key} unknown. Verify spelling errors.")
            print(f"Set {key:>20} to {CONFIGDICT[key]}")
        elif key in CONFIGDICT:
            print(f"Using default value for {key}: {CONFIGDICT[key]}")

    if (CONFIGDICT["nlevelsbelowcloudbase"] is not None) and (CONFIGDICT["fixedaeromodellevel"] is not None):
        print("Warning! I will ignore fixedaeromodellevel setting because using nlevelsbelowcloudbase!")
        CONFIGDICT["fixedaeromodellevel"] = None
    SOME_AEROS_OUT_OF_CLOUD = (CONFIGDICT["nlevelsbelowcloudbase"] is not None) or (CONFIGDICT["fixedaeromodellevel"] is not None)

    # TODO: Implement proper check of wspeed_type
    if (CONFIGDICT["deardorff_scale"] is not None) and (CONFIGDICT["wspeed"] is not None):
        print(f"Warning! Setting deardorff_scale={CONFIGDICT['deardorff_scale']:.2f} overrides wspeed={CONFIGDICT['wspeed']:.2f} settings!")
        CONFIGDICT["wspeed"] = None
    print(f"Successfully read configuration from {config_path}")

def get_config():
    """Get current global configuration dictionary."""
    return CONFIGDICT