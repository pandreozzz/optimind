"""File I/O and data staging utilities."""
import os
from glob import glob
from ..main.config import CONFIGDICT, TMPFLDDIR, COPYFIELDS
from ..main.config import ERA5_DATADIR, AERO_DATADIR, MODIS_DATADIR
from ..main.config import ERA5MLFILESIGN, ERA5TENDFILESIGN, ERA5SFCFILESIGN

def aero_pl_namelike(year, grid):
    """
    Generate a file glob pattern for aerosol pressure-level data.

    Args:
        year: int - Year for the data (e.g., 2020)
        grid: str - Grid name (e.g., "0p5x0p5")

    Returns:
        str: Glob pattern for aerosol files (aero_PL format)
    """
    return f"*_aero_{year}_{grid}_{CONFIGDICT['hourly']}_pl"


def aero_sfc_namelike(year, grid):
    """
    Generate a file glob pattern for aerosol surface data.

    Args:
        year: int - Year for the data (e.g., 2020)
        grid: str - Grid name (e.g., "0p5x0p5")

    Returns:
        str: Glob pattern for surface meteorology files (sfc format)
    """
    return f"*_meteo_{year}_{grid}_{CONFIGDICT['hourly']}_sfc"


def copy_all_files(year):
    """
    Copy or symlink input data files to working directory.

    Stages climatology/prognostic aerosol data and ERA5 meteorological fields
    from source directories to TMPDIR/fields/ for processing. Uses either rsync
    copy or symbolic links based on COPYFIELDS setting.

    Args:
        year: int - Year for data files (e.g., 2020)

    Side Effects:
        - Copies/links files to $TMPDIR/fields/
        - Updates LOGDIC with file names
        - Prints commands and operations to stdout
    """
    cpy_cmd = "rsync -aPh" if COPYFIELDS else "ln -sf"
    operation = "copying" if COPYFIELDS else "linking"

    dataext = "_zarr" if CONFIGDICT["use_zarr"] else ".nc"

    if CONFIGDICT["aerofromclimatology"]:
        print(f"{operation} climatology for aerosols...")
        cmd1 = f"{cpy_cmd} {CONFIGDICT['aerosolclimfile']} " +\
            f"{os.environ['TMPDIR']}/fields/aerosol_cams_climatology.nc"
        print(cmd1)
        os.system(cmd1)
    else:
        print(f"{operation} fields from prognostics...")
        aero_pl_filename = os.path.join(
            AERO_DATADIR,
            aero_pl_namelike(year, CONFIGDICT["gridspec"])
            ) + dataext
        # aero_sfc_name = os.path.join(
        #     AERO_DATADIR,
        #      aero_sfc_namelike(year, CONFIGDICT["gridspec"])
        #      ) + dataext

        aero_pl_filelist = glob(aero_pl_filename)
        if not aero_pl_filelist:
            raise ValueError(f"No files found with {aero_pl_filename}")
        #aero_sfc_filelist = glob.glob(aero_sfc_name)
        #if len(aero_sfc_filelist):
        #    raise ValueError(f"No files found with {aero_sfc_name}")

        aero_pl_filename = aero_pl_filelist[0]
       # aero_sfc_name = aero_sfc_filelist[0]

        cmd1 = f"{cpy_cmd} '{aero_pl_filename}' '{TMPFLDDIR}'/"
        print(cmd1)
        os.system(cmd1)

        #cmd2 = f"{cpy_cmd} {aero_sfc_filename} {TMPFLDDIR}/"
        #print(cmd2)
        #os.system(cmd2)

    print("Copying ERA5 fields...")

    era5_ml_file = os.path.join(
        ERA5_DATADIR,
        f"era5_{year}_{CONFIGDICT['gridspec']}_{CONFIGDICT['hourly']}_{ERA5MLFILESIGN}{dataext}"
        )
    era5_tend_ml_file = os.path.join(
        ERA5_DATADIR,
        f"era5_{year}_{CONFIGDICT['gridspec']}_{CONFIGDICT['hourly']}_{ERA5TENDFILESIGN}{dataext}"
        )
    era5_sfc_file = os.path.join(
        ERA5_DATADIR,
        f"era5_{year}_{CONFIGDICT['gridspec']}_{CONFIGDICT['hourly']}_{ERA5SFCFILESIGN}{dataext}"
        )

    for file_check in [era5_ml_file, era5_sfc_file]:
        if not os.path.exists(file_check):
            raise ValueError(f"{file_check} does not exist!")

    cmd1 = f"{cpy_cmd} '{era5_ml_file}' '{TMPFLDDIR}'/"
    print(cmd1)
    os.system(cmd1)

    cmd2 = f"{cpy_cmd} '{era5_tend_ml_file}' '{TMPFLDDIR}'/"
    print(cmd2)
    os.system(cmd2)

    cmd3 = f"{cpy_cmd} '{era5_sfc_file}' '{TMPFLDDIR}'/"
    print(cmd3)
    os.system(cmd3)

    print("Copying MODIS fields...")
    modis_nd_file = os.path.join(
        MODIS_DATADIR,
        f"modis_nd_nd13.monthlymeans_{year:4d}.AT.v1_{CONFIGDICT['gridspec']}.nc"
        )
    if not os.path.exists(modis_nd_file):
        raise ValueError(f"{modis_nd_file} does not exist")

    cmd1 = f"{cpy_cmd} '{modis_nd_file}' '{TMPFLDDIR}'/"
    print(cmd1)
    os.system(cmd1)
