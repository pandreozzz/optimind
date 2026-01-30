"""Dask spillover disk management utilities."""
import os
import shutil
from ..main.config import CONFIGDICT


def get_spill_directory() -> str:
    """
    Get or create the dask spillover directory.
    
    Retrieves the spillover directory path from CONFIGDICT['dask_spill_dir'].
    If not configured, defaults to $TMPDIR/dask-spillover. Creates the
    directory if it doesn't exist.
    
    Returns:
        str: Path to the spillover directory
    """
    spill_dir = CONFIGDICT.get("dask_spill_dir")
    if spill_dir is None:
        spill_dir = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'dask-spillover')
    if not os.path.exists(spill_dir):
        os.makedirs(spill_dir, exist_ok=True)
    return spill_dir


def get_spill_usage() -> dict:
    """
    Get disk usage statistics for the spillover directory.
    
    Returns:
        dict: Dictionary with spillover disk usage including:
            - spill_dir: Path to spillover directory
            - total_gb: Total disk space in GB
            - used_gb: Disk space used in GB
            - free_gb: Free disk space in GB
            - percent: Percentage of disk used (0-100)
        If error occurs, returns dict with 'spill_dir' and 'error' keys
    """
    spill_dir = get_spill_directory()
    try:
        usage = shutil.disk_usage(spill_dir)
        return {
            "spill_dir": spill_dir,
            "total_gb": usage.total / (1024**3),
            "used_gb": usage.used / (1024**3),
            "free_gb": usage.free / (1024**3),
            "percent": 100 * usage.used / usage.total
        }
    except Exception as e:
        print(f"Warning: Could not get spill disk usage: {e}")
        return {"spill_dir": spill_dir, "error": str(e)}


def print_spill_status(label: str = ""):
    """
    Print current spillover disk usage statistics to stdout.
    
    Displays formatted spillover disk usage for monitoring disk pressure
    during large dataset processing with dask spillover.
    
    Args:
        label: Optional checkpoint label (e.g., "After aerosol load")
    """
    usage = get_spill_usage()
    if "error" not in usage:
        label_str = f" [{label}]" if label else ""
        print(f"Spill disk{label_str}: {usage['used_gb']:.1f}GB / {usage['total_gb']:.1f}GB " +
              f"(Free: {usage['free_gb']:.1f}GB, {usage['percent']:.1f}%)", flush=True)
    else:
        print(f"Spill dir: {usage['spill_dir']} (status unknown)", flush=True)
