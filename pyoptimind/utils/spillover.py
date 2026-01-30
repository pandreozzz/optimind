"""Dask spillover disk management utilities."""

from __future__ import annotations

import os
import shutil
from typing import Dict, Any

from ..main.config import CONFIGDICT


def get_spill_directory() -> str:
    """
    Get or create the Dask spillover directory.

    Returns
    -------
    str
        Path to spillover directory.
    """
    spill_dir = CONFIGDICT.get("dask_spill_dir")
    if spill_dir is None:
        spill_dir = os.path.join(
            os.environ.get("TMPDIR", "/tmp"),
            "dask-spillover",
        )

    if not os.path.exists(spill_dir):
        os.makedirs(spill_dir, exist_ok=True)

    return spill_dir


def get_spill_usage() -> Dict[str, Any]:
    """
    Get disk usage statistics for the spillover directory.

    Returns
    -------
    dict
        {
            "spill_dir": str,
            "total_gb": float,
            "used_gb": float,
            "free_gb": float,
            "percent": float
        }
        or
        {
            "spill_dir": str,
            "error": str
        }
    """
    spill_dir = get_spill_directory()
    try:
        usage = shutil.disk_usage(spill_dir)
        return {
            "spill_dir": spill_dir,
            "total_gb": usage.total / (1024 ** 3),
            "used_gb": usage.used / (1024 ** 3),
            "free_gb": usage.free / (1024 ** 3),
            "percent": 100 * usage.used / usage.total,
        }
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Warning: Could not get spill disk usage: {exc}")
        return {"spill_dir": spill_dir, "error": str(exc)}


def print_spill_status(label: str = "") -> None:
    """
    Print spillover disk usage.

    Parameters
    ----------
    label : str, optional
        Optional checkpoint tag.
    """
    usage = get_spill_usage()
    if "error" not in usage:
        label_str = f" [{label}]" if label else ""
        print(
            f"Spill disk{label_str}: {usage['used_gb']:.1f}GB / {usage['total_gb']:.1f}GB "
            f"(Free: {usage['free_gb']:.1f}GB, {usage['percent']:.1f}%)",
            flush=True,
        )
    else:
        print(f"Spill dir: {usage['spill_dir']} (status unknown)", flush=True)
