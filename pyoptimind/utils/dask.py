"""Dask configuration and optimization utilities."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator, Optional

from dask import config as dask_config

from ..main.config import CONFIGDICT


@contextmanager
def optimize_dask_for_memory(
    chunk_size_mb: int = 50,
    spill_to_disk: bool = True,
    spill_dir: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Context manager to optimize Dask worker configuration for large dataset processing.

    Configures distributed worker memory management with spilling to disk when memory
    pressure becomes high. Prevents out‑of‑memory errors by automatically moving data
    to disk.

    Parameters
    ----------
    chunk_size_mb : int
        Target array chunk size in MiB.
    spill_to_disk : bool
        Whether disk spillover is enabled.
    spill_dir : str, optional
        Directory for spillover files. If None, uses CONFIGDICT or $TMPDIR/dask-spillover.

    Yields
    ------
    str
        Spillover directory path.
    """
    if spill_dir is None:
        spill_dir = CONFIGDICT.get("dask_spill_dir")

    if spill_dir is None:
        spill_dir = os.path.join(
            os.environ.get("TMPDIR", "/tmp"),
            "dask-spillover",
        )

    if not os.path.exists(spill_dir):
        os.makedirs(spill_dir, exist_ok=True)
        print(f"Created dask spillover directory: {spill_dir}", flush=True)

    try:
        with dask_config.set(
            {
                "distributed.worker.memory.spill": 0.95,
                "distributed.worker.memory.pause": 0.97,
                "distributed.worker.memory.terminate": 0.99,
                "distributed.worker.memory.spill-disk": spill_dir
                if spill_to_disk
                else None,
                "array.chunk-size": f"{chunk_size_mb}MiB",
            }
        ):
            yield spill_dir
    finally:
        # Nothing to clean up, but required to maintain contextmanager semantics
        pass
