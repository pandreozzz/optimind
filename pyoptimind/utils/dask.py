"""Dask configuration and optimization utilities."""
import os
from contextlib import contextmanager
from .memory import get_available_memory
from ..main.config import CONFIGDICT


@contextmanager
def optimize_dask_for_memory(chunk_size_mb: int = 50, spill_to_disk: bool = True, spill_dir = None):
    """
    Context manager to optimize dask worker configuration for large dataset processing.
    
    Configures dask distributed worker memory management with spilling to disk
    when memory pressure becomes high. This prevents out-of-memory errors by
    automatically moving data to disk when needed.
    
    Args:
        chunk_size_mb: Target chunk size in MB. If None, auto-estimates from available memory.
        spill_to_disk: Enable spilling to disk when memory pressure is high (default: True).
        spill_dir: Directory for disk spillover. If None, uses CONFIGDICT['dask_spill_dir']
                   or defaults to $TMPDIR/dask-spillover.
    
    Yields:
        str: Path to the spillover directory
    
    Note:
        Dask will pause processing at 97% memory and terminate workers at 99%.
    """
    import dask
    from dask import config as dask_config
    
    if spill_dir is None:
        spill_dir = CONFIGDICT.get("dask_spill_dir")
        if spill_dir is None:
            spill_dir = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'dask-spillover')
    
    # Create spillover directory if it doesn't exist
    if not os.path.exists(spill_dir):
        os.makedirs(spill_dir, exist_ok=True)
        print(f"Created dask spillover directory: {spill_dir}", flush=True)
    
    try:
        with dask_config.set({
            'distributed.worker.memory.spill': 0.95,  # Spill to disk at 95%
            'distributed.worker.memory.pause': 0.97,  # Pause at 97%
            'distributed.worker.memory.terminate': 0.99,  # Terminate at 99%
            'distributed.worker.memory.spill-disk': spill_dir,  # Spill location
            'array.chunk-size': f'{chunk_size_mb}MiB',
        }):
            yield spill_dir
    finally:
        pass
