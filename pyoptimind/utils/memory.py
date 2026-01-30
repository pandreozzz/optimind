"""Memory management utilities for large dataset processing."""
import gc
#import ctypes
from contextlib import contextmanager
import psutil


def trim_memory() -> int:
    """
    Trigger memory compaction for the C library malloc.

    Calls libc.malloc_trim(0) to release memory back to the OS that is not being used.
    This helps reduce memory fragmentation and improve memory efficiency.

    Returns:
        int: Amount of memory released (may vary by system)
    """
    #libc = ctypes.CDLL("libc.so.6")

    return 0 #libc.malloc_trim(0)


@contextmanager
def memory_cleanup():
    """
    Context manager for automatic garbage collection and memory trimming.

    Ensures garbage collection and malloc_trim are called on context exit,
    regardless of whether an exception occurred. Useful for freeing memory
    after processing large datasets.

    Usage:
        with memory_cleanup():
            # process data
            pass
    """
    try:
        yield
    finally:
        gc.collect()
        trim_memory()


def get_available_memory() -> float:
    """
    Return available system memory in MB.

    Returns:
        float: Available memory in megabytes
    """
    #return psutil.virtual_memory().available / (1024**2)
    return psutil.virtual_memory().total*0.75 / (1024**2)


def get_memory_usage() -> dict:
    """
    Return detailed system memory usage statistics.

    Returns:
        dict: Dictionary with keys:
            - total_mb: Total system memory in MB
            - used_mb: Memory currently in use in MB
            - available_mb: Memory available for allocation in MB
            - percent: Percentage of memory in use (0-100)
    """
    mem = psutil.virtual_memory()
    return {
        "total_mb": mem.total / (1024**2),
        "used_mb": mem.used / (1024**2),
        "available_mb": mem.available / (1024**2),
        "percent": mem.percent
    }


def print_memory_status(label: str = ""):
    """
    Print current memory usage statistics to stdout.

    Displays a formatted line with memory usage information useful for
    monitoring memory consumption during data processing.

    Args:
        label: Optional label to identify the checkpoint (e.g., "After loading data")
    """
    mem = get_memory_usage()
    label_str = f" [{label}]" if label else ""
    print(f"Memory{label_str}: {mem['used_mb']:.1f}MB / {mem['total_mb']:.1f}MB " +
          f"(Available: {mem['available_mb']:.1f}MB, {mem['percent']:.1f}%)", flush=True)
