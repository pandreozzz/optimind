"""Memory management utilities for large dataset processing."""
import os
import gc
#import ctypes
from contextlib import contextmanager
from typing import Optional
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
        Return an estimated memory budget for this process in MB.

        Behavior:
        - If running under a cgroup memory limit (common for Slurm cgroups or containers),
            use that limit.
        - Else if Slurm provides an explicit memory limit via environment variables, use it.
        - Else, for normal (non-Slurm) execution, use total host memory.

        A safety factor is applied to leave headroom for the OS and overhead.

    Returns:
        float: Estimated memory budget in megabytes
    """
    safety_fraction = 0.75
    host_mem = psutil.virtual_memory()
    host_total = float(host_mem.total)
    host_available = float(host_mem.available)

    cgroup_limit = _get_cgroup_memory_limit_bytes()
    slurm_limit = _get_slurm_memory_limit_bytes()

    effective_total: float
    limits = [v for v in (cgroup_limit, slurm_limit) if v is not None and v > 0]
    if limits:
        effective_total = float(min(limits))
    else:
        effective_total = host_available if _in_slurm_job() else host_total

    return (effective_total * safety_fraction) / (1024**2)


def _in_slurm_job() -> bool:
    return bool(os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_STEP_ID"))


def _get_slurm_memory_limit_bytes() -> Optional[int]:
    """Best-effort Slurm memory limit detection.

    Slurm commonly exports memory limits in MB via:
    - SLURM_MEM_PER_NODE
    - SLURM_MEM_PER_CPU (requires a CPU count)
    """
    mem_per_node_mb = os.environ.get("SLURM_MEM_PER_NODE")
    if mem_per_node_mb:
        try:
            return int(mem_per_node_mb) * 1024 * 1024
        except ValueError:
            pass

    mem_per_cpu_mb = os.environ.get("SLURM_MEM_PER_CPU")
    if not mem_per_cpu_mb:
        return None

    try:
        per_cpu = int(mem_per_cpu_mb)
    except ValueError:
        return None

    cpu_candidates = [
        os.environ.get("SLURM_CPUS_PER_TASK"),
        os.environ.get("SLURM_CPUS_ON_NODE"),
        os.environ.get("SLURM_JOB_CPUS_PER_NODE"),
    ]

    cpus: Optional[int] = None
    for candidate in cpu_candidates:
        if not candidate:
            continue
        # SLURM_JOB_CPUS_PER_NODE can be like: "32(x2)". We only need the first integer.
        try:
            cpus = int(str(candidate).split("(")[0])
            break
        except ValueError:
            continue

    if not cpus or cpus <= 0:
        return None

    return int(per_cpu * cpus) * 1024 * 1024


def _get_cgroup_memory_limit_bytes() -> Optional[int]:
    """Return cgroup memory limit in bytes if one applies.

    Supports both cgroup v2 (memory.max) and v1 (memory.limit_in_bytes).
    Returns None when no meaningful limit is detected.
    """
    try:
        # cgroup v2
        if os.path.exists("/sys/fs/cgroup/cgroup.controllers"):
            cg_path = _get_cgroup_v2_relative_path()
            if cg_path is None:
                return None
            full = os.path.join("/sys/fs/cgroup", cg_path.lstrip("/"))
            limit = _get_cgroup_v2_effective_limit(full)
            return _normalize_limit(limit) if limit is not None else None

        # cgroup v1
        cg_path = _get_cgroup_v1_relative_path(controller="memory")
        if cg_path is None:
            return None
        full = os.path.join("/sys/fs/cgroup/memory", cg_path.lstrip("/"))
        limit = _read_cgroup_int(os.path.join(full, "memory.limit_in_bytes"))
        if limit is None:
            return None
        return _normalize_limit(limit)
    except OSError:
        return None


def _normalize_limit(limit_bytes: int) -> Optional[int]:
    # Some systems report huge sentinel values when effectively unlimited.
    if limit_bytes <= 0:
        return None
    if limit_bytes >= (1 << 60):
        return None
    return int(limit_bytes)


def _get_cgroup_v2_effective_limit(cgroup_dir: str) -> Optional[int]:
    """Return the effective cgroup v2 memory limit.

    In cgroup v2, a child cgroup can have memory.max == 'max' while an ancestor
    cgroup enforces a limit. Walk upwards until a concrete limit is found.
    """
    root = os.path.abspath("/sys/fs/cgroup")
    current = os.path.abspath(cgroup_dir)

    while True:
        if not current.startswith(root):
            return None

        limit = _read_cgroup_int(os.path.join(current, "memory.max"), allow_max=True)
        if limit is not None:
            return limit

        if current == root:
            return None

        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def _get_cgroup_v2_relative_path() -> Optional[str]:
    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # v2 format: 0::/some/path
                parts = line.split(":", 2)
                if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
                    return parts[2]
    except OSError:
        return None
    return None


def _get_cgroup_v1_relative_path(controller: str) -> Optional[str]:
    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # v1 format: hierarchy:controllers:path
                parts = line.split(":", 2)
                if len(parts) != 3:
                    continue
                controllers = parts[1].split(",")
                if controller in controllers:
                    return parts[2]
    except OSError:
        return None
    return None


def _read_cgroup_int(path: str, *, allow_max: bool = False) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
    except OSError:
        return None

    if allow_max and raw == "max":
        return None

    try:
        return int(raw)
    except ValueError:
        return None


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
