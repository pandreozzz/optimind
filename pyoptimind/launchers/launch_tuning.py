#!/usr/bin/env python3
"""Command‑line interface for cloud optical property LUT tuning."""

import argparse
import logging
from pathlib import Path

from dask.distributed import Client

from ..main.config import CONFIGDICT, digest_config
from ..utils.daskctrl import optimize_dask_for_memory
from ..utils.memory import get_available_memory

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Kept for compatibility with modules that import it (filled elsewhere)
LOGDIC: dict = {}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Optimize aerosol size distributions using pyrcel and MODIS Nd data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\ntbdone"
        ),
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to process (e.g., 2020)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show default configuration options and exit",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="Path to directory for logs",
    )
    parser.add_argument(
        "--num-procs",
        type=int,
        default=4,
        help="Number of workers for the Dask local cluster",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)



    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Configuration file not found: {config_path}")

    logdir_path = Path(args.logdir)
    if not logdir_path.exists():
        parser.error(f"Logdir path not found: {logdir_path}")

    # Load configuration
    LOGGER.info("Loading configuration from %s", config_path)
    digest_config(str(config_path))

    # Show default config if requested
    if args.show_config:

        print("Default configuration options:\n")
        for key, value in sorted(CONFIGDICT.items()):
            print(f"{key:30} : {value}")
        return 0

    # Lazy import to avoid heavy deps at import time
    from ..main.tune_driver import run_tuning_year

    # Process the requested year
    LOGGER.info("Processing year %s", args.year)

    with optimize_dask_for_memory():
        totmem_mbytes = get_available_memory()
        print(f"Total memory {totmem_mbytes:.2f}MB")
        mem_per_worker_mb = max(totmem_mbytes * 0.98 / max(args.num_procs, 1), 256.0)

        with Client(n_workers=args.num_procs, memory_limit=f"{mem_per_worker_mb:.2f}MB"):
            # Temporary
            CONFIGDICT["nprocs"] = args.num_procs
            run_tuning_year(args.year, str(config_path), str(logdir_path))

    LOGGER.info("All years processed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
