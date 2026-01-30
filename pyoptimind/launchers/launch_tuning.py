#!/usr/bin/env python3
"""Command-line interface for cloud optical property LUT tuning."""
import argparse
import logging
from dask.distributed import Client
from pathlib import Path

from ..main.config import digest_config
from ..utils.memory import get_available_memory
from ..utils.dask import optimize_dask_for_memory


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

LOGDIC = {}

def main():
    """Main CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Optimize aerosol size distributions using pyrcel and MODIS Nd data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process year with configuration config.json
  python run_tuning.py --year 2020 --config config.json --logdir /tmp/logdir

  # Show configuration options and quit
  python run_tuning.py --year 2020 --config config.json --logdir /tmp/logdir --show-config
        """
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to process (e.g., 2020)"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file"
    )

    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show default configuration options and exit"
    )

    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="Path to directory for logs"
    )

    parser.add_argument(
        "--num-procs",
        type=int,
        default=4,
        help="Number of workers for the Dask local cluster"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Show default config
    if args.show_config:
        from ..main.config import CONFIGDICT
        print("Default configuration options:\n")
        for key, value in sorted(CONFIGDICT.items()):
            print(f"  {key:30} : {value}")
        return 0

    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Configuration file not found: {config_path}")

    logdir_path = Path(args.logdir)
    if not logdir_path.exists():
        parser.error(f"Logdir path not found: {logdir_path}")

    # Digesting config
    logger.info(f"Loading configuration from {config_path}")
    digest_config(str(config_path))

    # Import here to avoid import errors if tuning dependencies aren't installed
    from ..main.tune_driver import run_tuning_year

    # Process years
    logger.info(f"Processing year {args.year}")

    with optimize_dask_for_memory():
        totmem_mbytes = get_available_memory()
        print(f"Total memory {totmem_mbytes:.2f}MB")
        with Client(
            n_workers=args.num_procs,
            memory_limit=f"{totmem_mbytes*0.98/args.num_procs:.2f}MB"
            ):
            run_tuning_year(args.year, str(config_path), str(logdir_path))

    logger.info("All years processed successfully!")
    return 0


