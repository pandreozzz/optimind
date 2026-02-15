"""Convert grib/netcdf files to zarr with optional chunk controls."""

import argparse
import os
import sys
from typing import Optional, Sequence

import xarray as xr


SUPPORTED_EXTS = {".grib", ".nc", ".nc4", ".grb", ".grib2"}


def convert_to_zarr(
    fname_in: str,
    dirname_out: Optional[str] = None,
    *,
    open_chunks: str | dict | None = "auto",
    no_lev_chunk: bool = True,
    lev_dim: str = "lev",
    overwrite: bool = False,
) -> str:
    """Convert a supported input file to zarr.

    Returns the output zarr directory path.
    """
    fname_in = os.path.realpath(fname_in)
    basename, inext = os.path.splitext(fname_in)
    if inext not in SUPPORTED_EXTS:
        raise ValueError(f"Extension {inext} not supported")

    if dirname_out is None:
        dirname_out = f"{basename}_zarr"

    if os.path.exists(dirname_out) and not overwrite:
        raise FileExistsError(f"Output directory exists: {dirname_out}. Use --overwrite.")

    os.makedirs(dirname_out, exist_ok=True)
    print(f"converting {fname_in} to {dirname_out}...", flush=True)

    ds = xr.open_dataset(fname_in, chunks=open_chunks)
    if no_lev_chunk and lev_dim in ds.dims:
        ds = ds.chunk({lev_dim: -1})

    ds.to_zarr(dirname_out, mode="w")
    return dirname_out


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for standalone usage."""
    parser = argparse.ArgumentParser(description="Convert an input file to zarr")
    parser.add_argument("input", help="Input file (.grib/.nc/.grb/.grib2)")
    parser.add_argument("-o", "--output", default=None, help="Output zarr directory")
    parser.add_argument(
        "--open-chunks",
        default="auto",
        choices=["auto", "stored", "none"],
        help="Chunking at open: auto, stored ({}), or none",
    )
    parser.add_argument("--no-lev-chunk", action="store_true", help="Store lev as one chunk")
    parser.add_argument("--lev-dim", default="lev", help="Vertical dimension name")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output")
    return parser.parse_args(argv)


def _resolve_open_chunks(choice: str) -> str | dict | None:
    """Map the CLI open-chunks option to xarray.open_dataset chunks value."""
    mapping: dict[str, str | dict | None] = {"auto": "auto", "stored": {}, "none": None}
    if choice not in mapping:
        raise ValueError(f"Unknown open_chunks choice: {choice}")
    return mapping[choice]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    args = _parse_args(argv)
    convert_to_zarr(
        fname_in=args.input,
        dirname_out=args.output,
        open_chunks=_resolve_open_chunks(args.open_chunks),
        no_lev_chunk=args.no_lev_chunk,
        lev_dim=args.lev_dim,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
