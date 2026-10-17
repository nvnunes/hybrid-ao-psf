"""Command-line entry point for one standalone Hybrid AO PSF simulation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .standalone import run_configuration


def main(argv: Sequence[str] | None = None) -> int:
    """Run one combined INI file and write one HDF5 result."""
    parser = argparse.ArgumentParser(
        prog="hybrid-ao-psf",
        description="Run one MASTSEL-backed Hybrid AO PSF simulation.",
    )
    parser.add_argument(
        "configuration", type=Path, help="combined Hybrid-and-MASTSEL INI"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="override the output path declared in [hybrid]",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing result file",
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "float64"),
        help="override [hybrid] precision",
    )
    arguments = parser.parse_args(argv)
    try:
        run_configuration(
            arguments.configuration,
            output=arguments.output,
            overwrite=arguments.overwrite,
            dtype=arguments.precision,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"hybrid-ao-psf: {type(exc).__name__}: {exc}\n")
    return 0


__all__ = ["main"]
