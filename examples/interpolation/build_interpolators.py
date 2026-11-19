"""Build Hybrid interpolators from supplied science and NGS HO PSF libraries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy import units as u
from psf_fits import read_psf_samples

from hybrid_ao_psf import (
    PsfSamples,
    build_psf_interpolator,
    build_psf_metric_interpolator_from_psfs,
    save_psf_interpolator,
    save_psf_metric_interpolator,
)


def main(
    science_dir: Path, ngs_dir: Path, output_dir: Path, *, build_ngs_psf: bool = True
) -> None:
    """Read both PSF libraries and save the interpolators used by Hybrid."""
    output_dir.mkdir(parents=True, exist_ok=True)

    science_samples = _read_psf_library(science_dir)
    science_interpolator = build_psf_interpolator(science_samples)
    save_psf_interpolator(
        science_interpolator, output_dir / "science.pkl", overwrite=True
    )

    ngs_samples = _read_psf_library(ngs_dir)
    if np.any(ngs_samples.wavelength != ngs_samples.wavelength[0]):
        raise ValueError(f"{ngs_dir}: NGS cubes must share one sensing wavelength.")

    ngs_metric_interpolator = build_psf_metric_interpolator_from_psfs(ngs_samples)
    save_psf_metric_interpolator(
        ngs_metric_interpolator, output_dir / "ngs-metrics.pkl", overwrite=True
    )

    if build_ngs_psf:
        ngs_psf_interpolator = build_psf_interpolator(ngs_samples)
        save_psf_interpolator(
            ngs_psf_interpolator, output_dir / "ngs.pkl", overwrite=True
        )


def _read_psf_library(directory: Path) -> PsfSamples:
    """Assemble samples from the FITS cubes in a directory."""
    paths = sorted(directory.glob("*.fits"))
    if not paths:
        raise ValueError(f"{directory} contains no FITS PSF cubes.")

    cubes = [read_psf_samples(path) for path in paths]
    reference = cubes[0]
    for path, cube in zip(paths[1:], cubes[1:], strict=True):
        if not np.array_equal(cube.x, reference.x) or not np.array_equal(
            cube.y, reference.y
        ):
            raise ValueError(f"{path}: field points differ from {paths[0]}.")
        if cube.psfs.shape != reference.psfs.shape:
            raise ValueError(f"{path}: PSF dimensions differ from {paths[0]}.")
        if cube.tel_diameter != reference.tel_diameter or not np.array_equal(
            cube.tel_pupil, reference.tel_pupil
        ):
            raise ValueError(f"{path}: telescope or pupil differs from {paths[0]}.")

    return PsfSamples(
        psfs=np.stack([cube.psfs for cube in cubes]),
        x=reference.x,
        y=reference.y,
        wavelength=u.Quantity([cube.wavelength for cube in cubes]),
        pixel_scale=u.Quantity([cube.pixel_scale for cube in cubes]),
        zenith_angle=u.Quantity([cube.zenith_angle for cube in cubes]),
        tel_diameter=reference.tel_diameter,
        tel_pupil=reference.tel_pupil,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--science-psfs",
        required=True,
        type=Path,
        help="Directory of science FITS cubes.",
    )
    parser.add_argument(
        "--ngs-psfs", required=True, type=Path, help="Directory of NGS FITS cubes."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--skip-ngs-psf", action="store_true", help="Skip the NGS image interpolator."
    )
    args = parser.parse_args()
    main(
        args.science_psfs,
        args.ngs_psfs,
        args.output_dir,
        build_ngs_psf=not args.skip_ngs_psf,
    )
