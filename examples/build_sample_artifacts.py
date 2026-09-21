"""Build the small interpolation products used by ``examples/hybrid.ini``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy import units as u

from hybrid_ao_psf import (
    NgsHoMetricSamples,
    RbfInterpolationConfig,
    ScienceHoPsfSamples,
    build_ngs_ho_metric_interpolator,
    build_science_ho_psf_interpolator,
    save_ngs_ho_metric_interpolator,
    save_science_ho_psf_interpolator,
)

ROOT = Path(__file__).resolve().parent
PRODUCTS = ROOT / "products"
SCIENCE_X = np.array([0.0, 36.0, 60.0])
NGS_THETA = np.deg2rad(np.array([0.0, 120.0, 240.0]))
PSF_SIZE = 17


def main() -> None:
    """Write ready-to-evaluate sample products beside the sample INI."""
    PRODUCTS.mkdir(parents=True, exist_ok=True)
    save_science_ho_psf_interpolator(
        build_science_ho_psf_interpolator(
            ScienceHoPsfSamples(
                zenith_angle=20.0 * u.deg,
                wavelength=1.654 * u.um,
                x=SCIENCE_X * u.arcsec,
                y=np.zeros(3) * u.arcsec,
                psfs=_base_psfs(),
                pixel_scale=10.796265491541053 * u.mas,
                tel_diameter=7.9 * u.m,
                tel_pupil=np.ones((PSF_SIZE, PSF_SIZE), dtype=np.float32) * u.one,
                provenance=("Hybrid AO PSF standalone sample",),
            )
        ),
        PRODUCTS / "science.h5",
        overwrite=True,
    )
    save_ngs_ho_metric_interpolator(
        build_ngs_ho_metric_interpolator(
            NgsHoMetricSamples(
                zenith_angle=20.0 * u.deg,
                x=30.0 * np.cos(NGS_THETA) * u.arcsec,
                y=30.0 * np.sin(NGS_THETA) * u.arcsec,
                ee=np.array([0.42, 0.40, 0.38]) * u.one,
                fwhm=np.array([76.0, 80.0, 84.0]) * u.mas,
                sr=np.array([0.24, 0.22, 0.20]) * u.one,
                provenance=("Hybrid AO PSF standalone sample",),
            ),
            interpolation_config=RbfInterpolationConfig(smoothing=0.0),
        ),
        PRODUCTS / "ngs.pkl",
        overwrite=True,
    )


def _base_psfs() -> np.ndarray:
    axis = np.arange(PSF_SIZE, dtype=float) - PSF_SIZE // 2
    x, y = np.meshgrid(axis, axis)
    psfs = []
    for sigma_x, sigma_y, angle_deg in (
        (1.4, 1.6, 0.0),
        (1.6, 1.9, 25.0),
        (1.8, 1.5, -35.0),
    ):
        angle = np.deg2rad(angle_deg)
        xr = np.cos(angle) * x + np.sin(angle) * y
        yr = -np.sin(angle) * x + np.cos(angle) * y
        psf = np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
        psfs.append(psf / np.sum(psf))
    return np.asarray(psfs, dtype=np.float32)


if __name__ == "__main__":
    main()
