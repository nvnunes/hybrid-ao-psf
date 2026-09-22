from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from hybrid_ao_psf import (
    NgsHoMetricSamples,
    RbfInterpolationConfig,
    ScienceHoPsfSamples,
    build_ngs_ho_metric_interpolator,
    build_science_ho_psf_interpolator,
    cli,
    load_result,
    run_configuration,
    save_ngs_ho_metric_interpolator,
    save_science_ho_psf_interpolator,
)

DATA = Path(__file__).parent / "data"
EXAMPLE = Path(__file__).parents[1] / "examples" / "hybrid.ini"
SCIENCE_X_ARCSEC = np.array([0.0, 36.0, 60.0])
SCIENCE_Y_ARCSEC = np.zeros(3)
NGS_RADIUS_ARCSEC = 30.0
NGS_THETA_DEG = np.array([0.0, 120.0, 240.0])
WAVELENGTH_UM = 1.654
PIXEL_SCALE_MAS = 10.796265491541053
TELESCOPE_DIAMETER_M = 7.9
PSF_SIZE = 17


@pytest.mark.slow_mastsel
def test_real_mastsel_engine_matches_retained_hybrid_golden(tmp_path: Path) -> None:
    pytest.importorskip("mastsel")
    products = tmp_path / "products"
    products.mkdir()
    science_path = products / "science.pkl"
    ngs_path = products / "ngs.pkl"
    save_science_ho_psf_interpolator(
        build_science_ho_psf_interpolator(
            ScienceHoPsfSamples(
                zenith_angle=20.0 * u.deg,
                wavelength=WAVELENGTH_UM * u.um,
                x=SCIENCE_X_ARCSEC * u.arcsec,
                y=SCIENCE_Y_ARCSEC * u.arcsec,
                psfs=_base_psfs(),
                pixel_scale=PIXEL_SCALE_MAS * u.mas,
                tel_diameter=TELESCOPE_DIAMETER_M * u.m,
                tel_pupil=np.ones((PSF_SIZE, PSF_SIZE), dtype=np.float32) * u.one,
            )
        ),
        science_path,
    )
    ngs_x, ngs_y = _ngs_cartesian()
    save_ngs_ho_metric_interpolator(
        build_ngs_ho_metric_interpolator(
            NgsHoMetricSamples(
                zenith_angle=20.0 * u.deg,
                x=ngs_x * u.arcsec,
                y=ngs_y * u.arcsec,
                ee=np.array([0.42, 0.40, 0.38]) * u.one,
                fwhm=np.array([76.0, 80.0, 84.0]) * u.mas,
                sr=np.array([0.24, 0.22, 0.20]) * u.one,
            ),
            interpolation_config=RbfInterpolationConfig(smoothing=0.0),
        ),
        ngs_path,
    )
    configuration_path = tmp_path / "hybrid.ini"
    shutil.copyfile(EXAMPLE, configuration_path)
    result = run_configuration(configuration_path)
    persisted = load_result(tmp_path / "results" / "hybrid.h5")
    cli_output = tmp_path / "results" / "cli.h5"
    assert cli.main([str(configuration_path), "--output", str(cli_output)]) == 0
    cli_result = load_result(cli_output)
    golden = json.loads(
        (DATA / "mastsel_reference_golden.json").read_text(encoding="utf-8")
    )

    assert result.psfs.shape == (3, PSF_SIZE, PSF_SIZE)
    assert result.psfs.dtype == np.float32
    assert result.diagnostics is not None
    assert not np.any(result.psd_clipped)
    assert result.provenance["configuration"] == str(configuration_path)
    np.testing.assert_array_equal(persisted.psfs, result.psfs)
    np.testing.assert_array_equal(cli_result.psfs, result.psfs)
    np.testing.assert_allclose(
        result.ngs_metrics.ee.to_value(u.one), [0.42, 0.40, 0.38], rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        result.ctot_angle.to_value(u.mas**2), golden["ctot_mas2"], rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        result.jitter.to_value(u.mas), golden["jitter"], rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        np.sum(result.psfs, axis=(1, 2)), golden["psf_flux"], rtol=1e-5, atol=1e-7
    )
    center = PSF_SIZE // 2
    np.testing.assert_allclose(
        result.psfs[:, center - 3 : center + 4, center - 3 : center + 4],
        golden["psf_core"],
        rtol=1e-5,
        atol=1e-7,
    )


def _ngs_cartesian() -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(NGS_THETA_DEG)
    return NGS_RADIUS_ARCSEC * np.cos(theta), NGS_RADIUS_ARCSEC * np.sin(theta)


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
