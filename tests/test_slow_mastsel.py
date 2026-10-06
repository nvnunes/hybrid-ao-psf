from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from ngs_photometry import WFSPhotometryConfig, photons_per_frame_to_magnitudes

from hybrid_ao_psf import (
    HybridCtotCorrection,
    HybridRequest,
    NgsHoMetricPrediction,
    NgsHoMetricSamples,
    RbfInterpolationConfig,
    ScienceHoPsfSamples,
    _mastsel,
    build_ngs_ho_metric_interpolator,
    build_science_ho_psf_interpolator,
    cli,
    load_configuration,
    load_ngs_ho_metric_interpolator,
    load_result,
    load_science_ho_psf_interpolator,
    run_configuration,
    save_ngs_ho_metric_interpolator,
    save_science_ho_psf_interpolator,
    simulate,
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

    # The same MASTSEL-only INI and loaded providers also work through the
    # public Python magnitude path, independently of AO Predict.
    configuration = load_configuration(configuration_path)
    zeropoint = 1.0e10 * u.photon / (u.m**2 * u.s)
    photometry = WFSPhotometryConfig(
        telescope_diameter=TELESCOPE_DIAMETER_M * u.m,
        n_channels=1.0,
        frame_rate=500.0 * u.Hz,
        zeropoint=zeropoint,
    )
    magnitudes = photons_per_frame_to_magnitudes(
        configuration.request.ngs_flux / (500.0 * u.Hz), photometry
    )
    magnitude_result = simulate(
        replace(
            configuration.request,
            ngs_flux=None,
            ngs_magnitude=magnitudes,
            ngs_magnitude_zeropoint=zeropoint,
        ),
        load_science_ho_psf_interpolator(science_path),
        load_ngs_ho_metric_interpolator(ngs_path),
    )
    np.testing.assert_allclose(
        magnitude_result.ngs_flux.value, result.ngs_flux.value, rtol=1e-14
    )
    np.testing.assert_allclose(
        magnitude_result.ctot_wavefront.value, result.ctot_wavefront.value, rtol=1e-12
    )
    np.testing.assert_allclose(magnitude_result.psfs, result.psfs, rtol=1e-6, atol=1e-7)

    observed = []

    def noop(ctot, context):
        observed.append(context.mastsel_statistics)

    no_op = simulate(
        configuration.request,
        load_science_ho_psf_interpolator(science_path),
        load_ngs_ho_metric_interpolator(ngs_path),
        ctot_correction=HybridCtotCorrection(noop),
    )
    np.testing.assert_array_equal(no_op.ctot_wavefront, result.ctot_wavefront)
    np.testing.assert_array_equal(no_op.psfs, result.psfs)
    assert len(observed) == 1
    assert observed[0].measurement_noise is not None


@pytest.mark.slow_mastsel
@pytest.mark.parametrize("count", [1, 2, 3])
def test_capture_noise_matches_native_fixed_gain_expression(tmp_path, count):
    """Compare recovered noise with native zero-atmosphere expressions per NGS."""
    ngs_x, ngs_y = _ngs_cartesian()
    rates = np.array([500, 450, 400])[:count] * u.Hz
    flux = np.array([25000, 16000, 10000])[:count] * u.photon / u.s
    request = HybridRequest(
        science_x=np.array([0, 30, 0]) * u.arcsec,
        science_y=np.array([0, 0, 30]) * u.arcsec,
        ngs_x=ngs_x[:count] * u.arcsec,
        ngs_y=ngs_y[:count] * u.arcsec,
        ngs_flux=flux,
        wavelength=WAVELENGTH_UM * u.um,
        zenith_angle=20 * u.deg,
        mastsel_ini=EXAMPLE.read_text(),
    )
    metrics = NgsHoMetricPrediction(
        np.array([0.42, 0.40, 0.38])[:count] * u.one,
        np.array([76, 80, 84])[:count] * u.mas,
        np.full(count, 0.24) * u.one,
    )
    context = _mastsel._prepare_mastsel_context(
        str(tmp_path), request, metrics, request.mastsel_ini, rates, flux
    )
    model = context.mastsel
    native = model.computeNoiseResidual
    expected = []

    def record_native(fmin, fmax, samples, variance, bias):
        result = native(fmin, fmax, samples, variance, bias)
        f = np.linspace(fmin, fmax, samples)
        gain = bias * model.LoopGain_LO
        expected.extend(
            (
                float(abs(np.sum(model.fTipS_lambda1(gain, f, np.zeros(samples))))),
                float(abs(np.sum(model.fTiltS_lambda1(gain, f, np.zeros(samples))))),
            )
        )
        return result

    model.computeNoiseResidual = record_native
    baseline = _mastsel._compute_mastsel_ctot(context)
    expected.clear()
    with _mastsel._capture_mastsel(context) as captured:
        replay = _mastsel._compute_mastsel_ctot(context)
    completed = _mastsel._complete_mastsel_context(context, captured)
    np.testing.assert_array_equal(replay, baseline)
    np.testing.assert_allclose(
        np.diag(completed.mastsel_statistics.measurement_noise.value),
        expected,
        rtol=2e-13,
        atol=1e-13,
    )
    assert completed.mastsel is model
    assert "computeCovMatrices" not in model.__dict__
    assert "buildReconstuctor2" not in model.__dict__


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
