"""Precision declarations, original support endpoints and legacy preservation."""

import json

import numpy as np
import pytest
from astropy import units as u

from hybrid_ao_psf import (
    NgsHoMetricSamples,
    ReconstructorNoiseMeasurements,
    ReconstructorNoiseModel,
    build_ngs_ho_metric_interpolator,
    load_calibration_samples,
    save_calibration_samples,
)


def test_promoted_prior_keeps_original_lower_endpoint():
    source = ReconstructorNoiseModel(
        photons_per_frame=[0.1, 2.0] * u.photon,
        fwhm=[10.1, 20.2] * u.mas,
        variance=np.ones((2, 2)) * u.mas**2,
        dtype=np.float32,
        producer_dtype=np.float32,
    )
    for model in (source, source.prepare(dtype=np.float64)):
        result = model.predict(
            photons_per_frame=[0.1] * u.photon, ee=[1.0] * u.one, fwhm=[10.1] * u.mas
        )
        assert result.dtype == model.dtype
        with pytest.raises(ValueError, match="support"):
            model.predict(
                photons_per_frame=[0.09] * u.photon,
                ee=[1.0] * u.one,
                fwhm=[10.1] * u.mas,
            )


@pytest.mark.parametrize("angles", [[20.2], [20.2, 40.3], [0.0, 4.028]])
def test_ngs_original_zenith_queries_survive_preparation(angles):
    shape = (len(angles), 4)
    source = build_ngs_ho_metric_interpolator(
        NgsHoMetricSamples(
            x=[0.1, 1.1, 0.1, 1.1] * u.arcsec,
            y=[0.2, 0.2, 1.2, 1.2] * u.arcsec,
            zenith_angle=angles * u.deg,
            ee=np.full(shape, 0.4) * u.one,
            fwhm=np.full(shape, 100.0) * u.mas,
            sr=np.full(shape, 0.2) * u.one,
        ),
        dtype=np.float32,
    )
    for model in (source, source.prepare(dtype=np.float64)):
        result = model.predict(
            zenith_angle=angles[-1] * u.deg, x=[0.1] * u.arcsec, y=[0.2] * u.arcsec
        )
        assert result.fwhm.dtype == model.dtype
        with pytest.raises(ValueError, match="support|fixed"):
            model.predict(
                zenith_angle=(angles[-1] + 1.0) * u.deg,
                x=[0.1] * u.arcsec,
                y=[0.2] * u.arcsec,
            )


@pytest.mark.parametrize("change", ["missing", "mismatch", "integer"])
def test_v2_observation_precision_cannot_silently_round(tmp_path, change):
    path = tmp_path / "record.npz"
    save_calibration_samples(
        ReconstructorNoiseMeasurements(
            covariance=np.array([np.eye(2) * (1.0 + 1e-9)]) * u.mas**2, dtype=np.float64
        ),
        path,
    )
    with np.load(path) as saved:
        arrays = {key: saved[key] for key in saved.files}
    header = json.loads(str(arrays["__header__"]))
    if change == "missing":
        del header["precision"]
    elif change == "mismatch":
        header["precision"] = "float32"
    else:
        arrays["covariance"] = np.full((1, 2, 2), 16777217, dtype=np.int64)
    arrays["__header__"] = json.dumps(header)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="precision|schema"):
        load_calibration_samples(path)


@pytest.mark.parametrize("source_dtype", [np.float32, np.float64])
def test_promoted_sensing_prediction_and_noise_draw_keep_source_nodes(source_dtype):
    from hybrid_ao_psf import (
        CalibrationPhotometry,
        FixedGainIntegrator,
        SensingCalibrationMeasurements,
        SensingWfsMeasurements,
        SensingWfsSampling,
        build_sensing_response,
    )

    rng = np.random.default_rng(91)
    zero = rng.normal(size=(64, 2))
    shifted = np.stack(
        [zero[:16] + offset for offset in ([8, 0], [-8, 0], [0, 8], [0, -8])]
    )
    rows = [
        SensingCalibrationMeasurements(
            dtype=source_dtype,
            zenith_angle=20.2 * u.deg,
            position=[x, y] * u.arcsec,
            magnitude=m * u.mag,
            samples=SensingWfsMeasurements(
                zero=zero * u.mas, shifted=shifted * u.mas, dtype=source_dtype
            ),
            displacement=10 * u.mas,
            sampling=SensingWfsSampling(
                zero_draws=64, gain_draws=16, block_draws=4, seed=11
            ),
            controller=FixedGainIntegrator(gain=0.3, delay=2, frame_rate=500 * u.Hz),
            photometry=CalibrationPhotometry(
                magnitude_band="R", photons_per_frame_zeropoint=1e8 * u.photon
            ),
        )
        for x in (0.1, 1.1)
        for y in (0.2, 1.2)
        for m in (8.1, 10.1)
    ]
    source = build_sensing_response(
        rows, field_radius=10 * u.arcsec, dtype=source_dtype
    )
    for response in (
        source,
        source.prepare(dtype=np.float64),
        source.prepare(dtype=np.float32),
    ):
        prediction = response.model.predict(
            positions=[[0.1, 0.2]] * u.arcsec,
            magnitudes=8.1 * u.mag,
            zenith_angle=20.2 * u.deg,
        )
        assert prediction.gain.dtype == response.model.dtype
        draws = response.draw_noise(
            position=[0.1, 0.2] * u.arcsec,
            magnitude=8.1 * u.mag,
            zenith_angle=20.2 * u.deg,
            count=10,
            rng=np.random.default_rng(1),
        )
        assert draws.dtype == response.model.dtype and draws.shape == (10, 2)
        with pytest.raises(ValueError, match="measured zenith|support"):
            response.draw_noise(
                position=[0.1, 0.2] * u.arcsec,
                magnitude=8.1 * u.mag,
                zenith_angle=20.3 * u.deg,
                count=10,
                rng=np.random.default_rng(1),
            )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_real_calibration_boundaries_reject_complex(dtype):
    from hybrid_ao_psf import FixedGainIntegrator, SensingWfsMeasurements

    with pytest.raises(ValueError, match="real"):
        SensingWfsMeasurements(
            zero=np.ones((8, 2), dtype=complex) * (1 + 3j) * u.mas,
            shifted=np.ones((4, 8, 2), dtype=complex) * u.mas,
            dtype=dtype,
        )
    controller = FixedGainIntegrator(
        gain=0.3, delay=2, frame_rate=500 * u.Hz, dtype=dtype
    )
    with pytest.raises(ValueError, match="real"):
        controller.filter(np.ones((1, 10, 2), dtype=complex) * u.mas, np.eye(2))
    with pytest.raises(ValueError, match="real"):
        controller.filter(np.ones((1, 10, 2)) * u.mas, np.eye(2, dtype=complex))


def test_support_rounding_survives_preparation_and_product_roundtrip(tmp_path):
    from hybrid_ao_psf import (
        load_reconstructor_calibration,
        save_reconstructor_calibration,
    )

    source = ReconstructorNoiseModel(
        photons_per_frame=[0.1, 2.0] * u.photon,
        fwhm=[10.1, 20.2] * u.mas,
        variance=np.ones((2, 2)) * u.mas**2,
        dtype=np.float64,
        producer_dtype=np.float64,
    )
    reduced = source.prepare(dtype=np.float32)
    path = tmp_path / "model.json"
    save_reconstructor_calibration(reduced, path)
    for model in (
        reduced.prepare(dtype=np.float64),
        load_reconstructor_calibration(path).prepare(dtype=np.float64),
    ):
        assert model.producer_dtype == np.dtype("float64")
        assert model.support_precision == np.dtype("float32")
        result = model.predict(
            photons_per_frame=0.1 * u.photon, ee=1 * u.one, fwhm=10.1 * u.mas
        )
        assert result.dtype == np.float64


def test_double_metric_build_keeps_double_spatial_support():
    source = build_ngs_ho_metric_interpolator(
        NgsHoMetricSamples(
            x=[0.0, 1.0, 0.0, 1.0] * u.arcsec,
            y=[0.0, 0.0, 1.0, 1.0] * u.arcsec,
            zenith_angle=20.0 * u.deg,
            ee=np.full(4, 0.4) * u.one,
            fwhm=np.full(4, 100.0) * u.mas,
            sr=np.full(4, 0.2) * u.one,
        ),
        dtype=np.float64,
    )
    assert source.builder["support_precision"] == "float64"
    with pytest.raises(ValueError, match="support|field"):
        source.predict(
            zenith_angle=20.0 * u.deg, x=[-1e-7] * u.arcsec, y=[0.0] * u.arcsec
        )


@pytest.mark.parametrize("integer_axes", [False, True])
def test_science_build_retains_source_physical_axis_precision(integer_axes):
    from dataclasses import replace

    from hybrid_ao_psf import ScienceHoPsfSamples, build_science_ho_psf_interpolator

    samples = ScienceHoPsfSamples(
        x=[0, 2, 0, 2] * u.arcsec,
        y=[0, 0, 2, 2] * u.arcsec,
        psfs=np.ones((4, 5, 5)) / 25,
        wavelength=710 * u.nm,
        zenith_angle=20 * u.deg,
        pixel_scale=5 * u.mas,
        tel_diameter=8 * u.m,
        tel_pupil=np.ones((4, 4)) * u.one,
    )
    changes = {
        "zenith_angle": u.Quantity(20.2, u.deg, dtype=np.float32),
        "wavelength": u.Quantity(710.2, u.nm, dtype=np.float32),
    }
    if integer_axes:
        changes.update(
            x=u.Quantity([0, 2, 0, 2], u.arcsec, dtype=int),
            y=u.Quantity([0, 0, 2, 2], u.arcsec, dtype=int),
        )
    source = build_science_ho_psf_interpolator(
        replace(samples, **changes), dtype=np.float64
    )
    assert source.builder["support_precision"] == "float32"
    result = source.predict(
        zenith_angle=20.2 * u.deg,
        wavelength=710.2 * u.nm,
        x=[0.0] * u.arcsec,
        y=[0.0] * u.arcsec,
    )
    assert result.psfs.dtype == np.float64


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_noise_prior_wide_count_axis_does_not_snap_interior_queries(dtype):
    """A large upper support limit must not change faint-star photon counts."""
    model = ReconstructorNoiseModel(
        photons_per_frame=[1, 10, 1e8] * u.photon,
        fwhm=[10, 20] * u.mas,
        variance=np.array([[100, 400], [10, 40], [1e-6, 4e-6]]) * u.mas**2,
        dtype=dtype,
    )
    value = model.predict(
        photons_per_frame=[3] * u.photon, ee=[1] * u.one, fwhm=[10] * u.mas
    )
    np.testing.assert_allclose(
        value.value[0], np.eye(2) * (100 / 3), rtol=2e-6, atol=1e-10
    )
    with pytest.raises(ValueError, match="support"):
        model.predict(
            photons_per_frame=[1 - 1e-5] * u.photon, ee=[1] * u.one, fwhm=[10] * u.mas
        )
