"""Independent known-displacement gain and conditional-noise checks."""

import numpy as np
import pytest
from astropy import units as u

from hybrid_ao_psf import (
    CalibrationPhotometry,
    FixedGainIntegrator,
    SensingCalibrationMeasurements,
    SensingWfsMeasurements,
    SensingWfsSampling,
    estimate_sensing_statistics,
    load_sensing_measurements,
    save_sensing_measurements,
)


def samples():
    rng = np.random.default_rng(19)
    zero = rng.normal(size=(128, 2)) @ np.array([[2.0, 0.3], [0.1, 1.0]]) + [
        31.0,
        -23.0,
    ]
    matrix = np.array([[0.8, 0.05], [-0.03, 1.2]])
    common = rng.normal(size=(64, 2))
    shifted = np.stack(
        [
            common + 10 * matrix[:, 0],
            common - 10 * matrix[:, 0],
            common + 10 * matrix[:, 1],
            common - 10 * matrix[:, 1],
        ]
    )
    return SensingWfsMeasurements(
        zero=zero * u.mas, shifted=shifted * u.mas, dtype=np.float64
    ), matrix


def test_paired_gain_and_population_covariance_known_inputs():
    measured, expected = samples()
    stats = estimate_sensing_statistics(
        measured, displacement=10 * u.mas, block_draws=8, dtype=np.float64
    )
    np.testing.assert_allclose(
        stats.gain_matrix.value, expected, rtol=1e-14, atol=1e-14
    )
    assert stats.gain == pytest.approx(1.0)
    np.testing.assert_allclose(
        stats.mean.value, measured.zero.value.mean(0), rtol=1e-14
    )
    np.testing.assert_allclose(
        stats.covariance.value, np.cov(measured.zero.value.T, ddof=0), rtol=1e-14
    )
    assert stats.variance.to_value(u.mas**2) == pytest.approx(
        np.trace(np.cov(measured.zero.value.T, ddof=0)) / 2
    )
    white = stats.white.value
    np.testing.assert_allclose(white.mean(0), 0, atol=1e-8)
    np.testing.assert_allclose(
        white.T @ white / len(white), np.eye(2), rtol=1e-7, atol=1e-8
    )
    assert np.linalg.norm(measured.zero.value.mean(0)) > 30


def test_reusable_sensing_observations_preserve_raw_means_pairing_and_statistics(
    tmp_path,
):
    measured, _ = samples()
    point = SensingCalibrationMeasurements(
        zenith_angle=20 * u.deg,
        position=[12, -24] * u.arcsec,
        magnitude=16 * u.mag,
        samples=measured,
        displacement=10 * u.mas,
        sampling=SensingWfsSampling(
            zero_draws=128, gain_draws=64, block_draws=8, seed=77
        ),
        controller=FixedGainIntegrator(
            gain=0.3, delay=2, frame_rate=500 * u.Hz, dtype=np.float64
        ),
        photometry=CalibrationPhotometry(
            magnitude_band="fixture",
            photons_per_frame_zeropoint=1e8 * u.photon,
            dtype=np.float64,
        ),
        dtype=np.float64,
    )
    path = tmp_path / "measured.npz"
    save_sensing_measurements(point, path)
    restored = load_sensing_measurements(path)
    np.testing.assert_array_equal(restored.samples.zero, point.samples.zero)
    np.testing.assert_array_equal(restored.samples.shifted, point.samples.shifted)
    np.testing.assert_array_equal(
        restored.statistics().gain_matrix, point.statistics().gain_matrix
    )
    assert restored.sampling.seed == 77
    with pytest.raises(ValueError):
        restored.samples.zero.value.flags.writeable = True


def test_complete_sensing_offline_replay_shape_fit_and_roundtrip(tmp_path):
    from hybrid_ao_psf import (
        CovarianceCalibrationSettings,
        CovarianceTelemetry,
        ReconstructorNoiseModel,
        SensingCalibrationSettings,
        SensingReplayRequest,
        build_covariance_calibration,
        build_sensing_calibration,
        build_sensing_response,
        load_sensing_calibration,
        load_sensing_replay,
        measure_covariance_calibration,
        replay_sensing_calibration,
        save_sensing_calibration,
        save_sensing_replay,
    )

    controller = FixedGainIntegrator(
        gain=0.3, delay=2, frame_rate=500 * u.Hz, dtype=np.float64
    )
    photometry = CalibrationPhotometry(
        magnitude_band="fixture",
        photons_per_frame_zeropoint=1e8 * u.photon,
        dtype=np.float64,
    )
    sampling = SensingWfsSampling(zero_draws=128, gain_draws=64, block_draws=8, seed=77)
    measured, _ = samples()
    points = []
    for x in (-1, 1):
        for y in (-1, 1):
            for magnitude in (8, 10):
                points.append(
                    SensingCalibrationMeasurements(
                        zenith_angle=20 * u.deg,
                        position=[x, y] * u.arcsec,
                        magnitude=magnitude * u.mag,
                        samples=measured,
                        displacement=10 * u.mas,
                        sampling=sampling,
                        controller=controller,
                        photometry=photometry,
                        dtype=np.float64,
                    )
                )
    response = build_sensing_response(
        points,
        field_radius=2 * u.arcsec,
        magnitude_intervals=[[8, 10]] * u.mag,
        dtype=np.float64,
    )
    prediction = response.model.predict(
        positions=[[0, 0]] * u.arcsec, magnitudes=9 * u.mag, zenith_angle=20 * u.deg
    )
    assert prediction.gain.value[0] == pytest.approx(1.0)
    rng = np.random.default_rng(72)
    positions = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    shared = rng.normal(size=(3, 600, 2))
    tilt = shared[:, :, None, :] + rng.normal(size=(3, 600, 4, 2)) * 0.2
    telemetry = CovarianceTelemetry(
        zenith_angle=20 * u.deg,
        positions=positions * u.arcsec,
        frame_rate=500 * u.Hz,
        tilt=tilt * u.mas,
        coma=(tilt * 2) * u.nm,
        dtype=np.float64,
    )
    covariances = build_covariance_calibration(
        [
            measure_covariance_calibration(
                telemetry,
                controller,
                settings=CovarianceCalibrationSettings(
                    feedback=(0.5, 1, 1.5), bootstrap_draws=7
                ),
                dtype=np.float64,
            )
        ],
        dtype=np.float64,
    ).model
    prior = ReconstructorNoiseModel(
        photons_per_frame=[1, 1e6] * u.photon,
        fwhm=[50, 100] * u.mas,
        variance=np.ones((2, 2)) * 0.01 * u.mas**2,
        dtype=np.float64,
    )
    requests = [
        SensingReplayRequest(
            key=f"case{n}",
            population="fixture",
            zenith_angle=20 * u.deg,
            positions=positions[:n] * 0.5 * u.arcsec,
            magnitudes=np.full(n, 9) * u.mag,
            ee=np.full(n, 0.5) * u.one,
            fwhm=np.full(n, 80) * u.mas,
            seed=901 + n,
            dtype=np.float64,
        )
        for n in (1, 2, 3)
    ]
    settings = SensingCalibrationSettings(
        zero_draws=128,
        gain_draws=64,
        block_draws=8,
        replay_realizations=3,
        replay_frames=256,
        discard_frames=96,
    )
    replays = list(
        replay_sensing_calibration(
            response, requests, covariances, prior, settings=settings, dtype=np.float64
        )
    )
    for replay in replays:
        request = replay.request
        rng = np.random.default_rng(request.seed)
        total = settings.replay_frames + settings.discard_frames
        x = np.zeros((3, total, 2))
        blocks = replay.reconstructor.value.reshape(
            2, len(request.positions), 2
        ).transpose(1, 0, 2)
        for g in range(len(request.positions)):
            white = response.draw_noise(
                position=request.positions[g],
                magnitude=request.magnitudes[g],
                zenith_angle=request.zenith_angle,
                rng=rng,
                count=3 * total,
            )
            scaled = (white * np.sqrt(replay.variances.value[g])).reshape(3, total, 2)
            x += scaled @ blocks[g].T
        output = np.zeros_like(x)
        B = replay.feedback.value
        for k in range(2, total):
            output[:, k] = (
                output[:, k - 1] - 0.3 * (output[:, k - 2] @ B.T) + 0.3 * x[:, k - 2]
            )
        displacement = -output[:, settings.discard_frames :]
        pooled = displacement.mean(axis=(0, 1))
        centred = displacement - pooled
        np.testing.assert_allclose(
            replay.displacements.value,
            centred.astype(replay.dtype),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            replay.pooled_mean.value, pooled, rtol=1e-12, atol=1e-12
        )
        flat = centred.reshape(-1, 2)
        np.testing.assert_allclose(
            replay.measured_covariance.value,
            flat.T @ flat / len(flat),
            rtol=1e-12,
            atol=1e-12,
        )
        assert np.linalg.norm(centred.mean(axis=1)) > 0
        path = tmp_path / f"{request.key}.npz"
        save_sensing_replay(replay, path)
        np.testing.assert_array_equal(
            load_sensing_replay(path).displacements, replay.displacements
        )
    from dataclasses import replace

    with pytest.raises(ValueError, match="response calibration"):
        build_sensing_calibration(
            response,
            [
                replace(
                    replays[0],
                    variances=replays[0].variances * 1.1,
                    analytical_covariance=replays[0].analytical_covariance * 1.1,
                ),
                *replays[1:],
            ],
            dtype=np.float64,
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        replace(replays[0], measured_covariance=np.diag([-1.0, 1.0]) * u.mas**2)
    # Stored float32 histories need not have an exactly zero pooled mean, and
    # the distinct realization means are part of the calibrated distribution.
    assert np.linalg.norm(replays[0].displacements.value.mean(axis=(0, 1))) > 0
    changed = replays[0].displacements.copy()
    changed[0, 0] *= 100
    for updates in (
        {"displacements": changed},
        {"measured_covariance": replays[0].measured_covariance * 1.1},
        {"pooled_mean": replays[0].pooled_mean + [1, 0] * u.mas},
        {"realization_means": replays[0].realization_means + [1, 0] * u.mas},
    ):
        with pytest.raises(ValueError, match="history moments"):
            replace(replays[0], **updates)
    # The same validation must apply to external retained records on load.
    path = tmp_path / f"{replays[0].request.key}.npz"
    with np.load(path, allow_pickle=False) as stored:
        arrays = dict(stored)
    arrays["measured_covariance"] = arrays["measured_covariance"] * 1.2
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="history moments"):
        load_sensing_replay(path)
    result = build_sensing_calibration(response, replays, dtype=np.float64)
    assert result.diagnostics["asterisms"] == 3
    assert result.diagnostics["final_fit"]["family_weights"] == {
        "single": 0.5,
        "multi": 0.5,
    }
    path = tmp_path / "model.json"
    save_sensing_calibration(result.model, path)
    model = load_sensing_calibration(path)
    case = requests[2]
    width = model.noise_covariance(
        positions=case.positions,
        magnitudes=case.magnitudes,
        zenith_angle=case.zenith_angle,
        reconstructor=replays[2].reconstructor.value,
    )
    np.testing.assert_array_equal(width, replays[2].analytical_covariance)
    assert model.blur.parameters() == result.model.blur.parameters()
    with pytest.raises(ValueError):
        model.response.gain_table.value.flags.writeable = True


def test_empty_replay_asterism_is_rejected_before_loop_work():
    from hybrid_ao_psf import SensingReplayRequest

    with pytest.raises(ValueError, match="must not be empty"):
        SensingReplayRequest(
            key="empty",
            population="test",
            zenith_angle=20 * u.deg,
            positions=np.empty((0, 2)) * u.arcsec,
            magnitudes=np.empty(0) * u.mag,
            ee=np.empty(0) * u.one,
            fwhm=np.empty(0) * u.mas,
            seed=7,
            dtype=np.float64,
        )
