from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from ao_stats import PsfMetadata
from astropy import units as u

from hybrid_ao_psf import (
    DiagnosticsLevel,
    HybridRequest,
    NgsHoMetricPrediction,
    NonPsdPolicy,
    ScienceHoPsfPrediction,
    apply_finite_fov_otfs,
    blur_psfs_with_ctot,
    engine,
    image_plane_gaussian_otf,
)


@dataclass(frozen=True)
class _ScienceProvider:
    prediction: ScienceHoPsfPrediction

    def predict(self, **_: object) -> ScienceHoPsfPrediction:
        return self.prediction


@dataclass(frozen=True)
class _NgsProvider:
    prediction: NgsHoMetricPrediction

    def predict(self, **_: object) -> NgsHoMetricPrediction:
        return self.prediction


def _request(
    *,
    policy: NonPsdPolicy = NonPsdPolicy.ERROR,
    diagnostics: DiagnosticsLevel = DiagnosticsLevel.DEBUG,
) -> HybridRequest:
    return HybridRequest(
        science_x=np.array([0.0, 1.0]) * u.arcsec,
        science_y=np.array([0.0, 0.0]) * u.arcsec,
        ngs_x=np.array([10.0, -5.0]) * u.arcsec,
        ngs_y=np.array([0.0, 5.0 * np.sqrt(3.0)]) * u.arcsec,
        wavelength=1.65 * u.um,
        zenith_angle=30.0 * u.deg,
        ngs_flux=np.array([1000.0, 2000.0]) * u.photon / u.s,
        ngs_frame_rate=np.array([500.0, 500.0]) * u.Hz,
        mastsel_ini="""[telescope]
ZenithAngle = 0.0

[sources_science]
Wavelength = [1e-6]
Zenith = [0.0]
Azimuth = [0.0]

[sources_LO]
Wavelength = [710e-9]
Zenith = [1.0]
Azimuth = [0.0]

[sensor_LO]
NumberLenslets = [1]
""",
        non_psd_policy=policy,
        diagnostics_level=diagnostics,
    )


def _providers() -> tuple[_ScienceProvider, _NgsProvider]:
    psfs = np.zeros((2, 9, 9), dtype=np.float32)
    psfs[:, 4, 4] = 1.0
    metadata = PsfMetadata(
        wavelength=1.65 * u.um,
        pixel_scale=8.0 * u.mas,
        tel_diameter=8.0 * u.m,
        tel_pupil=np.ones((8, 8)) * u.one,
    )
    science = _ScienceProvider(
        ScienceHoPsfPrediction(
            psfs=psfs,
            pixel_scale=8.0 * u.mas,
            metadata=metadata,
            meta={"airmass": 1.2 * u.one},
        )
    )
    ngs = _NgsProvider(
        NgsHoMetricPrediction(
            ee=np.array([0.4, 0.5]) * u.one,
            fwhm=np.array([100.0, 110.0]) * u.mas,
            sr=np.array([0.1, 0.2]) * u.one,
        )
    )
    return science, ngs


def test_simulate_builds_owned_result_and_passes_resolved_mastsel_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeMavisLO:
        error = False
        mas2nm = 2.0

        def __init__(self, path: str, name: str, *, verbose: bool) -> None:
            calls["constructor"] = (path, name, verbose)

        def computeTotalResidualMatrix(self, *args: object, **kwargs: object) -> np.ndarray:
            calls["args"] = args
            calls["kwargs"] = kwargs
            return np.stack([400.0 * np.eye(2), 800.0 * np.eye(2)])

    monkeypatch.setattr(engine, "_load_mavis_lo", lambda: FakeMavisLO)
    science, ngs = _providers()
    result = engine.simulate(_request(), science, ngs)

    args = calls["args"]
    np.testing.assert_allclose(args[0], [[0.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(args[1], [[10.0, 0.0], [-5.0, 5.0 * np.sqrt(3.0)]])
    np.testing.assert_allclose(args[2], [1000.0, 2000.0])
    np.testing.assert_allclose(args[3], [500.0, 500.0])
    assert calls["kwargs"] == {"aNGS_FWHM_DL_mas": None, "doAll": True}
    assert result.psfs.dtype == np.float32
    assert result.psfs.shape == (2, 9, 9)
    assert np.all(np.sum(result.psfs, axis=(-2, -1)) < 1.0)
    expected_psfs = science.prediction.psfs.copy()
    engine.apply_ctot_blur(
        expected_psfs,
        np.stack([400.0 * np.eye(2), 800.0 * np.eye(2)]) * u.nm**2,
        pixel_scale=science.prediction.pixel_scale,
        angle_to_wavefront_scale=2.0 * u.nm / u.mas,
    )
    np.testing.assert_allclose(result.psfs, expected_psfs)
    assert result.metadata is not science.prediction.metadata
    assert result.metadata.pixel_scale == 8.0 * u.mas
    np.testing.assert_allclose(result.jitter.to_value(u.mas), [np.sqrt(200.0), 20.0])
    assert result.diagnostics is not None
    np.testing.assert_array_equal(result.diagnostics.pre_blur_psfs, science.prediction.psfs)
    np.testing.assert_array_equal(result.diagnostics.psd_valid_mask, [True, True])
    assert result.diagnostics.psd_valid_count == 2
    assert result.diagnostics.ctot_angle_trace_min == 200.0 * u.mas**2
    assert result.diagnostics.ctot_angle_trace_max == 400.0 * u.mas**2
    assert result.diagnostics.ctot_angle_determinant_min.to_value(u.mas**4) == (
        pytest.approx(10000.0)
    )
    assert result.diagnostics.ctot_angle_determinant_max.to_value(u.mas**4) == (
        pytest.approx(40000.0)
    )
    assert not np.any(result.psd_clipped)
    assert "zenithangle = 30" in result.effective_mastsel_ini.lower()
    assert "zenith = [0, 1]" in result.effective_mastsel_ini.lower()


def test_validation_diagnostics_omit_pre_blur_psfs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMavisLO:
        error = False
        mas2nm = 2.0

        def __init__(self, *_: object, **__: object) -> None:
            pass

        def computeTotalResidualMatrix(self, *_: object, **__: object) -> np.ndarray:
            return np.stack([400.0 * np.eye(2), 800.0 * np.eye(2)])

    monkeypatch.setattr(engine, "_load_mavis_lo", lambda: FakeMavisLO)
    science, ngs = _providers()

    result = engine.simulate(
        _request(diagnostics=DiagnosticsLevel.VALIDATION), science, ngs
    )

    assert result.diagnostics is not None
    assert result.diagnostics.pre_blur_psfs is None
    np.testing.assert_array_equal(result.diagnostics.psd_valid_mask, [True, True])


def test_none_diagnostics_omits_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMavisLO:
        error = False
        mas2nm = 2.0

        def __init__(self, *_: object, **__: object) -> None:
            pass

        def computeTotalResidualMatrix(self, *_: object, **__: object) -> np.ndarray:
            return np.stack([400.0 * np.eye(2), 800.0 * np.eye(2)])

    monkeypatch.setattr(engine, "_load_mavis_lo", lambda: FakeMavisLO)
    science, ngs = _providers()

    result = engine.simulate(
        _request(diagnostics=DiagnosticsLevel.NONE), science, ngs
    )

    assert result.diagnostics is None


def test_rotated_roundoff_ctot_remains_active_after_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    angle = 0.01
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    raw_ctot = rotation @ np.diag([-0.5e-9, 20000.0]) @ rotation.T

    class FakeMavisLO:
        error = False
        mas2nm = 2.0

        def __init__(self, *_: object, **__: object) -> None:
            pass

        def computeTotalResidualMatrix(self, *_: object, **__: object) -> np.ndarray:
            return np.stack([raw_ctot, raw_ctot])

    monkeypatch.setattr(engine, "_load_mavis_lo", lambda: FakeMavisLO)
    science, ngs = _providers()

    result = engine.simulate(
        _request(diagnostics=DiagnosticsLevel.VALIDATION), science, ngs
    )

    np.testing.assert_array_equal(result.psd_clipped, [True, True])
    assert result.diagnostics is not None
    np.testing.assert_array_equal(result.diagnostics.psd_valid_mask, [True, True])
    assert not np.array_equal(result.psfs, science.prediction.psfs)


def test_condition_ctot_distinguishes_numerical_and_material_negative_eigenvalues() -> None:
    numerical = np.array([[[-0.5e-9, 0.0], [0.0, 2.0]]]) * u.nm**2
    conditioned, clipped = engine.condition_ctot(numerical)
    np.testing.assert_allclose(conditioned.to_value(u.nm**2), [[[0.0, 0.0], [0.0, 2.0]]])
    np.testing.assert_array_equal(clipped, [True])

    material = np.array([[[-1.0, 0.0], [0.0, 2.0]]]) * u.nm**2
    with pytest.raises(ValueError, match="materially non-positive-semidefinite"):
        engine.condition_ctot(material, policy="error")
    conditioned, clipped = engine.condition_ctot(material, policy="clip")
    np.testing.assert_allclose(conditioned.to_value(u.nm**2), [[[0.0, 0.0], [0.0, 2.0]]])
    np.testing.assert_array_equal(clipped, [True])


def test_ctot_blur_preserves_zero_and_allows_finite_field_spill() -> None:
    psfs = np.zeros((2, 9, 9), dtype=np.float32)
    psfs[:, 0, 0] = np.array([2.0, 5.0], dtype=np.float32)
    original = psfs.copy()
    engine.apply_ctot_blur(
        psfs,
        np.zeros((2, 2, 2)) * u.nm**2,
        pixel_scale=4.0 * u.mas,
        angle_to_wavefront_scale=2.0 * u.nm / u.mas,
    )
    np.testing.assert_allclose(psfs, original)

    engine.apply_ctot_blur(
        psfs,
        np.stack([400.0 * np.eye(2), 800.0 * np.eye(2)]) * u.nm**2,
        pixel_scale=4.0 * u.mas,
        angle_to_wavefront_scale=2.0 * u.nm / u.mas,
    )
    flux = np.sum(psfs, axis=(-2, -1))
    assert np.all(flux > 0.0)
    assert np.all(flux < np.array([2.0, 5.0]))


def test_public_blur_matches_engine_without_mutating_external_psfs() -> None:
    psfs = np.zeros((2, 9, 9), dtype=np.float32)
    psfs[:, 0, 0] = [2.0, 5.0]
    original = psfs.copy()
    ctot = np.stack([400.0 * np.eye(2), np.zeros((2, 2))]) * u.nm**2
    expected = psfs.copy()
    engine.apply_ctot_blur(
        expected,
        ctot,
        pixel_scale=4.0 * u.mas,
        angle_to_wavefront_scale=2.0 * u.nm / u.mas,
    )

    actual = blur_psfs_with_ctot(
        psfs,
        ctot,
        pixel_scale=4.0 * u.mas,
        angle_to_wavefront_scale=2.0 * u.nm / u.mas,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(psfs, original)
    assert actual.dtype == psfs.dtype


def test_public_blur_rejects_unconditioned_ctot() -> None:
    psfs = np.ones((1, 9, 9), dtype=np.float32)
    with pytest.raises(ValueError, match="already be positive semidefinite"):
        blur_psfs_with_ctot(
            psfs,
            np.array([[[-1.0, 0.0], [0.0, 2.0]]]) * u.nm**2,
            pixel_scale=4.0 * u.mas,
            angle_to_wavefront_scale=2.0 * u.nm / u.mas,
        )
    with pytest.raises(ValueError, match="symmetric"):
        blur_psfs_with_ctot(
            psfs,
            np.array([[[2.0, 1.0], [0.0, 2.0]]]) * u.nm**2,
            pixel_scale=4.0 * u.mas,
            angle_to_wavefront_scale=2.0 * u.nm / u.mas,
        )


def test_finite_field_otf_applies_mean_shift_without_renormalizing() -> None:
    psfs = np.zeros((1, 9, 9), dtype=np.float32)
    psfs[0, 4, 4] = 2.0
    otf = image_plane_gaussian_otf(
        np.zeros((2, 2)),
        (18, 18),
        mean_shift_pix=np.array([2.0, -1.0]),
    )
    apply_finite_fov_otfs(psfs, otf[None])

    expected = np.zeros_like(psfs)
    expected[0, 3, 6] = 2.0
    np.testing.assert_allclose(psfs, expected, rtol=0.0, atol=2.0e-7)
