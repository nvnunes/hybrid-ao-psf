from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from ao_stats import PsfMetadata
from astropy import units as u
from ngs_photometry import (
    WFSPhotometryConfig,
    magnitudes_to_effective_photons_per_second,
)

from hybrid_ao_psf import (
    DiagnosticsLevel,
    HybridCtotCorrection,
    HybridRequest,
    NgsHoMetricPrediction,
    NonPsdPolicy,
    ScienceHoPsfPrediction,
    _mastsel,
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
    frame_rate: str = "500.0",
) -> HybridRequest:
    return HybridRequest(
        science_x=np.array([0.0, 1.0]) * u.arcsec,
        science_y=np.array([0.0, 0.0]) * u.arcsec,
        ngs_x=np.array([10.0, -5.0]) * u.arcsec,
        ngs_y=np.array([0.0, 5.0 * np.sqrt(3.0)]) * u.arcsec,
        ngs_flux=np.array([1000.0, 2000.0]) * u.photon / u.s,
        wavelength=1.65 * u.um,
        zenith_angle=30.0 * u.deg,
        mastsel_ini=f"""[telescope]
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

[RTC]
SensorFrameRate_LO = {frame_rate}
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


@pytest.fixture
def hook_runtime(monkeypatch):
    """A deterministic runtime exposing the same baseline method boundaries."""
    instances = []

    class Runtime:
        error = False
        mas2nm = 2.0
        LoopGain_LO = 0.3
        loopDelaySteps_LO = 2
        noNoise = False
        NumberLenslets = (1, 1)
        PixelScale_LO = (270.0, 280.0)
        N_sa_tot_LO = (1, 1)

        def __init__(self, directory, name, **kwargs):
            self.path = Path(directory) / f"{name}.ini"
            self.calls = {"total": 0, "covariance": 0, "reconstructor": 0}
            self.bias = [0.7, 0.8]
            self.avar = [(0.01, 0.02), (0.02, 0.03)]
            instances.append(self)

        def computeCovMatrices(self, science, ngs):
            self.calls["covariance"] += 1
            cas = np.tile(np.arange(2 * len(ngs)), (2 * len(science), 1)) + 10
            return np.eye(2), cas, np.eye(2 * len(ngs)) * 100

        def buildReconstuctor2(self, science, ngs, **kwargs):
            self.calls["reconstructor"] += 1
            r = (
                np.arange(4 * len(science) * len(ngs)).reshape(
                    2 * len(science), 2 * len(ngs)
                )
                / 100
            )
            return r, r.T

        def computeTotalResidualMatrix(self, science, ngs, *args, **kwargs):
            self.calls["total"] += 1
            assert self.path.exists()
            self.computeCovMatrices(science, ngs)
            self.buildReconstuctor2(science, ngs)
            return np.array([[[100, 20], [20, 400]], [[200, -30], [-30, 500]]])

    monkeypatch.setattr(_mastsel, "_load_mavis_lo", lambda: Runtime)
    return Runtime, instances


def test_noop_preserves_complete_result_and_runtime_lifetime(hook_runtime):
    _, instances = hook_runtime
    science, ngs = _providers()
    baseline = engine.simulate(_request(), science, ngs)
    calls = []

    def noop(ctot, context):
        calls.append(context)
        assert context.mastsel is instances[-1]
        assert context.mastsel.path.read_text() == context.effective_mastsel_ini
        assert not ctot.flags.writeable
        statistics = context.mastsel_statistics
        np.testing.assert_array_equal(
            statistics.reconstructor, np.arange(16).reshape(2, 2, 4) / 100
        )
        np.testing.assert_array_equal(
            statistics.centre_ngs_covariance, statistics.science_ngs_covariance[0]
        )
        assert statistics.measurement_noise.shape == (4, 4)
        assert "computeCovMatrices" not in context.mastsel.__dict__

    actual = engine.simulate(
        _request(),
        science,
        ngs,
        ctot_correction=HybridCtotCorrection(noop, {"fit": "fit.json"}),
    )
    assert len(calls) == 1 and len(instances) == 2
    for name in ("psfs", "ctot_wavefront", "ctot_angle", "jitter", "psd_clipped"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(baseline, name))
    assert actual.provenance["fit"] == "fit.json"
    assert instances[-1].calls == {"total": 1, "covariance": 1, "reconstructor": 1}
    assert not instances[-1].path.exists()


def test_doubling_uses_one_effective_covariance_for_blur_and_diagnostics(hook_runtime):
    _, instances = hook_runtime
    science, ngs = _providers()
    baseline = engine.simulate(replace(_request(), dtype=np.float64), science, ngs)
    actual = engine.simulate(
        replace(_request(), dtype=np.float64),
        science,
        ngs,
        ctot_correction=HybridCtotCorrection(
            lambda ctot, context: (2 * ctot).to(u.um**2)
        ),
    )
    np.testing.assert_allclose(actual.ctot_wavefront, 2 * baseline.ctot_wavefront)
    np.testing.assert_allclose(actual.jitter, np.sqrt(2) * baseline.jitter)
    np.testing.assert_allclose(
        actual.ctot_angle.to_value(u.mas**2).trace(axis1=1, axis2=2), [250, 350]
    )
    expected = blur_psfs_with_ctot(
        science.prediction.psfs,
        actual.ctot_wavefront,
        pixel_scale=science.prediction.pixel_scale,
        angle_to_wavefront_scale=baseline.angle_to_wavefront_scale,
        dtype=np.float64,
    )
    np.testing.assert_array_equal(actual.psfs, expected)
    assert actual.diagnostics.ctot_angle_trace_mean == 300 * u.mas**2
    assert instances[-1].calls["total"] == 1


@pytest.mark.parametrize(
    "replacement, error",
    [
        (np.eye(2), TypeError),
        (np.zeros((1, 2, 2)) * u.nm**2, ValueError),
        (np.zeros((2, 2, 2)) * u.mas**2, u.UnitConversionError),
        (np.full((2, 2, 2), np.nan) * u.nm**2, ValueError),
    ],
)
def test_invalid_correction_rejects_and_cleans_up(hook_runtime, replacement, error):
    _, instances = hook_runtime
    with pytest.raises(error):
        engine.simulate(
            _request(),
            *_providers(),
            ctot_correction=HybridCtotCorrection(lambda ctot, context: replacement),
        )
    assert not instances[-1].path.exists()
    assert "computeCovMatrices" not in instances[-1].__dict__


@pytest.mark.parametrize("policy", [NonPsdPolicy.ERROR, NonPsdPolicy.CLIP])
def test_corrected_covariance_obeys_existing_psd_policy(hook_runtime, policy):
    corrected = np.array([[[-1, 0], [0, 100]]] * 2) * u.nm**2
    kwargs = {"ctot_correction": HybridCtotCorrection(lambda ctot, context: corrected)}
    if policy is NonPsdPolicy.ERROR:
        with pytest.raises(ValueError, match="materially"):
            engine.simulate(_request(policy=policy), *_providers(), **kwargs)
    else:
        result = engine.simulate(_request(policy=policy), *_providers(), **kwargs)
        np.testing.assert_array_equal(result.psd_clipped, [True, True])
        np.testing.assert_array_equal(result.ctot_wavefront.value[:, 0, 0], [0, 0])


@pytest.mark.parametrize("stage", ["baseline", "correction"])
def test_failures_restore_instance_bindings_and_configuration(hook_runtime, stage):
    runtime, instances = hook_runtime
    original_init = runtime.__init__

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.computeCovMatrices = self.computeCovMatrices
        self.original_covariance = self.computeCovMatrices

    runtime.__init__ = initialize

    def fail(*args, **kwargs):
        raise RuntimeError("intentional failure")

    if stage == "baseline":
        runtime.computeTotalResidualMatrix = fail
    with pytest.raises(RuntimeError, match="intentional"):
        engine.simulate(
            _request(), *_providers(), ctot_correction=HybridCtotCorrection(fail)
        )
    model = instances[-1]
    assert model.computeCovMatrices is model.original_covariance
    assert "buildReconstuctor2" not in model.__dict__
    assert not model.path.exists()


@pytest.mark.parametrize("count", [1, 2])
def test_centre_missing_adds_only_required_covariance(hook_runtime, count):
    _, instances = hook_runtime
    request = replace(
        _request(),
        science_x=np.array([3, 4]) * u.arcsec,
        ngs_x=_request().ngs_x[:count],
        ngs_y=_request().ngs_y[:count],
        ngs_flux=_request().ngs_flux[:count],
    )
    science, ngs = _providers()
    ngs = _NgsProvider(
        NgsHoMetricPrediction(
            ngs.prediction.ee[:count],
            ngs.prediction.fwhm[:count],
            ngs.prediction.sr[:count],
        )
    )

    def noop(ctot, context):
        assert (context.mastsel_statistics.centre_ngs_covariance is None) == (
            count == 1
        )

    engine.simulate(request, science, ngs, ctot_correction=HybridCtotCorrection(noop))
    assert instances[-1].calls == {"total": 1, "covariance": count, "reconstructor": 1}


def test_nonfixed_gain_does_not_disable_generic_hook(hook_runtime):
    runtime, _ = hook_runtime
    runtime.LoopGain_LO = "optimize"

    def double(ctot, context):
        assert context.mastsel_statistics.measurement_noise is None
        return 2 * ctot

    result = engine.simulate(
        _request(), *_providers(), ctot_correction=HybridCtotCorrection(double)
    )
    np.testing.assert_array_equal(result.ctot_wavefront.value[:, 0, 0], [200, 400])


def test_provenance_collision_and_invalid_hook_fail_before_runtime(hook_runtime):
    _, instances = hook_runtime
    with pytest.raises(ValueError, match="collides"):
        engine.simulate(
            _request(),
            *_providers(),
            ctot_correction=HybridCtotCorrection(
                lambda ctot, context: None, {"science_provider": "other"}
            ),
        )
    with pytest.raises(TypeError, match="ctot_correction"):
        engine.simulate(
            _request(), *_providers(), ctot_correction=lambda ctot, context: None
        )
    assert not instances


def test_simulate_builds_owned_result_and_passes_resolved_mastsel_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeMavisLO:
        error = False
        mas2nm = 2.0

        def __init__(self, path: str, name: str, *, verbose: bool) -> None:
            calls["constructor"] = (path, name, verbose)

        def computeTotalResidualMatrix(
            self, *args: object, **kwargs: object
        ) -> np.ndarray:
            calls["args"] = args
            calls["kwargs"] = kwargs
            return np.stack([400.0 * np.eye(2), 800.0 * np.eye(2)])

    monkeypatch.setattr(_mastsel, "_load_mavis_lo", lambda: FakeMavisLO)
    science, ngs = _providers()
    result = engine.simulate(_request(), science, ngs)

    args = calls["args"]
    np.testing.assert_allclose(args[0], [[0.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(args[1], [[10.0, 0.0], [-5.0, 5.0 * np.sqrt(3.0)]])
    np.testing.assert_allclose(args[2], [1000.0, 2000.0])
    np.testing.assert_allclose(args[3], [500.0, 500.0])
    np.testing.assert_allclose(result.ngs_frame_rate.to_value(u.Hz), [500.0, 500.0])
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
    np.testing.assert_array_equal(
        result.diagnostics.pre_blur_psfs, science.prediction.psfs
    )
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


def test_magnitude_request_resolves_the_same_flux_and_psf_as_direct_flux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[np.ndarray] = []

    class FakeMavisLO:
        error = False
        mas2nm = 2.0

        def __init__(self, path: str, name: str, *, verbose: bool) -> None:
            pass

        def computeTotalResidualMatrix(
            self, *args: object, **kwargs: object
        ) -> np.ndarray:
            calls.append(np.asarray(args[2]))
            return np.stack([400.0 * np.eye(2), 800.0 * np.eye(2)])

    monkeypatch.setattr(_mastsel, "_load_mavis_lo", lambda: FakeMavisLO)
    zeropoint = 6_277_173_913.043478 * u.photon / (u.m**2 * u.s)
    magnitudes = np.array([12.0, 15.0]) * u.mag
    photometry = WFSPhotometryConfig(
        telescope_diameter=8.0 * u.m,
        n_channels=1.0,
        frame_rate=500.0 * u.Hz,
        zeropoint=zeropoint,
    )
    expected_flux = magnitudes_to_effective_photons_per_second(magnitudes, photometry)
    base = replace(_request(), dtype=np.float64)
    ini = base.mastsel_ini.replace(
        "ZenithAngle = 0.0", "ZenithAngle = 0.0\nTelescopeDiameter = 8.0"
    )
    magnitude_request = replace(
        base,
        mastsel_ini=ini,
        dtype=np.float64,
        ngs_flux=None,
        ngs_magnitude=magnitudes,
        ngs_magnitude_zeropoint=zeropoint,
    )
    science, ngs = _providers()
    magnitude_result = engine.simulate(magnitude_request, science, ngs)
    direct_result = engine.simulate(
        replace(base, mastsel_ini=ini, ngs_flux=expected_flux), science, ngs
    )
    np.testing.assert_allclose(calls[0], expected_flux.to_value(u.photon / u.s))
    np.testing.assert_array_equal(calls[0], calls[1])
    np.testing.assert_array_equal(
        magnitude_result.ctot_wavefront.value, direct_result.ctot_wavefront.value
    )
    np.testing.assert_array_equal(magnitude_result.psfs, direct_result.psfs)
    assert magnitude_result.ngs_magnitude is not None
    np.testing.assert_array_equal(
        magnitude_result.ngs_magnitude.value, magnitudes.value
    )
    assert magnitude_result.ngs_magnitude_zeropoint == zeropoint
    assert direct_result.ngs_magnitude is None


def test_directional_field_pairs_and_covariance_axes_reach_mastsel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, np.ndarray] = {}
    ctot = np.array(
        [
            [[100.0, 20.0], [20.0, 400.0]],
            [[200.0, -30.0], [-30.0, 500.0]],
        ]
    )

    class FakeMavisLO:
        error = False
        mas2nm = 2.0

        def __init__(self, *_: object, **__: object) -> None:
            pass

        def computeTotalResidualMatrix(
            self, science, ngs, *_: object, **__: object
        ) -> np.ndarray:
            calls["science"] = np.asarray(science)
            calls["ngs"] = np.asarray(ngs)
            return ctot

    monkeypatch.setattr(_mastsel, "_load_mavis_lo", lambda: FakeMavisLO)
    request = replace(
        _request(),
        science_x=np.array([12.0, -4.0]) * u.arcsec,
        science_y=np.array([-3.0, 8.0]) * u.arcsec,
        ngs_x=np.array([9.0, -7.0]) * u.arcsec,
        ngs_y=np.array([2.0, -6.0]) * u.arcsec,
    )
    science, ngs = _providers()
    directional_psfs = np.zeros((2, 33, 33), dtype=np.float32)
    directional_psfs[:, 15, 18] = 1.0
    directional_psfs[:, 21, 12] = 0.2
    science = _ScienceProvider(replace(science.prediction, psfs=directional_psfs))

    result = engine.simulate(request, science, ngs)

    np.testing.assert_array_equal(calls["science"], [[12.0, -3.0], [-4.0, 8.0]])
    np.testing.assert_array_equal(calls["ngs"], [[9.0, 2.0], [-7.0, -6.0]])
    np.testing.assert_array_equal(result.ctot_wavefront.to_value(u.nm**2), ctot)
    expected = blur_psfs_with_ctot(
        directional_psfs,
        ctot * u.nm**2,
        pixel_scale=8.0 * u.mas,
        angle_to_wavefront_scale=2.0 * u.nm / u.mas,
    )
    np.testing.assert_array_equal(result.psfs, expected)


def test_simulate_uses_frame_rate_from_ini(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, np.ndarray] = {}

    class FakeMavisLO:
        error = False
        mas2nm = 2.0

        def __init__(self, *_: object, **__: object) -> None:
            pass

        def computeTotalResidualMatrix(self, *args: object, **__: object) -> np.ndarray:
            calls["frame_rate"] = np.asarray(args[3], dtype=float)
            return np.stack([400.0 * np.eye(2), 800.0 * np.eye(2)])

    monkeypatch.setattr(_mastsel, "_load_mavis_lo", lambda: FakeMavisLO)
    science, ngs = _providers()
    result = engine.simulate(_request(frame_rate="250.0"), science, ngs)

    np.testing.assert_allclose(calls["frame_rate"], [250.0, 250.0])
    np.testing.assert_allclose(result.ngs_frame_rate.to_value(u.Hz), [250.0, 250.0])


@pytest.mark.parametrize("frame_rate", ["0", "-1", "nan", "invalid"])
def test_simulate_rejects_invalid_ini_frame_rate(frame_rate: str) -> None:
    science, ngs = _providers()
    with pytest.raises(ValueError, match="RTC.SensorFrameRate_LO"):
        engine.simulate(_request(frame_rate=frame_rate), science, ngs)


def test_simulate_requires_ini_frame_rate() -> None:
    request = _request()
    request = replace(
        request,
        mastsel_ini=request.mastsel_ini.replace("SensorFrameRate_LO = 500.0\n", ""),
    )
    science, ngs = _providers()
    with pytest.raises(ValueError, match="RTC.SensorFrameRate_LO"):
        engine.simulate(request, science, ngs)


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

    monkeypatch.setattr(_mastsel, "_load_mavis_lo", lambda: FakeMavisLO)
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

    monkeypatch.setattr(_mastsel, "_load_mavis_lo", lambda: FakeMavisLO)
    science, ngs = _providers()

    result = engine.simulate(_request(diagnostics=DiagnosticsLevel.NONE), science, ngs)

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

    monkeypatch.setattr(_mastsel, "_load_mavis_lo", lambda: FakeMavisLO)
    science, ngs = _providers()

    result = engine.simulate(
        replace(_request(diagnostics=DiagnosticsLevel.VALIDATION), dtype=np.float64),
        science,
        ngs,
    )

    np.testing.assert_array_equal(result.psd_clipped, [True, True])
    assert result.diagnostics is not None
    np.testing.assert_array_equal(result.diagnostics.psd_valid_mask, [True, True])
    assert not np.array_equal(result.psfs, science.prediction.psfs)


def test_condition_ctot_distinguishes_numerical_and_material_negative_eigenvalues() -> (
    None
):
    numerical = np.array([[[-0.5e-9, 0.0], [0.0, 2.0]]]) * u.nm**2
    conditioned, clipped = engine.condition_ctot(numerical)
    np.testing.assert_allclose(
        conditioned.to_value(u.nm**2), [[[0.0, 0.0], [0.0, 2.0]]]
    )
    np.testing.assert_array_equal(clipped, [True])

    material = np.array([[[-1.0, 0.0], [0.0, 2.0]]]) * u.nm**2
    with pytest.raises(ValueError, match="materially non-positive-semidefinite"):
        engine.condition_ctot(material, policy="error")
    conditioned, clipped = engine.condition_ctot(material, policy="clip")
    np.testing.assert_allclose(
        conditioned.to_value(u.nm**2), [[[0.0, 0.0], [0.0, 2.0]]]
    )
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
    psfs = np.zeros((1, 9, 9), dtype=np.float64)
    psfs[0, 4, 4] = 2.0
    psfs[0, 2, 5] = 0.5
    psfs[0, 6, 3] = 0.25
    otf = image_plane_gaussian_otf(
        np.zeros((2, 2)),
        (18, 18),
        mean_shift_pix=np.array([2.0, -1.0]),
        dtype=np.float64,
    )
    apply_finite_fov_otfs(psfs, otf[None])

    expected = np.zeros_like(psfs)
    expected[0, 3, 6] = 2.0
    expected[0, 1, 7] = 0.5
    expected[0, 5, 5] = 0.25
    np.testing.assert_allclose(psfs, expected, rtol=0.0, atol=2.0e-7)


def test_directional_covariance_blurs_x_columns_and_y_rows() -> None:
    yy, xx = np.indices((65, 65), dtype=float)
    psfs = np.zeros((1, 65, 65), dtype=np.float32)
    psfs[0] = np.exp(
        -((xx - 34.0) ** 2 + (yy - 29.0) ** 2) / (2.0 * 1.7**2)
    ) + 0.2 * np.exp(-((xx - 28.0) ** 2 + (yy - 36.0) ** 2) / (2.0 * 1.7**2))
    ctot_xy = np.array([[[4.0, 1.2], [1.2, 1.0]]])

    blurred = blur_psfs_with_ctot(
        psfs,
        ctot_xy * u.nm**2,
        pixel_scale=1.0 * u.mas,
        angle_to_wavefront_scale=1.0 * u.nm / u.mas,
    )

    def moments(image: np.ndarray) -> np.ndarray:
        weights = image / image.sum()
        dx = xx - np.sum(weights * xx)
        dy = yy - np.sum(weights * yy)
        return np.array(
            [
                [np.sum(weights * dx * dx), np.sum(weights * dx * dy)],
                [np.sum(weights * dx * dy), np.sum(weights * dy * dy)],
            ]
        )

    np.testing.assert_allclose(
        moments(blurred[0]) - moments(psfs[0]), ctot_xy[0], atol=2.0e-4
    )
