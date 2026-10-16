from __future__ import annotations

from importlib import metadata

import ao_stats
from mastsel import MavisLO

import hybrid_ao_psf
from hybrid_ao_psf import interpolation


def test_package_root_exports_supported_contracts() -> None:
    assert hybrid_ao_psf.__version__ == "0.1.0"
    core_names = {
        "HYBRID_SECTION",
        "RESULT_SCHEMA",
        "RESULT_SCHEMA_VERSION",
        "DiagnosticsLevel",
        "HybridConfiguration",
        "HybridCtotContext",
        "HybridCtotCorrection",
        "HybridDiagnostics",
        "HybridMastselStatistics",
        "HybridRequest",
        "HybridResult",
        "NgsHoMetricPrediction",
        "NgsMetricProvider",
        "NonPsdPolicy",
        "ScienceHoPsfPrediction",
        "SciencePsfProvider",
        "__version__",
        "apply_finite_fov_otfs",
        "blur_psfs_with_ctot",
        "image_plane_gaussian_otf",
        "load_configuration",
        "load_result",
        "run_configuration",
        "save_result",
        "simulate",
        "simulate_configuration",
    }
    calibration_names = {
        "E2ECalibration",
        "FixedGainIntegrator",
        "CalibrationPhotometry",
        "CovarianceTelemetry",
        "ComaWavefronts",
        "ComaWfsSampling",
        "ComaWfsMeasurements",
        "ComaResponseMeasurements",
        "SensingSpot",
        "SensingWfsSampling",
        "SensingWfsMeasurements",
        "ReconstructorNoiseInputs",
        "ReconstructorNoiseMeasurements",
        "CovarianceCalibrationSettings",
        "ComaCalibrationSettings",
        "SensingCalibrationSettings",
        "ReconstructorCalibrationSettings",
        "E2ECalibrationConfiguration",
        "load_e2e_calibration_configuration",
        "CalibrationFileIdentity",
        "CalibrationArtifactIdentity",
        "save_calibration_samples",
        "load_calibration_samples",
        "read_calibration_artifact_identity",
    }
    assert set(hybrid_ao_psf.__all__) == core_names | calibration_names | set(
        interpolation.__all__
    )


def test_required_dependencies_are_available() -> None:
    assert ao_stats.__version__
    assert metadata.version("mastsel")
    assert MavisLO is not None


def test_distribution_metadata_matches_package() -> None:
    assert metadata.version("hybrid-ao-psf") == hybrid_ao_psf.__version__
