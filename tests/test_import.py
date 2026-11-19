from __future__ import annotations

from importlib import metadata

from ao_core import stats
from mastsel import MavisLO

import hybrid_ao_psf
from hybrid_ao_psf import interpolation


def test_package_root_exports_supported_contracts() -> None:
    assert hybrid_ao_psf.__version__ == "0.1.0"
    assert not hasattr(hybrid_ao_psf, "HYBRID_SECTION")
    core_names = {
        "diagnostic_fields",
        "diagnostic_values",
        "DiagnosticsLevel",
        "HybridConfiguration",
        "HybridCtotContext",
        "HybridCtotCorrection",
        "HybridDiagnostics",
        "CovarianceDiagnostics",
        "MastselDiagnostics",
        "HybridMastselStatistics",
        "HybridRequest",
        "HybridResult",
        "LowOrderModel",
        "MastselModel",
        "LowOrderDistribution",
        "NonPsdPolicy",
        "PsfMetricProvider",
        "PsfProvider",
        "PsfFileProvider",
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
        "SensingPrediction",
        "SensingResponseModel",
        "SensingResponseResult",
        "build_sensing_response",
        "SensingNoiseRequest",
        "SensingNoiseMeasurements",
        "simulate_sensing_noise",
        "save_sensing_noise",
        "load_sensing_noise",
        "SensingModel",
        "SensingCalibrationResult",
        "build_sensing_calibration",
        "save_sensing_calibration",
        "load_sensing_calibration",
        "SensingCalibrationMeasurements",
        "save_sensing_measurements",
        "load_sensing_measurements",
        "ComaCalibrationMeasurements",
        "ComaCalibrationResult",
        "ComaResponseModel",
        "build_coma_calibration",
        "save_coma_calibration",
        "load_coma_calibration",
        "save_coma_measurements",
        "load_coma_measurements",
        "CovarianceCalibrationMeasurements",
        "CovarianceCalibrationResult",
        "CovarianceFunctions",
        "measure_covariance_calibration",
        "build_covariance_calibration",
        "save_covariance_calibration",
        "load_covariance_calibration",
        "save_covariance_measurements",
        "load_covariance_measurements",
        "ReconstructorCalibrationMeasurements",
        "ReconstructorCalibrationResult",
        "ReconstructorNoiseModel",
        "build_reconstructor_calibration",
        "build_lo_reconstructor",
        "solve_lo_reconstructor",
        "save_reconstructor_calibration",
        "load_reconstructor_calibration",
        "save_reconstructor_measurements",
        "load_reconstructor_measurements",
        "E2ECalibratedModel",
        "E2ELowOrderDiagnostics",
        "E2ELowOrderPrediction",
        "load_e2e_calibrated_model",
        "E2ECalibration",
        "FixedGainIntegrator",
        "NoiseBlurModel",
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
        "CalibrationFile",
        "CalibrationModelFiles",
        "save_calibration_data",
        "load_calibration_data",
        "read_calibration_model_files",
    }
    assert set(hybrid_ao_psf.__all__) == core_names | calibration_names | set(
        interpolation.__all__
    )


def test_required_dependencies_are_available() -> None:
    assert stats.PsfMetadata is not None
    assert metadata.version("mastsel")
    assert MavisLO is not None


def test_distribution_metadata_matches_package() -> None:
    assert metadata.version("hybrid-ao-psf") == hybrid_ao_psf.__version__


def test_psf_validation_is_internal_and_comparison_remains_public() -> None:
    for surface in (hybrid_ao_psf, interpolation):
        for family in ("psf", "psf_metric", "science_ho_psf", "ngs_ho_metric"):
            for operation in ("interpolator", "coordinates"):
                name = f"validate_{family}_{operation}"
                assert name not in surface.__all__
                assert not hasattr(surface, name)
        assert callable(surface.compare_psf_interpolator)
        assert callable(surface.compare_psf_metric_interpolator)


def test_interpolation_exports_only_generic_psf_names() -> None:
    for surface in (hybrid_ao_psf, interpolation):
        assert not any(
            name.startswith(("ScienceHo", "NgsHo")) for name in surface.__all__
        )
        assert not hasattr(surface, "SciencePsfProvider")
        assert not hasattr(surface, "NgsMetricProvider")


def test_sensing_types_have_no_calibration_name_aliases() -> None:
    assert hybrid_ao_psf.SensingModel.__name__ == "SensingModel"
    assert hybrid_ao_psf.SensingResponseResult.__name__ == "SensingResponseResult"
    assert not hasattr(hybrid_ao_psf, "SensingCalibration")
    assert not hasattr(hybrid_ao_psf, "SensingResponseCalibration")


def test_former_calibration_file_names_are_not_public() -> None:
    for name in (
        "CalibrationProductIdentity",
        "CalibrationFileIdentity",
        "read_calibration_product_identity",
    ):
        assert name not in hybrid_ao_psf.__all__
        assert not hasattr(hybrid_ao_psf, name)


def test_former_calibration_data_names_are_not_public() -> None:
    for name in ("save_calibration_samples", "load_calibration_samples"):
        assert name not in hybrid_ao_psf.__all__
        assert not hasattr(hybrid_ao_psf, name)
