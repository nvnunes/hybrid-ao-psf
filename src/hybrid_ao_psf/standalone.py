"""Standalone configuration, file loading, and output workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import DTypeLike

from ._coordinates import _cartesian_to_polar
from ._e2e_model import E2ECalibratedModel, load_e2e_calibrated_model
from ._mastsel import MastselModel
from ._version import __version__
from .config import HybridConfiguration, load_configuration
from .contracts import HybridResult, LowOrderModel
from .engine import simulate
from .interpolation import (
    load_psf_interpolator,
    load_psf_metric_interpolator,
)
from .result_io import _save_result


def simulate_configuration(configuration: HybridConfiguration) -> HybridResult:
    """Load the configuration's fitted providers and run one simulation.

    Args:
        configuration: Parsed Hybrid configuration with a model-specific
            request and paths for the required trusted interpolation files.

    Returns:
        Final science PSFs, jitter and any requested diagnostics.

    Raises:
        TypeError: If configuration is not a HybridConfiguration.
        ValueError: If model-specific inputs are missing or invalid.
        FileNotFoundError: If a required file is missing.
    """
    if not isinstance(configuration, HybridConfiguration):
        raise TypeError("configuration must be a HybridConfiguration.")
    science_psfs = load_psf_interpolator(configuration.science_psf_file)
    return simulate(
        configuration.request, science_psfs, lo_model=_load_model(configuration)
    )


def _load_model(
    configuration: HybridConfiguration,
) -> MastselModel | E2ECalibratedModel:
    """Open the files required by the selected low-order model."""
    if configuration.lo_model is LowOrderModel.E2E_CALIBRATED:
        if configuration.e2e_configuration is None:
            raise ValueError("E2E configuration requires its shared model INI")
        return load_e2e_calibrated_model(
            configuration.e2e_configuration, dtype=configuration.request.dtype
        )
    if configuration.ngs_metric_file is None or configuration.mastsel_ini is None:
        raise ValueError("MASTSEL configuration requires its INI and NGS metrics")
    return MastselModel(
        mastsel_ini=configuration.mastsel_ini,
        ngs_metrics=load_psf_metric_interpolator(configuration.ngs_metric_file),
        non_psd_policy=configuration.non_psd_policy,
        ngs_magnitude_zeropoint=configuration.ngs_magnitude_zeropoint,
        _ngs_metric_file=configuration.ngs_metric_file,
    )


def _simulation_settings(
    configuration: HybridConfiguration, model: MastselModel | E2ECalibratedModel
) -> dict[str, str]:
    """Record the selected model, original INI and resolved provider paths."""
    settings = {
        "name": configuration.lo_model.value,
        "version": __version__,
        "base_config": configuration.base_config,
        "diagnostics_level": configuration.request.diagnostics_level.value,
        "science_ho_psf_interpolator_path": str(configuration.science_psf_file),
        "ngs_ho_metric_interpolator_path": model._input_files["ngs_metrics"],
    }
    if configuration.ngs_mag_standard is not None:
        settings["ngs_mag_standard"] = configuration.ngs_mag_standard
    if isinstance(model, MastselModel):
        settings["non_psd_policy"] = configuration.non_psd_policy.value
    else:
        for family in ("covariance", "coma", "sensing", "reconstructor"):
            settings[f"{family}_model_path"] = model._input_files[f"{family}_model"]
    return settings


def run_configuration(
    path: str | Path,
    *,
    output: str | Path | None = None,
    overwrite: bool = False,
    dtype: DTypeLike | None = None,
) -> HybridResult:
    """Run a single Hybrid INI and save its standalone HDF5 result.

    Args:
        path: Path to the combined or E2E-Calibrated Hybrid INI.
        output: Optional destination relative to the current working
            directory if supplied as a relative path. The default uses the
            output path resolved from the INI.
        overwrite: Whether an existing result file may be replaced.
        dtype: Optional float32 or float64 override of the configured precision.

    Returns:
        The saved simulation result.

    Raises:
        FileNotFoundError: If the INI or a required provider is missing.
        FileExistsError: If the result already exists and overwrite is false.
        ValueError: If configuration or scientific inputs are invalid.
    """
    configuration = load_configuration(path, dtype=dtype)
    destination = (
        configuration.output if output is None else Path(output).expanduser().resolve()
    )

    science_psfs = load_psf_interpolator(configuration.science_psf_file)
    model = _load_model(configuration)
    result = simulate(configuration.request, science_psfs, lo_model=model)
    request = configuration.request
    ngs_r, ngs_theta = _cartesian_to_polar(request.ngs_positions.astype(np.float64))
    setup = {}
    if configuration.ngs_magnitude_zeropoint is not None:
        setup["ngs_magnitude_zeropoint"] = configuration.ngs_magnitude_zeropoint.astype(
            request.dtype
        )
    _save_result(
        destination,
        result,
        overwrite=overwrite,
        simulation=_simulation_settings(configuration, model),
        setup=setup,
        options={
            "zenith_angle": request.zenith_angle,
            "ngs_r": ngs_r,
            "ngs_theta": ngs_theta,
            "ngs_magnitude": request.ngs_magnitude,
        },
    )
    return result


__all__ = ["run_configuration", "simulate_configuration"]
