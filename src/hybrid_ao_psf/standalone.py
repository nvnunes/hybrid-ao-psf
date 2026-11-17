"""Standalone configuration, file loading, and output workflow."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from numpy.typing import DTypeLike

from ._e2e_model import load_e2e_calibrated_model
from .config import HybridConfiguration, load_configuration
from .contracts import HybridResult, LowOrderModel
from .engine import simulate
from .interpolation import (
    load_psf_interpolator,
    load_psf_metric_interpolator,
)
from .result_io import save_result


def simulate_configuration(configuration: HybridConfiguration) -> HybridResult:
    """Load the configuration's fitted providers and run one simulation.

    Args:
        configuration: Parsed Hybrid configuration with a model-specific
            request and paths for the required trusted interpolation files.

    Returns:
        The simulation result, with configuration and provider file paths
        in its provenance.

    Raises:
        TypeError: If configuration is not a HybridConfiguration.
        ValueError: If model-specific inputs are missing or invalid.
        FileNotFoundError: If a required file is missing.
    """
    if not isinstance(configuration, HybridConfiguration):
        raise TypeError("configuration must be a HybridConfiguration.")
    science_psfs = load_psf_interpolator(configuration.science_psf_file)
    if configuration.request.lo_model is LowOrderModel.E2E_CALIBRATED:
        if configuration.e2e_configuration is None:
            raise ValueError("E2E configuration requires its shared model INI")
        model = load_e2e_calibrated_model(
            configuration.e2e_configuration, dtype=configuration.request.dtype
        )
        result = simulate(configuration.request, science_psfs, e2e_model=model)
    else:
        if configuration.ngs_metric_file is None:
            raise ValueError("MASTSEL configuration requires NGS metrics")
        ngs_metrics = load_psf_metric_interpolator(configuration.ngs_metric_file)
        result = simulate(configuration.request, science_psfs, ngs_metrics)
    return replace(
        result,
        provenance={
            **result.provenance,
            "configuration": str(configuration.source),
            "science_psf_file": str(configuration.science_psf_file),
            **(
                {"ngs_metric_file": str(configuration.ngs_metric_file)}
                if configuration.ngs_metric_file is not None
                else {}
            ),
        },
    )


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
    result = simulate_configuration(configuration)
    destination = (
        configuration.output if output is None else Path(output).expanduser().resolve()
    )
    save_result(destination, result, overwrite=overwrite)
    return result


__all__ = ["run_configuration", "simulate_configuration"]
