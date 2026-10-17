"""Standalone configuration, artifact-loading, and output workflow."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ._e2e_model import load_e2e_calibrated_model
from .config import HybridConfiguration, load_configuration
from .contracts import HybridResult, LowOrderModel
from .engine import simulate
from .interpolation import (
    load_ngs_ho_metric_interpolator,
    load_science_ho_psf_interpolator,
)
from .result_io import save_result


def simulate_configuration(configuration: HybridConfiguration) -> HybridResult:
    """Load configured providers and execute the public stateless engine."""
    if not isinstance(configuration, HybridConfiguration):
        raise TypeError("configuration must be a HybridConfiguration.")
    science_provider = load_science_ho_psf_interpolator(
        configuration.science_psf_artifact
    )
    if configuration.request.lo_model is LowOrderModel.E2E_CALIBRATED:
        if configuration.e2e_configuration is None:
            raise ValueError("E2E configuration requires its shared model INI")
        model = load_e2e_calibrated_model(
            configuration.e2e_configuration, dtype=configuration.request.dtype
        )
        result = simulate(configuration.request, science_provider, e2e_model=model)
    else:
        if configuration.ngs_metric_artifact is None:
            raise ValueError("MASTSEL configuration requires NGS metrics")
        ngs_provider = load_ngs_ho_metric_interpolator(
            configuration.ngs_metric_artifact
        )
        result = simulate(configuration.request, science_provider, ngs_provider)
    return replace(
        result,
        provenance={
            **result.provenance,
            "configuration": str(configuration.source),
            "science_psf_artifact": str(configuration.science_psf_artifact),
            **(
                {"ngs_metric_artifact": str(configuration.ngs_metric_artifact)}
                if configuration.ngs_metric_artifact is not None
                else {}
            ),
        },
    )


def run_configuration(
    path: str | Path,
    *,
    output: str | Path | None = None,
    overwrite: bool = False,
    dtype=None,
) -> HybridResult:
    """Run one selected-model INI and write its versioned standalone result."""
    configuration = load_configuration(path, dtype=dtype)
    result = simulate_configuration(configuration)
    destination = (
        configuration.output if output is None else Path(output).expanduser().resolve()
    )
    save_result(destination, result, overwrite=overwrite)
    return result


__all__ = ["run_configuration", "simulate_configuration"]
