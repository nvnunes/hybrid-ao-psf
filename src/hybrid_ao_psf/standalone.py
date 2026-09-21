"""Standalone configuration, artifact-loading, and output workflow."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .config import HybridConfiguration, load_configuration
from .contracts import HybridResult
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
    ngs_provider = load_ngs_ho_metric_interpolator(configuration.ngs_metric_artifact)
    result = simulate(configuration.request, science_provider, ngs_provider)
    return replace(
        result,
        provenance={
            **result.provenance,
            "configuration": str(configuration.source),
            "science_psf_artifact": str(configuration.science_psf_artifact),
            "ngs_metric_artifact": str(configuration.ngs_metric_artifact),
        },
    )


def run_configuration(
    path: str | Path,
    *,
    output: str | Path | None = None,
    overwrite: bool = False,
) -> HybridResult:
    """Run one combined INI and write its versioned standalone result."""
    configuration = load_configuration(path)
    result = simulate_configuration(configuration)
    destination = (
        configuration.output
        if output is None
        else Path(output).expanduser().resolve()
    )
    save_result(destination, result, overwrite=overwrite)
    return result


__all__ = ["run_configuration", "simulate_configuration"]
