"""Save and load standalone Hybrid results using AO Core."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from ao_core.io import initialize_result, read_result, write_observation
from ao_core.io.ini import polar_to_cartesian
from ao_core.stats import PsfMetadata
from astropy import units as u

from ._coordinates import _cartesian_to_polar
from ._diagnostic_io import _restore_diagnostics, diagnostic_fields, diagnostic_values
from ._io import atomic_output_path
from .contracts import DiagnosticsLevel, HybridResult, LowOrderModel, _jitter_from_ctot


def _result_diagnostic_fields(
    result: HybridResult,
    values: Mapping[str, Any],
    simulation: Mapping[str, Any],
    options: Mapping[str, Any],
) -> tuple[DiagnosticsLevel, dict[str, dict[str, Any]]]:
    """Declare the configured fields or the diagnostics present on a direct save."""
    configured = "name" in simulation
    if configured:
        model = LowOrderModel(simulation["name"])
        level = DiagnosticsLevel(simulation["diagnostics_level"])
        num_ngs = (
            len(options["ngs_r"]) if model is LowOrderModel.E2E_CALIBRATED else None
        )
    else:
        diagnostics = result.diagnostics
        if diagnostics is None:
            return DiagnosticsLevel.NONE, {}
        mastsel = diagnostics.mastsel
        calibrated = diagnostics.e2e_calibrated
        debug = (
            diagnostics.psfs_ho is not None
            or calibrated is not None
            or (mastsel is not None and mastsel.effective_ini is not None)
        )
        level = DiagnosticsLevel.DEBUG if debug else DiagnosticsLevel.VALIDATION
        model = (
            LowOrderModel.MASTSEL
            if mastsel is not None or diagnostics.covariance.psd_clipped is not None
            else LowOrderModel.E2E_CALIBRATED
        )
        num_ngs = None
        if mastsel is not None:
            num_ngs = len(mastsel.ngs_metrics.ee)
        elif calibrated is not None:
            num_ngs = len(calibrated.diagnostics.ngs.metrics.ee)
    specs = diagnostic_fields(
        model,
        level,
        dtype=result.psfs.dtype,
        psf_shape=result.psfs.shape[-2:],
        num_ngs=num_ngs,
    )
    if not configured:
        specs = {path: spec for path, spec in specs.items() if path in values}
    return level, specs


def _check_result_diagnostics(result: HybridResult) -> None:
    """Check the shared result against the saved covariance and debug mixture."""
    diagnostics = result.diagnostics
    if diagnostics is None:
        return
    ctot = diagnostics.covariance.ctot
    if not np.allclose(
        result.jitter.value, _jitter_from_ctot(ctot).value, rtol=1e-6, atol=1e-6
    ):
        raise ValueError("jitter must equal the radial jitter derived from ctot.")
    if diagnostics.e2e_calibrated is not None:
        if diagnostics.covariance.psd_clipped is not None:
            raise ValueError("E2E-Calibrated results do not apply covariance clipping.")
        distribution = diagnostics.e2e_calibrated.displacement_distribution
        if not np.allclose(
            ctot.value,
            distribution.covariance.value,
            rtol=max(1e-10, 16 * np.finfo(ctot.dtype).eps),
            atol=1e-8,
        ):
            raise ValueError(
                "Diagnostic ctot must match the applied distribution covariance."
            )


def _observation_scalar(value: u.Quantity, name: str) -> u.Quantity:
    """Return a common physical value for a single science observation."""
    if value.isscalar:
        return value
    if np.all(value == value[0]):
        return value[0]
    raise ValueError(f"A standalone result requires one common {name}.")


def _write_result(
    output: Path,
    result: HybridResult,
    simulation: Mapping[str, Any],
    setup: Mapping[str, Any],
    options: Mapping[str, Any],
) -> None:
    diagnostics = diagnostic_values(result.diagnostics)
    level, fields = _result_diagnostic_fields(result, diagnostics, simulation, options)
    radius, angle = _cartesian_to_polar(result.positions.astype(np.float64))
    metadata = result.metadata
    simulation = {
        "diagnostics_level": level.value,
        **simulation,
        "precision": result.psfs.dtype.name,
        "extra_stat_fields": {"jitter": "mas"},
    }
    if fields:
        simulation["diagnostic_fields"] = fields
    with h5py.File(output, "w") as file:
        initialize_result(
            file,
            simulation=simulation,
            setup={**setup, "sci_r": radius, "sci_theta": angle},
            options={
                **options,
                "wavelength": _observation_scalar(metadata.wavelength, "wavelength"),
            },
            save_psfs=True,
        )
        write_observation(
            file,
            meta={
                "pixel_scale": _observation_scalar(metadata.pixel_scale, "pixel scale"),
                "tel_diameter": metadata.tel_diameter,
                "tel_pupil": metadata.tel_pupil,
            },
            psfs=result.psfs,
            stats={"jitter": result.jitter},
            diagnostics=diagnostics,
        )


def _save_result(
    path: str | Path,
    result: HybridResult,
    *,
    overwrite: bool,
    simulation: Mapping[str, Any],
    setup: Mapping[str, Any],
    options: Mapping[str, Any],
) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Result already exists: {output}")
    _check_result_diagnostics(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(output) as temporary_output:
        _write_result(temporary_output, result, simulation, setup, options)
    return output


def save_result(
    path: str | Path,
    result: HybridResult,
    *,
    overwrite: bool = False,
) -> Path:
    """Save one Hybrid observation and its diagnostics in AO Core's HDF5 format.

    Results must have science positions, a common wavelength and a common
    pixel scale. Use ``run_configuration()`` to also store the INI and run settings.

    Args:
        path: Destination HDF5 path. Parent directories are created as needed.
        result: Science PSFs, jitter, metadata and optional diagnostics.
        overwrite: Whether to atomically replace an existing destination.

    Returns:
        The absolute path of the saved file.

    Raises:
        FileExistsError: If the destination exists and overwrite is false.
        TypeError: If result is not a HybridResult.
        ValueError: If result data disagree with the standalone format or
            the scientific diagnostics disagree with the result.
    """
    if not isinstance(result, HybridResult):
        raise TypeError("result must be a HybridResult.")
    return _save_result(
        path, result, overwrite=overwrite, simulation={}, setup={}, options={}
    )


def load_result(path: str | Path) -> HybridResult:
    """Read science outputs and Hybrid diagnostics from a standalone AO Core file.

    Args:
        path: Path to the single-observation file. Referenced inputs are
            unnecessary when reading a result.

    Returns:
        Immutable outputs with Cartesian science positions.

    Raises:
        FileNotFoundError: If the file is missing.
        KeyError: If required Hybrid outputs are absent.
        ValueError: If the stored file or scientific values are invalid.
    """
    with h5py.File(Path(path).expanduser(), "r") as file:
        values = read_result(
            file, sections=("setup", "options", "meta", "psfs", "stats", "diagnostics")
        )
    wavelength = values["options"]["wavelength"]
    if not wavelength.isscalar:
        raise ValueError(
            "A Hybrid result must contain a single standalone observation."
        )
    setup = values["setup"]
    x, y = polar_to_cartesian(setup["sci_r"], setup["sci_theta"].to_value(u.deg))
    psfs = values["psfs"]["data"]
    result = HybridResult(
        positions=np.column_stack((x, y)),
        psfs=psfs,
        jitter=values["stats"]["jitter"],
        metadata=PsfMetadata(
            wavelength=wavelength,
            pixel_scale=values["meta"]["pixel_scale"],
            tel_diameter=values["meta"]["tel_diameter"],
            tel_pupil=values["meta"]["tel_pupil"],
        ),
        diagnostics=_restore_diagnostics(values["diagnostics"], psfs.dtype)
        if "diagnostics" in values
        else None,
    )
    _check_result_diagnostics(result)
    return result


__all__ = ["load_result", "save_result"]
