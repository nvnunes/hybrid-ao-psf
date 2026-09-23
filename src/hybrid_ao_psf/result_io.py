"""Versioned HDF5 persistence for standalone Hybrid AO PSF results."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from ao_stats import PsfMetadata
from astropy import units as u

from ._io import atomic_output_path
from .contracts import HybridDiagnostics, HybridResult, NgsHoMetricPrediction

RESULT_SCHEMA = "hybrid-ao-psf-result"
RESULT_SCHEMA_VERSION = 1


def save_result(
    path: str | Path,
    result: HybridResult,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one result using the current Hybrid AO PSF HDF5 schema."""
    if not isinstance(result, HybridResult):
        raise TypeError("result must be a HybridResult.")
    output = Path(path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Result already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(output) as temporary_output:
        _write_result(temporary_output, result)
    return output


def _write_result(output: Path, result: HybridResult) -> None:
    with h5py.File(output, "w") as root:
        root.attrs["schema"] = RESULT_SCHEMA
        root.attrs["schema_version"] = RESULT_SCHEMA_VERSION

        result_group = root.create_group("result")
        result_group.create_dataset("psfs", data=result.psfs, dtype=np.float32)
        _write_metadata(result_group.create_group("metadata"), result.metadata)
        _write_quantity_mapping(
            result_group.create_group("source_metadata"), result.source_metadata
        )
        metrics = result_group.create_group("ngs_metrics")
        _write_quantity(metrics, "ee", result.ngs_metrics.ee, dtype=np.float32)
        _write_quantity(metrics, "fwhm", result.ngs_metrics.fwhm, dtype=np.float32)
        _write_quantity(metrics, "sr", result.ngs_metrics.sr, dtype=np.float32)
        ctot = result_group.create_group("ctot")
        _write_quantity(ctot, "wavefront", result.ctot_wavefront, dtype=np.float64)
        _write_quantity(ctot, "angle", result.ctot_angle, dtype=np.float64)
        _write_quantity(
            ctot,
            "angle_to_wavefront_scale",
            result.angle_to_wavefront_scale,
            dtype=np.float64,
        )
        ctot.create_dataset("psd_clipped", data=result.psd_clipped, dtype=bool)
        ngs = result_group.create_group("ngs")
        _write_quantity(ngs, "flux", result.ngs_flux, dtype=np.float32)
        if result.ngs_magnitude is not None:
            _write_quantity(ngs, "magnitude", result.ngs_magnitude, dtype=np.float64)
            _write_quantity(
                ngs,
                "magnitude_zeropoint",
                result.ngs_magnitude_zeropoint,
                dtype=np.float64,
            )
        _write_quantity(
            ngs, "frame_rate", result.ngs_frame_rate, dtype=np.float32
        )
        _write_quantity(result_group, "jitter", result.jitter, dtype=np.float32)

        configuration = root.create_group("configuration")
        configuration.create_dataset(
            "effective_mastsel_ini",
            data=result.effective_mastsel_ini,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        provenance = root.create_group("provenance")
        for name, value in result.provenance.items():
            provenance.attrs[name] = value
        if result.diagnostics is not None:
            diagnostics = root.create_group("diagnostics")
            diagnostics.create_dataset(
                "psd_valid_mask",
                data=result.diagnostics.psd_valid_mask,
                dtype=bool,
            )
            _write_quantity(
                diagnostics,
                "ctot_angle_trace_min",
                result.diagnostics.ctot_angle_trace_min,
                dtype=np.float64,
            )
            _write_quantity(
                diagnostics,
                "ctot_angle_trace_max",
                result.diagnostics.ctot_angle_trace_max,
                dtype=np.float64,
            )
            _write_quantity(
                diagnostics,
                "ctot_angle_trace_mean",
                result.diagnostics.ctot_angle_trace_mean,
                dtype=np.float64,
            )
            _write_quantity(
                diagnostics,
                "ctot_angle_determinant_min",
                result.diagnostics.ctot_angle_determinant_min,
                dtype=np.float64,
            )
            _write_quantity(
                diagnostics,
                "ctot_angle_determinant_max",
                result.diagnostics.ctot_angle_determinant_max,
                dtype=np.float64,
            )
            _write_quantity(
                diagnostics,
                "ctot_angle_determinant_mean",
                result.diagnostics.ctot_angle_determinant_mean,
                dtype=np.float64,
            )
            if result.diagnostics.pre_blur_psfs is not None:
                diagnostics.create_dataset(
                    "pre_blur_psfs",
                    data=result.diagnostics.pre_blur_psfs,
                    dtype=np.float32,
                )


def load_result(path: str | Path) -> HybridResult:
    """Load and validate one current-schema Hybrid AO PSF result."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Hybrid AO PSF result not found: {source}")
    with h5py.File(source, "r") as root:
        schema = _text(root.attrs.get("schema", ""))
        version = int(root.attrs.get("schema_version", -1))
        if schema != RESULT_SCHEMA or version != RESULT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported Hybrid AO PSF result schema: "
                f"schema={schema!r}, version={version}."
            )
        result_group = root["result"]
        metadata = _read_metadata(result_group["metadata"])
        metrics = result_group["ngs_metrics"]
        ctot = result_group["ctot"]
        ngs = result_group["ngs"]
        diagnostics_group = root.get("diagnostics")
        diagnostics = None
        if diagnostics_group is not None:
            pre_blur = diagnostics_group.get("pre_blur_psfs")
            diagnostics = HybridDiagnostics(
                psd_valid_mask=diagnostics_group["psd_valid_mask"][...],
                ctot_angle_trace_min=_read_quantity(
                    diagnostics_group["ctot_angle_trace_min"]
                ),
                ctot_angle_trace_max=_read_quantity(
                    diagnostics_group["ctot_angle_trace_max"]
                ),
                ctot_angle_trace_mean=_read_quantity(
                    diagnostics_group["ctot_angle_trace_mean"]
                ),
                ctot_angle_determinant_min=_read_quantity(
                    diagnostics_group["ctot_angle_determinant_min"]
                ),
                ctot_angle_determinant_max=_read_quantity(
                    diagnostics_group["ctot_angle_determinant_max"]
                ),
                ctot_angle_determinant_mean=_read_quantity(
                    diagnostics_group["ctot_angle_determinant_mean"]
                ),
                pre_blur_psfs=None if pre_blur is None else pre_blur[...]
            )
        provenance_group = root["provenance"]
        provenance = {
            str(name): _text(value)
            for name, value in provenance_group.attrs.items()
        }
        return HybridResult(
            psfs=result_group["psfs"][...],
            metadata=metadata,
            source_metadata=_read_quantity_mapping(
                result_group["source_metadata"]
            ),
            ngs_metrics=NgsHoMetricPrediction(
                ee=_read_quantity(metrics["ee"]),
                fwhm=_read_quantity(metrics["fwhm"]),
                sr=_read_quantity(metrics["sr"]),
            ),
            ctot_wavefront=_read_quantity(ctot["wavefront"]),
            ctot_angle=_read_quantity(ctot["angle"]),
            angle_to_wavefront_scale=_read_quantity(
                ctot["angle_to_wavefront_scale"]
            ),
            ngs_flux=_read_quantity(ngs["flux"]),
            ngs_magnitude=(
                _read_quantity(ngs["magnitude"]) if "magnitude" in ngs else None
            ),
            ngs_magnitude_zeropoint=(
                _read_quantity(ngs["magnitude_zeropoint"])
                if "magnitude_zeropoint" in ngs else None
            ),
            ngs_frame_rate=_read_quantity(ngs["frame_rate"]),
            jitter=_read_quantity(result_group["jitter"]),
            effective_mastsel_ini=_text(
                root["configuration/effective_mastsel_ini"][()]
            ),
            psd_clipped=ctot["psd_clipped"][...],
            provenance=provenance,
            diagnostics=diagnostics,
        )


def _write_metadata(group: h5py.Group, metadata: PsfMetadata) -> None:
    _write_quantity(group, "wavelength", metadata.wavelength, dtype=np.float64)
    _write_quantity(group, "pixel_scale", metadata.pixel_scale, dtype=np.float64)
    _write_quantity(group, "tel_diameter", metadata.tel_diameter, dtype=np.float64)
    _write_quantity(group, "tel_pupil", metadata.tel_pupil, dtype=np.float32)


def _read_metadata(group: h5py.Group) -> PsfMetadata:
    return PsfMetadata(
        wavelength=_read_quantity(group["wavelength"]),
        pixel_scale=_read_quantity(group["pixel_scale"]),
        tel_diameter=_read_quantity(group["tel_diameter"]),
        tel_pupil=_read_quantity(group["tel_pupil"]),
    )


def _write_quantity(
    group: h5py.Group,
    name: str,
    quantity: u.Quantity,
    *,
    dtype: np.dtype | type,
) -> None:
    dataset = group.create_dataset(name, data=np.asarray(quantity.value), dtype=dtype)
    dataset.attrs["unit"] = quantity.unit.to_string()


def _read_quantity(dataset: h5py.Dataset) -> u.Quantity:
    unit = u.Unit(_text(dataset.attrs["unit"]))
    return np.asarray(dataset[()]) * unit


def _write_quantity_mapping(
    group: h5py.Group, values: dict[str, u.Quantity] | object
) -> None:
    for name, value in values.items():
        _write_quantity(group, str(name), value, dtype=np.float64)


def _read_quantity_mapping(group: h5py.Group) -> dict[str, u.Quantity]:
    return {str(name): _read_quantity(dataset) for name, dataset in group.items()}


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


__all__ = [
    "RESULT_SCHEMA",
    "RESULT_SCHEMA_VERSION",
    "load_result",
    "save_result",
]
