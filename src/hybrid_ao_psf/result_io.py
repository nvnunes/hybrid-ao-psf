"""Versioned HDF5 persistence for standalone Hybrid AO PSF results."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import h5py
import numpy as np
from ao_stats import PsfMetadata
from astropy import units as u

from ._e2e_model import E2ELowOrderPrediction
from ._io import atomic_output_path
from .contracts import (
    HybridDiagnostics,
    HybridResult,
    LowOrderDistribution,
    LowOrderModel,
    NgsHoMetricPrediction,
)

RESULT_SCHEMA = "hybrid-ao-psf-result"
RESULT_SCHEMA_VERSION = 3


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
        root.attrs["lo_model"] = result.lo_model.value
        root.attrs["precision"] = "mixed" if result.dtype is None else result.dtype.name

        result_group = root.create_group("result")
        result_group.create_dataset("psfs", data=result.psfs)
        _write_metadata(result_group.create_group("metadata"), result.metadata)
        _write_quantity_mapping(
            result_group.create_group("source_metadata"), result.source_metadata
        )
        metrics = result_group.create_group("ngs_metrics")
        _write_quantity(metrics, "ee", result.ngs_metrics.ee)
        _write_quantity(metrics, "fwhm", result.ngs_metrics.fwhm)
        _write_quantity(metrics, "sr", result.ngs_metrics.sr)
        ctot = result_group.create_group("ctot")
        if result.ctot_wavefront is not None:
            _write_quantity(ctot, "wavefront", result.ctot_wavefront)
        _write_quantity(ctot, "angle", result.ctot_angle)
        if result.angle_to_wavefront_scale is not None:
            _write_quantity(
                ctot, "angle_to_wavefront_scale", result.angle_to_wavefront_scale
            )
        ctot.create_dataset("psd_clipped", data=result.psd_clipped, dtype=bool)
        ngs = result_group.create_group("ngs")
        if result.ngs_flux is not None:
            _write_quantity(ngs, "flux", result.ngs_flux)
        if result.ngs_magnitude is not None:
            _write_quantity(ngs, "magnitude", result.ngs_magnitude)
            if result.ngs_magnitude_zeropoint is not None:
                _write_quantity(
                    ngs, "magnitude_zeropoint", result.ngs_magnitude_zeropoint
                )
        _write_quantity(ngs, "frame_rate", result.ngs_frame_rate)
        _write_quantity(result_group, "jitter", result.jitter)

        configuration = root.create_group("configuration")
        if result.effective_mastsel_ini is not None:
            configuration.create_dataset(
                "effective_mastsel_ini",
                data=result.effective_mastsel_ini,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
        inputs = result_group.create_group("coordinates")
        for name in ("science_positions", "ngs_positions", "zenith_angle"):
            if getattr(result, name) is not None:
                _write_quantity(inputs, name, getattr(result, name))
        distribution = result_group.create_group("lo_distribution")
        distribution.create_dataset("weights", data=result.lo_distribution.weights)
        _write_quantity(distribution, "covariances", result.lo_distribution.covariances)
        if result.e2e_prediction is not None:
            prediction = result_group.create_group("e2e_prediction")
            for item in fields(result.e2e_prediction):
                if item.name in ("dtype", "metrics"):
                    continue
                value = getattr(result.e2e_prediction, item.name)
                if isinstance(value, u.Quantity):
                    _write_quantity(prediction, item.name, value)
                elif isinstance(value, np.ndarray):
                    prediction.create_dataset(item.name, data=value)
                else:
                    prediction.attrs[item.name] = value
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
            )
            _write_quantity(
                diagnostics,
                "ctot_angle_trace_max",
                result.diagnostics.ctot_angle_trace_max,
            )
            _write_quantity(
                diagnostics,
                "ctot_angle_trace_mean",
                result.diagnostics.ctot_angle_trace_mean,
            )
            _write_quantity(
                diagnostics,
                "ctot_angle_determinant_min",
                result.diagnostics.ctot_angle_determinant_min,
            )
            _write_quantity(
                diagnostics,
                "ctot_angle_determinant_max",
                result.diagnostics.ctot_angle_determinant_max,
            )
            _write_quantity(
                diagnostics,
                "ctot_angle_determinant_mean",
                result.diagnostics.ctot_angle_determinant_mean,
            )
            if result.diagnostics.pre_blur_psfs is not None:
                diagnostics.create_dataset(
                    "pre_blur_psfs",
                    data=result.diagnostics.pre_blur_psfs,
                )


def load_result(path: str | Path) -> HybridResult:
    """Load self-contained version-3 results or MASTSEL versions 1 and 2.

    Version 3 requires explicit model identity, applied distribution and matching
    model-specific fields. No provider/native files are opened. Fitted E2E
    numerical state is validated by its scientific owner. Versions 1 and 2 are
    MASTSEL-only; unlabelled/mixed results retain each stored array precision
    with dtype=None. Uniform declarations must match all floating datasets.
    Unsupported schemas, precision, shapes or covariance relationships raise.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Hybrid AO PSF result not found: {source}")
    with h5py.File(source, "r") as root:
        schema = _text(root.attrs.get("schema", ""))
        version = int(root.attrs.get("schema_version", -1))
        if schema != RESULT_SCHEMA or version not in (1, 2, RESULT_SCHEMA_VERSION):
            raise ValueError(
                "Unsupported Hybrid AO PSF result schema: "
                f"schema={schema!r}, version={version}."
            )
        precision = _text(root.attrs.get("precision", "")) if version >= 2 else "mixed"
        if precision not in ("mixed", "float32", "float64"):
            raise ValueError(
                "Result precision must be float32, float64 or legacy mixed."
            )
        dtype = None if precision == "mixed" else np.dtype(precision)
        if dtype is not None:

            def check_precision(name, item):
                if (
                    isinstance(item, h5py.Dataset)
                    and item.dtype.kind in "fc"
                    and item.dtype != dtype
                ):
                    raise ValueError(
                        f"Stored precision mismatch at {name}: expected {dtype}."
                    )

            root.visititems(check_precision)
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
                dtype=dtype,
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
                pre_blur_psfs=None if pre_blur is None else pre_blur[...],
            )
        provenance_group = root["provenance"]
        provenance = {
            str(name): _text(value) for name, value in provenance_group.attrs.items()
        }
        if version >= 3:
            if "lo_model" not in root.attrs:
                raise ValueError("version-3 result requires explicit lo_model")
            lo_model = LowOrderModel(_text(root.attrs["lo_model"]))
            distribution_group = result_group["lo_distribution"]
            distribution = LowOrderDistribution(
                weights=distribution_group["weights"][...],
                covariances=_read_quantity(distribution_group["covariances"]),
                dtype=distribution_group["weights"].dtype,
            )
        else:
            lo_model = LowOrderModel.MASTSEL
            distribution = None
        metric_values = NgsHoMetricPrediction(
            dtype=dtype,
            ee=_read_quantity(metrics["ee"]),
            fwhm=_read_quantity(metrics["fwhm"]),
            sr=_read_quantity(metrics["sr"]),
        )
        prediction = None
        if version >= 3 and "e2e_prediction" in result_group:
            group = result_group["e2e_prediction"]
            values = {
                name: (_read_quantity(item) if "unit" in item.attrs else item[...])
                for name, item in group.items()
            }
            if values.keys() & group.attrs.keys():
                raise ValueError("duplicate E2E prediction fields")
            values.update(dict(group.attrs))
            prediction = E2ELowOrderPrediction._create(
                dtype, metrics=metric_values, **values
            )
        coordinates = {
            name: (
                _read_quantity(result_group["coordinates"][name])
                if version >= 3 and name in result_group["coordinates"]
                else None
            )
            for name in ("science_positions", "ngs_positions", "zenith_angle")
        }
        return HybridResult(
            dtype=dtype,
            psfs=result_group["psfs"][...],
            metadata=metadata,
            source_metadata=_read_quantity_mapping(result_group["source_metadata"]),
            ngs_metrics=metric_values,
            ctot_wavefront=_read_quantity(ctot["wavefront"])
            if "wavefront" in ctot
            else None,
            ctot_angle=_read_quantity(ctot["angle"]),
            angle_to_wavefront_scale=_read_quantity(ctot["angle_to_wavefront_scale"])
            if "angle_to_wavefront_scale" in ctot
            else None,
            ngs_flux=_read_quantity(ngs["flux"]) if "flux" in ngs else None,
            ngs_magnitude=(
                _read_quantity(ngs["magnitude"]) if "magnitude" in ngs else None
            ),
            ngs_magnitude_zeropoint=(
                _read_quantity(ngs["magnitude_zeropoint"])
                if "magnitude_zeropoint" in ngs
                else None
            ),
            ngs_frame_rate=_read_quantity(ngs["frame_rate"]),
            jitter=_read_quantity(result_group["jitter"]),
            effective_mastsel_ini=(
                _text(root["configuration/effective_mastsel_ini"][()])
                if "effective_mastsel_ini" in root["configuration"]
                else None
            ),
            psd_clipped=ctot["psd_clipped"][...],
            provenance=provenance,
            diagnostics=diagnostics,
            lo_model=lo_model,
            lo_distribution=distribution,
            e2e_prediction=prediction,
            **coordinates,
        )


def _write_metadata(group: h5py.Group, metadata: PsfMetadata) -> None:
    _write_quantity(group, "wavelength", metadata.wavelength)
    _write_quantity(group, "pixel_scale", metadata.pixel_scale)
    _write_quantity(group, "tel_diameter", metadata.tel_diameter)
    _write_quantity(group, "tel_pupil", metadata.tel_pupil)


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
    dtype: np.dtype | type | None = None,
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
        _write_quantity(group, str(name), value)


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
