"""Portable calibration data and calibrated model file checks."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from astropy import units as u

from ._calibration_contracts import (
    ComaResponseMeasurements,
    ComaWavefronts,
    ComaWfsMeasurements,
    CovarianceTelemetry,
    ReconstructorNoiseInputs,
    ReconstructorNoiseMeasurements,
    SensingSpot,
    SensingWfsMeasurements,
)
from ._io import atomic_output_path
from ._validation import validate_psf_dtype

_DATA_SCHEMAS = {
    "covariance-telemetry": (
        CovarianceTelemetry,
        {
            "zenith_angle": "deg",
            "positions": "arcsec",
            "frame_rate": "Hz",
            "tilt": "mas",
            "coma": "nm",
        },
    ),
    "coma-wavefronts": (ComaWavefronts, {"wavefront": "nm", "coma": "nm"}),
    "coma-wfs": (ComaWfsMeasurements, {"measured": "mas", "noiseless": "mas"}),
    "coma-response": (
        ComaResponseMeasurements,
        {
            "coma": "nm",
            "magnitudes": "mag",
            "tilt_mean": "mas",
            "tilt_batches": "mas",
            "tilt_noiseless": "mas",
        },
    ),
    "sensing-spot": (
        SensingSpot,
        {
            "pixel_scale": "mas",
            "wavelength": "m",
            "photons_per_frame": "ph",
            "displacements": "mas",
        },
    ),
    "sensing-wfs": (SensingWfsMeasurements, {"zero": "mas", "shifted": "mas"}),
    "reconstructor-inputs": (
        ReconstructorNoiseInputs,
        {"photons_per_frame": "ph", "ee": "", "fwhm": "mas"},
    ),
    "reconstructor-noise": (ReconstructorNoiseMeasurements, {"covariance": "mas2"}),
}


def save_calibration_data(
    data: CovarianceTelemetry
    | ComaWavefronts
    | ComaWfsMeasurements
    | ComaResponseMeasurements
    | SensingSpot
    | SensingWfsMeasurements
    | ReconstructorNoiseInputs
    | ReconstructorNoiseMeasurements,
    path: str | Path,
) -> None:
    """Atomically save typed calibration data to numeric NPZ.

    The NPZ contains arrays plus a strict versioned JSON header with physical
    units and record kind. It contains no pickle/native setup/source identities.
    It is reusable input, not a fitted model. Existing destinations survive
    failed/interrupted writes and retain their permission bits.

    Args:
        data: Typed calibration inputs or measurements.
        path: Destination NPZ file. Create its parent directories if
            needed. Atomically replace any existing destination.

    Raises:
        TypeError: If the input is not the expected calibration record.
    """
    for kind, (cls, units) in _DATA_SCHEMAS.items():
        if type(data) is cls:
            break
    else:
        raise TypeError("unsupported calibration data type")
    header = {
        "schema": "hybrid-ao-psf-calibration-samples",
        "version": 2 if hasattr(data, "dtype") else 1,
        "kind": kind,
        "units": units,
    }
    if hasattr(data, "dtype"):
        header["precision"] = data.dtype.name
    arrays = {
        item.name: getattr(data, item.name)
        for item in fields(data)
        if item.name != "dtype"
    }
    arrays = {
        name: value.value if isinstance(value, u.Quantity) else value
        for name, value in arrays.items()
    }
    with atomic_output_path(Path(path)) as temporary, temporary.open("wb") as stream:
        np.savez(stream, __header__=json.dumps(header, sort_keys=True), **arrays)


def validate_calibration_data_precision(header, saved, *, history_fields=()):
    """Reject missing/mismatched v2 declarations before coercing constructors.

    Simulation history fields retain their actual captured precision independently
    of calculation precision, including legacy histories rewritten as v2.
    """
    if not isinstance(header, dict):
        raise ValueError("invalid calibration-data header")  # noqa: TRY004 -- persisted schema, not a call argument.
    if header.get("version") != 2:
        return
    if header.get("precision") not in ("float32", "float64"):
        raise ValueError("version-2 calibration data require precision float32 or float64")
    dtype = validate_psf_dtype(header["precision"])
    for name in saved.files:
        if name == "__header__":
            continue
        if name == "pupil" and saved[name].dtype == np.dtype(bool):
            continue
        if name in history_fields:
            validate_psf_dtype(saved[name].dtype)
        elif saved[name].dtype != dtype:
            raise ValueError(f"calibration-data precision mismatch at {name}")


def load_calibration_data(
    path: str | Path,
) -> (
    CovarianceTelemetry
    | ComaWavefronts
    | ComaWfsMeasurements
    | ComaResponseMeasurements
    | SensingSpot
    | SensingWfsMeasurements
    | ReconstructorNoiseInputs
    | ReconstructorNoiseMeasurements
):
    """Load exactly a supported calibration-data schema, validating units, axes and values.

    Native and historical measurement formats are rejected. The downstream
    one-time importer owns conversion from accepted older local records.
    Returns the corresponding immutable calibration-data type.

    Args:
        path: Path to the saved NPZ file.

    Returns:
        The typed calibration data reconstructed at its stored precision.

    Raises:
        FileNotFoundError: If a required file is missing.
        ValueError: If its schema or numerical contents are invalid.
    """
    with np.load(Path(path), allow_pickle=False) as saved:
        if "__header__" not in saved.files or saved["__header__"].ndim != 0:
            raise ValueError("missing scalar calibration-data header")
        header = json.loads(str(saved["__header__"]))
        if (
            not isinstance(header, dict)
            or set(header)
            != {"schema", "version", "kind", "units"}
            | ({"precision"} if header.get("version") == 2 else set())
            or header["schema"] != "hybrid-ao-psf-calibration-samples"
            or type(header["version"]) is not int
            or header["version"] not in (1, 2)
            or header["kind"] not in _DATA_SCHEMAS
        ):
            raise ValueError("unsupported calibration-data schema")
        validate_calibration_data_precision(header, saved)
        cls, units = _DATA_SCHEMAS[header["kind"]]
        if header["units"] != units or set(saved.files) != {
            "__header__",
            *(item.name for item in fields(cls) if item.name != "dtype"),
        }:
            raise ValueError(
                "calibration data fields/units do not match their schema"
            )
        values = {
            item.name: saved[item.name] for item in fields(cls) if item.name != "dtype"
        }
        if any(item.name == "dtype" for item in fields(cls)):
            values["dtype"] = header.get("precision", np.float64)
        for name, unit in units.items():
            values[name] = values[name] * u.Unit(unit)
        return cls(**values)


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class CalibrationFile:
    """One resolved calibration file and its SHA-256 checksum.

    Attributes:
        path: Absolute manifest, array file or NGS metric file path.
        sha256: Lowercase 64-character SHA-256 digest of accepted bytes.
    """

    path: Path
    sha256: str

    def __post_init__(self):
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())
        if not isinstance(self.sha256, str) or not re.fullmatch(
            "[0-9a-f]{64}", self.sha256
        ):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")

    def verify(self) -> None:
        """Reject missing or changed file bytes, without loading native/model code."""
        if _digest(self.path) != self.sha256:
            raise ValueError(f"Calibration file content changed: {self.path}")


@dataclass(frozen=True)
class CalibrationModelFiles:
    """Manifest and array files for one calibrated model, with their checksums.

    Attributes:
        manifest: Path and checksum of the model.json manifest.
        array_files: Paths and checksums of numerical files, empty for JSON-only.

    Runtime loaders validate each scientific schema separately. Orchestration
    can retain these paths and checksums and verify the files before execution,
    without knowing private model schema fields or constructing native adapters.
    """

    manifest: CalibrationFile
    array_files: tuple[CalibrationFile, ...] = ()

    def __post_init__(self):
        if (
            not isinstance(self.manifest, CalibrationFile)
            or self.manifest.path.name != "model.json"
        ):
            raise ValueError("manifest must identify model.json")
        values = tuple(self.array_files)
        if any(not isinstance(value, CalibrationFile) for value in values):
            raise TypeError("array_files must contain CalibrationFile values")
        paths = [self.manifest.path, *(value.path for value in values)]
        if len(set(paths)) != len(paths):
            raise ValueError("model file paths must be distinct")
        object.__setattr__(self, "array_files", values)

    def verify(self) -> None:
        """Verify the manifest and array files before loading for execution."""
        self.manifest.verify()
        for array_file in self.array_files:
            array_file.verify()


def read_calibration_model_files(path: str | Path) -> CalibrationModelFiles:
    """Read a model manifest and verify its referenced array file.

    Shared fitted schemas use array_file and array_file_sha256 together, or
    neither for JSON-only products. This checks file contents, not the scientific
    schema, coefficients, units or ranges; each fitted loader validates those.
    Changed manifest bytes during the read or mismatched array file digests raise.

    Args:
        path: Path to the calibrated model's model.json manifest.

    Returns:
        The resolved manifest and optional array file paths with their
        SHA-256 digests.

    Raises:
        FileNotFoundError: If the manifest or array file is missing.
        TypeError: If the manifest does not contain an object.
        ValueError: If the path, manifest fields or checksum is invalid.
    """
    path = Path(path).expanduser().resolve()
    if path.name != "model.json":
        raise ValueError("model manifest must be model.json")
    raw = path.read_bytes()
    header = json.loads(raw)
    if not isinstance(header, dict):
        raise TypeError("model.json must contain an object")
    if {"payload", "payload_sha256"} & header.keys():
        raise ValueError("Model manifest uses unsupported array file fields")
    array_files = ()
    if ("array_file" in header) != ("array_file_sha256" in header):
        raise ValueError("array_file and array_file_sha256 must appear together")
    if "array_file" in header:
        if not isinstance(header["array_file"], str) or not header["array_file"]:
            raise ValueError("array file must be a nonempty relative NPZ path")
        relative = Path(header["array_file"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix != ".npz"
        ):
            raise ValueError("array file must be a relative NPZ path within the model")
        array_file = (path.parent / relative).resolve()
        if not array_file.is_relative_to(path.parent):
            raise ValueError("array file must remain within the model directory")
        array_files = (CalibrationFile(array_file, header["array_file_sha256"]),)
    model_files = CalibrationModelFiles(
        CalibrationFile(path, hashlib.sha256(raw).hexdigest()), array_files
    )
    model_files.verify()
    return model_files
