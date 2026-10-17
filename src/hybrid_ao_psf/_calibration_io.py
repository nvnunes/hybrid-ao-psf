"""Portable calibration observations and fitted-product content identities."""

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

_SAMPLE_SCHEMAS = {
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


def save_calibration_samples(
    samples: CovarianceTelemetry
    | ComaWavefronts
    | ComaWfsMeasurements
    | ComaResponseMeasurements
    | SensingSpot
    | SensingWfsMeasurements
    | ReconstructorNoiseInputs
    | ReconstructorNoiseMeasurements,
    path: str | Path,
) -> None:
    """Atomically save a supported physical observation record to numeric NPZ.

    The NPZ contains arrays plus a strict versioned JSON header with physical
    units and record kind. It contains no pickle/native setup/source identities.
    It is reusable input, not a fitted model. Existing destinations survive
    failed/interrupted writes and retain their permission bits.
    """
    for kind, (cls, units) in _SAMPLE_SCHEMAS.items():
        if type(samples) is cls:
            break
    else:
        raise TypeError("unsupported calibration sample record")
    header = {
        "schema": "hybrid-ao-psf-calibration-samples",
        "version": 2 if hasattr(samples, "dtype") else 1,
        "kind": kind,
        "units": units,
    }
    if hasattr(samples, "dtype"):
        header["precision"] = samples.dtype.name
    arrays = {
        item.name: getattr(samples, item.name)
        for item in fields(samples)
        if item.name != "dtype"
    }
    arrays = {
        name: value.value if isinstance(value, u.Quantity) else value
        for name, value in arrays.items()
    }
    with atomic_output_path(Path(path)) as temporary, temporary.open("wb") as stream:
        np.savez(stream, __header__=json.dumps(header, sort_keys=True), **arrays)


def validate_observation_precision(header, saved, *, history_fields=()):
    """Reject missing/mismatched v2 declarations before coercing constructors.

    Replay history fields retain their actual captured precision independently
    of calculation precision, including legacy histories rewritten as v2.
    """
    if not isinstance(header, dict):
        raise ValueError("invalid observation header")  # noqa: TRY004 -- persisted schema, not a call argument.
    if header.get("version") != 2:
        return
    if header.get("precision") not in ("float32", "float64"):
        raise ValueError("version-2 observations require precision float32 or float64")
    dtype = validate_psf_dtype(header["precision"])
    for name in saved.files:
        if name == "__header__":
            continue
        if name == "pupil" and saved[name].dtype == np.dtype(bool):
            continue
        if name in history_fields:
            validate_psf_dtype(saved[name].dtype)
        elif saved[name].dtype != dtype:
            raise ValueError(f"observation precision mismatch at {name}")


def load_calibration_samples(
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
    """Load exactly a supported sample schema, validating units, axes and values.

    Native and historical measurement formats are rejected. The downstream
    one-time importer owns conversion from accepted older local records.
    Returns the corresponding immutable physical observation type.
    """
    with np.load(Path(path), allow_pickle=False) as saved:
        if "__header__" not in saved.files or saved["__header__"].ndim != 0:
            raise ValueError("missing scalar calibration-samples header")
        header = json.loads(str(saved["__header__"]))
        if (
            not isinstance(header, dict)
            or set(header)
            != {"schema", "version", "kind", "units"}
            | ({"precision"} if header.get("version") == 2 else set())
            or header["schema"] != "hybrid-ao-psf-calibration-samples"
            or type(header["version"]) is not int
            or header["version"] not in (1, 2)
            or header["kind"] not in _SAMPLE_SCHEMAS
        ):
            raise ValueError("unsupported calibration-samples schema")
        validate_observation_precision(header, saved)
        cls, units = _SAMPLE_SCHEMAS[header["kind"]]
        if header["units"] != units or set(saved.files) != {
            "__header__",
            *(item.name for item in fields(cls) if item.name != "dtype"),
        }:
            raise ValueError(
                "calibration sample fields/units do not match their schema"
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
class CalibrationFileIdentity:
    """One resolved product file and its SHA-256 content identity.

    Attributes:
        path: Absolute model entrypoint or numeric payload path.
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
            raise ValueError(f"Calibration product content changed: {self.path}")


@dataclass(frozen=True)
class CalibrationArtifactIdentity:
    """Immutable runtime content binding for one fitted product.

    Attributes:
        entrypoint: Identity of the explicit model.json entrypoint.
        payloads: Identities of referenced numerical files, empty for JSON-only.

    Runtime loaders validate each scientific schema separately. Orchestration
    can persist these supported identities and verify them before execution,
    without knowing private model schema fields or constructing native adapters.
    """

    entrypoint: CalibrationFileIdentity
    payloads: tuple[CalibrationFileIdentity, ...] = ()

    def __post_init__(self):
        if (
            not isinstance(self.entrypoint, CalibrationFileIdentity)
            or self.entrypoint.path.name != "model.json"
        ):
            raise ValueError("entrypoint must identify model.json")
        values = tuple(self.payloads)
        if any(not isinstance(value, CalibrationFileIdentity) for value in values):
            raise TypeError("payloads must contain CalibrationFileIdentity values")
        paths = [self.entrypoint.path, *(value.path for value in values)]
        if len(set(paths)) != len(paths):
            raise ValueError("artifact file identities must be distinct")
        object.__setattr__(self, "payloads", values)

    def verify(self) -> None:
        """Verify all bound files, including payloads, before loading for execution."""
        self.entrypoint.verify()
        for identity in self.payloads:
            identity.verify()


def read_calibration_artifact_identity(path: str | Path) -> CalibrationArtifactIdentity:
    """Read model.json identity and verify its optional relative NPZ payload.

    Shared fitted schemas use payload and payload_sha256 together, or neither
    for JSON-only products. This checks content binding, not the scientific
    schema, coefficients, units or support; each fitted loader validates those.
    Changed manifest bytes during the read or mismatched payload digests raise.
    """
    path = Path(path).expanduser().resolve()
    if path.name != "model.json":
        raise ValueError("fitted product entrypoint must be model.json")
    raw = path.read_bytes()
    header = json.loads(raw)
    if not isinstance(header, dict):
        raise TypeError("model.json must contain an object")
    payloads = ()
    if ("payload" in header) != ("payload_sha256" in header):
        raise ValueError("payload and payload_sha256 must appear together")
    if "payload" in header:
        if not isinstance(header["payload"], str) or not header["payload"]:
            raise ValueError("payload must be a nonempty relative NPZ path")
        relative = Path(header["payload"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix != ".npz"
        ):
            raise ValueError("payload must be a relative NPZ path within the product")
        payload = (path.parent / relative).resolve()
        if not payload.is_relative_to(path.parent):
            raise ValueError("payload must remain within the product directory")
        payloads = (CalibrationFileIdentity(payload, header["payload_sha256"]),)
    identity = CalibrationArtifactIdentity(
        CalibrationFileIdentity(path, hashlib.sha256(raw).hexdigest()), payloads
    )
    identity.verify()
    return identity
