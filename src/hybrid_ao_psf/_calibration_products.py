"""Bounded model publication and strict scientific-product envelopes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

import numpy as np

from ._io import atomic_output_path


def immutable_metadata(value):
    """Copy finite JSON diagnostics into recursively immutable containers."""
    value = json.loads(json.dumps(json_metadata(value), allow_nan=False))

    def freeze(item):
        if isinstance(item, dict):
            return MappingProxyType({key: freeze(part) for key, part in item.items()})
        if isinstance(item, list):
            return tuple(freeze(part) for part in item)
        return item

    return freeze(value)


def json_metadata(value):
    """Convert immutable numerical diagnostics back to ordinary JSON values."""
    if isinstance(value, Mapping):
        return {key: json_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_metadata(item) for item in value]
    return value


def save_product(path, kind, metadata, arrays=None):
    """Publish one model.json after its complete numeric payload is durable.

    Content-named payloads keep the previously accepted manifest usable if
    writing or replacement is interrupted. Writers must be serialized. Old
    payloads remain available for bound readers; measurements are not replaced.
    Both outputs use the existing atomic writer and file-mode convention.
    """
    path = Path(path).expanduser().resolve()
    if path.name != "model.json":
        raise ValueError("calibration product path must name model.json")
    header = {"schema": kind, "version": 1, **json_metadata(metadata)}
    if set(metadata) & {"schema", "version", "payload", "payload_sha256"}:
        raise ValueError("product metadata cannot replace envelope fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    if arrays is not None:
        # The temporary NPZ is complete before the manifest can refer to it.
        import tempfile

        with tempfile.TemporaryDirectory(
            prefix=".calibration-", dir=path.parent
        ) as work:
            temporary = Path(work) / "payload.npz"
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            with temporary.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            payload = path.parent / f"interpolators-{digest}.npz"
            from ._calibration_io import _digest

            if not payload.exists() or _digest(payload) != digest:
                with (
                    atomic_output_path(payload) as target,
                    temporary.open("rb") as source,
                    target.open("wb") as destination,
                ):
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
            header.update(payload=payload.name, payload_sha256=digest)
    publish_manifest(path, header)


def publish_manifest(path, header, *, diagnostics=None):
    """Replace this product's model/optional diagnostic JSON with bounded rollback.

    Numeric payloads must already be complete. The two known metadata files are
    backed up before writing; model.json commits last. Cancellation restores both
    old files, including when raised after rename. If restoration itself fails,
    retain the backups outside temporary cleanup and report their recovery path.
    Writers must be serialized. This is not a general transaction facility.
    """
    path = Path(path)
    texts = [(path, json.dumps(header, indent=2, allow_nan=False) + "\n")]
    if diagnostics is not None:
        texts.insert(
            0,
            (
                path.parent / "diagnostics.json",
                json.dumps(diagnostics, indent=2, allow_nan=False) + "\n",
            ),
        )
    previous = {}
    try:
        for target, _ in texts:
            if target.exists():
                backup = target.with_name(f".{target.stem}-previous-{uuid4().hex}.json")
                with atomic_output_path(backup) as temporary:
                    temporary.write_bytes(target.read_bytes())
                backup.chmod(stat.S_IMODE(target.stat().st_mode))
                previous[target] = backup
            else:
                previous[target] = None
        for target, text in texts:
            with atomic_output_path(target) as temporary:
                temporary.write_text(text)
    except BaseException as error:
        recovery = []
        for target, backup in previous.items():
            try:
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    os.replace(backup, target)
            except (OSError, KeyboardInterrupt) as recovery_error:
                recovery.append((backup, recovery_error))
        if recovery:
            error.add_note(
                "Metadata restoration failed; recovery files: "
                + ", ".join(str(backup) for backup, _ in recovery)
            )
            raise error from recovery[0][1]
        raise
    else:
        for backup in previous.values():
            if backup is not None:
                backup.unlink(missing_ok=True)


def load_product(path, kind):
    """Read the exact envelope and verify payload identity before numeric loading."""
    from ._calibration_io import read_calibration_artifact_identity

    path = Path(path).expanduser().resolve()
    identity = read_calibration_artifact_identity(path)
    header = json.loads(path.read_text())
    if (
        header.get("schema") != kind
        or type(header.get("version")) is not int
        or header["version"] != 1
    ):
        raise ValueError(f"unsupported {kind} product schema")
    arrays = None
    if "payload" in header:
        with np.load(path.parent / header["payload"], allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
    identity.verify()
    return header, arrays


def controller_header(controller):
    """Serialize only the fixed numerical gain, frame delay and cadence."""
    from astropy import units as u

    return {
        "gain": controller.gain,
        "delay": controller.delay,
        "frame_rate_hz": float(controller.frame_rate.to_value(u.Hz)),
    }


def read_controller_header(header):
    """Validate the exact common numerical controller declaration."""
    from astropy import units as u

    from ._calibration_contracts import FixedGainIntegrator

    if set(header) != {"gain", "delay", "frame_rate_hz"}:
        raise ValueError("invalid numerical controller fields")
    return FixedGainIntegrator(
        gain=header["gain"],
        delay=header["delay"],
        frame_rate=header["frame_rate_hz"] * u.Hz,
    )


def photometry_header(photometry):
    """Serialize the magnitude convention and incident photons/frame zero point."""
    return {
        "magnitude_band": photometry.magnitude_band,
        "photons_per_frame_zeropoint": float(
            photometry.photons_per_frame_zeropoint.value
        ),
    }


def read_photometry_header(header):
    """Validate the exact common photometric conversion declaration."""
    from astropy import units as u

    from ._calibration_contracts import CalibrationPhotometry

    if set(header) != {"magnitude_band", "photons_per_frame_zeropoint"}:
        raise ValueError("invalid photometry fields")
    return CalibrationPhotometry(
        magnitude_band=header["magnitude_band"],
        photons_per_frame_zeropoint=header["photons_per_frame_zeropoint"] * u.photon,
    )
