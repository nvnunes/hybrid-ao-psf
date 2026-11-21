"""Bounded model saving and strict scientific-product envelopes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from collections.abc import Mapping
from itertools import count
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

import numpy as np

from ._calibration_io import _digest
from ._io import atomic_output_path

_PRODUCT_VERSIONS = {"hybrid-ao-psf-sensing-calibration": 3}

_ARRAY_FILE_STEMS = {
    "hybrid-ao-psf-covariance-functions": "covariance-interpolators",
    "hybrid-ao-psf-sensing-calibration": "sensing-calibration-interpolators",
    "hybrid-ao-psf-reconstructor-noise": "reconstructor-noise-interpolator",
}


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


def array_file_destination(path: Path, kind: str, digest: str) -> tuple[Path, bool]:
    """Choose a descriptive array file path and report whether it can be reused.

    An identical current array file can be reused after verifying its bytes.
    Otherwise allocate a new name so the previous manifest stays usable until
    saving completes. Serialized writers leave existing revisions intact.
    """
    stem = _ARRAY_FILE_STEMS[kind]
    try:
        current = json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        current = {}
    if isinstance(current, dict) and current.get("array_file_sha256") == digest:
        name = current.get("array_file")
        pattern = rf"{re.escape(stem)}(?:-[0-9]+)?\.npz"
        if isinstance(name, str) and re.fullmatch(pattern, name):
            array_file = path.with_name(name)
            if (
                array_file.is_file()
                and not array_file.is_symlink()
                and _digest(array_file) == digest
            ):
                return array_file, True

    for revision in count(1):
        name = f"{stem}.npz" if revision == 1 else f"{stem}-{revision}.npz"
        array_file = path.with_name(name)
        if not array_file.exists() and not array_file.is_symlink():
            return array_file, False


def save_product(path, kind, metadata, arrays=None, *, diagnostics=None):
    """Save one model.json after its complete numeric array file is durable.

    Separate array file revisions keep the previously accepted manifest usable if
    writing or replacement is interrupted. Writers must be serialized. Old
    array files remain available for bound readers; measurements are not replaced.
    Both outputs use the existing atomic writer and file-mode convention.
    """
    path = Path(path).expanduser().resolve()
    if path.name != "model.json":
        raise ValueError("calibration product path must name model.json")
    header = {
        "schema": kind,
        "version": _PRODUCT_VERSIONS.get(kind, 2),
        **json_metadata(metadata),
    }
    if set(metadata) & {"schema", "version", "array_file", "array_file_sha256"}:
        raise ValueError("product metadata cannot replace envelope fields")
    diagnostics = json_metadata(diagnostics)
    json.dumps(diagnostics, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arrays is not None:
        # The temporary NPZ is complete before the manifest can refer to it.
        import tempfile

        with tempfile.TemporaryDirectory(
            prefix=".calibration-", dir=path.parent
        ) as work:
            temporary = Path(work) / "array_file.npz"
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            digest = _digest(temporary)
            array_file, reused = array_file_destination(path, kind, digest)
            if not reused:
                with (
                    atomic_output_path(array_file) as target,
                    temporary.open("rb") as source,
                    target.open("wb") as destination,
                ):
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
            header.update(array_file=array_file.name, array_file_sha256=digest)
    save_manifest(path, header, diagnostics=diagnostics)


def save_manifest(path, header, *, diagnostics=None):
    """Replace this product's model/optional diagnostic JSON with bounded rollback.

    Numeric array files must already be complete. The two known metadata files are
    backed up before writing; model.json commits last. Cancellation restores both
    old files, including when raised after rename. If restoration itself fails,
    retain the backups outside temporary cleanup and report their recovery path.
    Writers must be serialized. This is not a general transaction facility.
    """
    path = Path(path)
    model_text = json.dumps(header, indent=2, allow_nan=False) + "\n"
    texts = [(path, model_text)]
    if diagnostics is not None:
        texts.insert(
            0,
            (
                path.parent / "diagnostics.json",
                json.dumps(
                    {
                        "model_sha256": hashlib.sha256(
                            model_text.encode("utf-8")
                        ).hexdigest(),
                        "diagnostics": diagnostics,
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
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
                temporary.write_text(text, encoding="utf-8", newline="\n")
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
    """Read the exact envelope and verify array file checksums before numeric loading."""
    from ._calibration_io import read_calibration_model_files

    path = Path(path).expanduser().resolve()
    model_files = read_calibration_model_files(path)
    header = json.loads(path.read_text())
    versions = (_PRODUCT_VERSIONS[kind],) if kind in _PRODUCT_VERSIONS else (1, 2)
    if (
        header.get("schema") != kind
        or type(header.get("version")) is not int
        or header["version"] not in versions
    ):
        raise ValueError(f"unsupported {kind} product schema")
    if header["version"] >= 2:
        if (
            header.get("precision") not in ("float32", "float64")
            or header.get("grid_precision", header.get("precision"))
            not in ("float32", "float64")
            or "producer_precision" not in header
            or header["producer_precision"] not in (None, "float32", "float64")
        ):
            raise ValueError("invalid calibration precision declaration")
    elif {"precision", "producer_precision", "grid_precision"} & header.keys():
        raise ValueError("version 1 does not declare calibration precision")
    arrays = None
    if "array_file" in header:
        with np.load(path.parent / header["array_file"], allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        if header["version"] >= 2 and any(
            (
                value.dtype != np.dtype(bool)
                if name == "measured_mask"
                else value.dtype.kind not in "iu"
                if name == "source_mapping"
                else value.dtype != np.dtype(header["precision"])
            )
            for name, value in arrays.items()
        ):
            raise ValueError("calibration array file precision disagrees with header")
    model_files.verify()
    return header, arrays


def controller_header(controller):
    """Serialize only the fixed numerical gain, frame delay and cadence."""
    from astropy import units as u

    return {
        "gain": controller.gain,
        "delay": controller.delay,
        "frame_rate_hz": float(controller.frame_rate.to_value(u.Hz)),
    }


def read_controller_header(header, *, dtype=np.float64):
    """Validate the exact common numerical controller declaration."""
    from astropy import units as u

    from ._calibration_contracts import FixedGainIntegrator

    if set(header) != {"gain", "delay", "frame_rate_hz"}:
        raise ValueError("invalid numerical controller fields")
    return FixedGainIntegrator(
        dtype=dtype,
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


def read_photometry_header(header, *, dtype=np.float64):
    """Validate the exact common photometric conversion declaration."""
    from astropy import units as u

    from ._calibration_contracts import CalibrationPhotometry

    if set(header) != {"magnitude_band", "photons_per_frame_zeropoint"}:
        raise ValueError("invalid photometry fields")
    return CalibrationPhotometry(
        dtype=dtype,
        magnitude_band=header["magnitude_band"],
        photons_per_frame_zeropoint=header["photons_per_frame_zeropoint"] * u.photon,
    )
