"""Save configured calibration models and diagnostics."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ._calibration_products import array_file_destination, json_metadata, save_manifest


def save_result(result, path: Path, save_model) -> Path:
    """Stage one product and save its model and diagnostics with bounded rollback.

    Serialize large array files once, then move them on the same filesystem. The
    model-specific saver owns schema validation. The common manifest writer
    writes model.json last and restores previous metadata on cancellation.
    Retained measurements are outside this operation and are never replaced.
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = json_metadata(result.diagnostics)
    json.dumps(diagnostics, allow_nan=False)
    with tempfile.TemporaryDirectory(prefix=".save-", dir=path.parent) as temporary:
        temporary = Path(temporary)
        staged = temporary / "model.json"
        save_model(result.model, staged)
        header = json.loads(staged.read_text())
        if "array_file" in header:
            source = temporary / header["array_file"]
            destination, reused = array_file_destination(
                path, header["schema"], header["array_file_sha256"]
            )
            if not reused:
                source.replace(destination)
            header["array_file"] = destination.name
        save_manifest(path, header, diagnostics=diagnostics)
    return path
