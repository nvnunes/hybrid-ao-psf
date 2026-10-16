"""Configured publication without native sources or implicit acquisition."""

from __future__ import annotations

import hashlib
import json
import stat
import tempfile
from pathlib import Path

from ._calibration_products import json_metadata, publish_manifest


def publish_result(result, path: Path, save_model) -> None:
    """Stage one product and commit its model/diagnostics with bounded rollback.

    Serialize large payloads once, then move them on the same filesystem. The
    model-specific saver owns schema validation. The common manifest publisher
    commits model.json last and restores previous metadata on cancellation.
    Retained measurements are outside this operation and are never replaced.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = json_metadata(result.diagnostics)
    json.dumps(diagnostics, allow_nan=False)
    with tempfile.TemporaryDirectory(prefix=".build-", dir=path.parent) as temporary:
        temporary = Path(temporary)
        staged = temporary / "model.json"
        save_model(result.model, staged)
        header = json.loads(staged.read_text())
        if "payload" in header:
            source, destination = (
                temporary / header["payload"],
                path.parent / header["payload"],
            )
            if destination.exists():
                with destination.open("rb") as stream:
                    unchanged = (
                        hashlib.file_digest(stream, "sha256").hexdigest()
                        == header["payload_sha256"]
                    )
                if not unchanged:
                    source.chmod(stat.S_IMODE(destination.stat().st_mode))
                    source.replace(destination)
            else:
                source.replace(destination)
        publish_manifest(
            path,
            header,
            diagnostics={
                "model_sha256": hashlib.sha256(staged.read_bytes()).hexdigest(),
                "diagnostics": diagnostics,
            },
        )
