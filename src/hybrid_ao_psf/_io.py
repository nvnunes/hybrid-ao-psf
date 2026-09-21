"""Package-private atomic output support."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


@contextmanager
def atomic_output_path(path: Path) -> Iterator[Path]:
    """Yield a sibling temporary path and atomically replace ``path`` on success.

    New files use ordinary creation permissions subject to the process umask.
    Replacements retain the destination's existing permission bits.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    temporary_path = _create_sibling(path, mode=existing_mode)
    try:
        yield temporary_path
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _create_sibling(path: Path, *, mode: int | None) -> Path:
    for _ in range(100):
        candidate = path.parent / f".{path.name}.{uuid4().hex}.tmp"
        try:
            descriptor = os.open(
                candidate,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o666,
            )
        except FileExistsError:
            continue
        try:
            if mode is not None:
                os.fchmod(descriptor, mode)
        finally:
            os.close(descriptor)
        return candidate
    raise FileExistsError(f"Could not allocate a temporary output beside {path}.")


__all__: list[str] = []
