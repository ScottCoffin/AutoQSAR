"""
qsarena.artifacts — interrupt-safe artifact writes.

Every run artifact (manifests, status files, metrics, predictions, caches, reports) is written to a
temporary file in the destination directory and then moved into place with ``os.replace``, which is
atomic on POSIX and on NTFS. A run killed mid-write therefore leaves either the previous complete
file or the new complete file, never a truncated one, and resume can trust whatever it finds.

``set_atomic_writes(False)`` (``caching.atomic_writes: false``) falls back to direct writes; it
exists only for file systems where rename-over is not allowed.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

__all__ = [
    "set_atomic_writes",
    "atomic_writes_enabled",
    "atomic_write_text",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_csv",
    "atomic_write_pickle",
    "supersede_directory",
    "write_artifact_manifest",
]

_ATOMIC = True
#: os.replace can fail transiently on Windows while an antivirus scanner or a viewer holds the target.
_REPLACE_ATTEMPTS = 8
_REPLACE_BACKOFF_SECONDS = 0.25


def set_atomic_writes(enabled: bool) -> None:
    global _ATOMIC
    _ATOMIC = bool(enabled)


def atomic_writes_enabled() -> bool:
    return _ATOMIC


def _replace_with_retry(source: Path, target: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except PermissionError as exc:  # pragma: no cover - Windows file-lock timing
            last_error = exc
            time.sleep(_REPLACE_BACKOFF_SECONDS * (attempt + 1))
    assert last_error is not None
    raise last_error


def _write_via_temp(path: str | Path, writer: Callable[[Path], None]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not _ATOMIC:
        writer(target)
        return target
    temp = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        writer(temp)
        _replace_with_retry(temp, target)
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:  # pragma: no cover - best effort cleanup
                pass
    return target


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    return _write_via_temp(path, lambda p: p.write_text(text, encoding=encoding, newline="\n"))


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    return _write_via_temp(path, lambda p: p.write_bytes(data))


def atomic_write_json(path: str | Path, payload: Any, *, indent: int = 2, default: Any = str) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=indent, default=default) + "\n")


def atomic_write_csv(path: str | Path, frame: Any, **to_csv_kwargs: Any) -> Path:
    """Write a pandas DataFrame. ``index=False`` unless the caller says otherwise."""
    to_csv_kwargs.setdefault("index", False)
    return _write_via_temp(path, lambda p: frame.to_csv(p, **to_csv_kwargs))


def atomic_write_pickle(path: str | Path, obj: Any) -> Path:
    def writer(p: Path) -> None:
        with p.open("wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return _write_via_temp(path, writer)


def supersede_directory(path: str | Path, *, stamp: str | None = None) -> Path | None:
    """Rename an existing non-empty directory to ``<name>_superseded_<stamp>`` and return the new
    path (``None`` if there was nothing to move). ``--fresh`` uses this so earlier results are never
    deleted."""
    source = Path(path)
    if not source.exists() or not any(source.iterdir()):
        return None
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    destination = source.with_name(f"{source.name}_superseded_{stamp}")
    suffix = 1
    while destination.exists():
        suffix += 1
        destination = source.with_name(f"{source.name}_superseded_{stamp}_{suffix}")
    try:
        _replace_with_retry(source, destination)
    except OSError:
        shutil.move(str(source), str(destination))
    return destination


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def merge_json(path: str | Path, updates: Mapping[str, Any]) -> Path:
    payload = read_json(path, default={}) or {}
    if not isinstance(payload, dict):
        payload = {}
    payload.update(dict(updates))
    return atomic_write_json(path, payload)


#: Files that change after the manifest is written (or are the manifest itself).
_MANIFEST_EXCLUDE = {"artifact_manifest.csv", "run.log", "events.jsonl"}


def write_artifact_manifest(output_dir: str | Path) -> Path:
    """``artifact_manifest.csv``: path, size and SHA-256 of every file in a run directory, so a
    copy of the run can be checked byte for byte (``--fresh`` backups and temp files excluded)."""
    root = Path(output_dir)
    rows = ["path,bytes,sha256"]
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in _MANIFEST_EXCLUDE or path.name.endswith(".tmp"):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        rows.append(f"{relative},{path.stat().st_size},{digest.hexdigest()}")
    return atomic_write_text(root / "artifact_manifest.csv", "\n".join(rows) + "\n")
