"""
qsarena.provenance — environment manifests and split hashes for every results directory.

``write_environment_manifest(output_dir)`` records what produced a result: the exact versions of
the packages that change model output (RDKit, scikit-learn, the boosting libraries, PyTorch, DGL,
Chemprop, Uni-Mol, ...), a full installed-distribution list (the ``pip freeze`` equivalent), the
git commit and dirty flag of the source checkout, and the host hardware. The benchmark runner and
the reliability study both call it, so a results directory is self-describing.

``split_hash`` is the same order-sensitive SHA-256 over stripped SMILES that the benchmark runner
stores as ``split_train_hash`` / ``split_test_hash`` in ``metrics.csv``; it is duplicated here so
that callers do not have to import the 10k-line runner to check a split.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Iterable

__all__ = [
    "KEY_PACKAGES",
    "split_hash",
    "package_versions",
    "git_state",
    "hardware_summary",
    "environment_manifest",
    "write_environment_manifest",
]

#: Distributions whose version can change a model's numbers. Missing ones are recorded as None.
KEY_PACKAGES: tuple[str, ...] = (
    "qsarena",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "rdkit",
    "xgboost",
    "lightgbm",
    "catboost",
    "torch",
    "tensorflow",
    "dgl",
    "dgllife",
    "chemprop",
    "unimol-tools",
    "tabpfn",
    "PyTDC",
)

# Same contract as _SUBPROCESS_TEXT_KWARGS in the runner: never decode with the ambient locale.
_TEXT_KWARGS = {"text": True, "encoding": "utf-8", "errors": "replace"}


def split_hash(smiles: Iterable[str]) -> str:
    """Order-sensitive SHA-256 of newline-joined, stripped SMILES (the runner's split hash)."""
    payload = "\n".join(str(s).strip() for s in smiles)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def package_versions(names: Iterable[str] = KEY_PACKAGES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _all_distributions() -> list[str]:
    seen: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name and name.lower() not in seen:
            seen[name.lower()] = f"{name}=={dist.version}"
    return sorted(seen.values(), key=str.lower)


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=15, check=False, **_TEXT_KWARGS)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_state(repo_dir: str | Path | None = None) -> dict[str, object]:
    """Commit, branch and dirty flag of the checkout containing ``repo_dir`` (None if not git)."""
    cwd = Path(repo_dir) if repo_dir else Path(__file__).resolve().parents[1]
    commit = _run(["git", "rev-parse", "HEAD"], cwd)
    if commit is None:
        return {"commit": None, "branch": None, "dirty": None}
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd)
    return {
        "commit": commit,
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd),
        "dirty": bool(status) if status is not None else None,
    }


def hardware_summary() -> dict[str, object]:
    gpus = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    return {
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "gpus": [g.strip() for g in gpus.splitlines() if g.strip()] if gpus else [],
    }


def environment_manifest(repo_dir: str | Path | None = None, extra: dict | None = None) -> dict:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "key_packages": package_versions(),
        "installed_distributions": _all_distributions(),
        "git": git_state(repo_dir),
        "hardware": hardware_summary(),
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_environment_manifest(
    output_dir: str | Path,
    repo_dir: str | Path | None = None,
    extra: dict | None = None,
    filename: str = "environment_manifest.json",
) -> Path:
    """Write :func:`environment_manifest` as JSON into ``output_dir`` and return the path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    path.write_text(json.dumps(environment_manifest(repo_dir, extra), indent=2), encoding="utf-8")
    return path
