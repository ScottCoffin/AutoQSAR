"""
qsarena — shared utilities for the QSARena HPC containerization layer.

Modules:
  precision   — TF32/BF16 precision startup hook for all torch-based workflows
  conformers  — RDKit ETKDGv3+MMFF 3D conformer generation with persistent cache
  unimolv2    — Uni-Mol2 V2 workflow wrapper (84M default, OOM retry, bf16 AMP)

Submodules are intentionally not imported here: ``precision`` and ``unimolv2``
pull in torch, which is an optional extra. Import the submodule you need.
"""

from __future__ import annotations

__all__ = ["__version__"]

#: Fallback used when running from a source checkout that was never pip-installed.
_FALLBACK_VERSION = "0.1.0"

try:  # pragma: no cover - trivial metadata lookup
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    try:
        __version__ = _pkg_version("qsarena")
    except PackageNotFoundError:
        __version__ = _FALLBACK_VERSION
except Exception:  # pragma: no cover - importlib.metadata always present on 3.10+
    __version__ = _FALLBACK_VERSION
