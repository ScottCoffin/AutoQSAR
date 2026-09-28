"""Run the Chemprop v2 CLI with determinism errors downgraded to warnings.

Chemprop turns on ``torch.use_deterministic_algorithms(True)`` whenever ``--pytorch-seed`` is
given. On PyTorch builds without a deterministic CUDA ``cumsum`` (e.g. 2.5.1), the AUROC
validation metric then raises for every classification run. ``warn_only=True`` still selects
deterministic kernels wherever they exist, so seeded runs that already worked are unchanged;
only the op that has no deterministic variant warns instead of aborting training.

Usage: ``python chemprop_cli_launcher.py <chemprop CLI args>`` (same arguments as ``chemprop``).
"""

from __future__ import annotations

import sys

import torch

_original_use_deterministic_algorithms = torch.use_deterministic_algorithms


def _use_deterministic_algorithms_warn_only(mode: bool, *, warn_only: bool = False) -> None:
    _original_use_deterministic_algorithms(mode, warn_only=True)


torch.use_deterministic_algorithms = _use_deterministic_algorithms_warn_only


def main() -> int:
    from chemprop.cli.main import main as chemprop_main

    sys.argv = ["chemprop", *sys.argv[1:]]
    result = chemprop_main()
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())
