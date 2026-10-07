"""Dataset-property meta-analysis: which dataset properties predict which model family wins.

Built only from deposited single-seed benchmark artifacts (no retraining). Entry point:
:func:`qsarena.meta_analysis.pipeline.run_meta_analysis`, also ``python -m qsarena.meta_analysis``.
Methods: ``docs/meta_analysis/METHODS.md``; artifact inventory: ``docs/meta_analysis/INVENTORY.md``.
"""

from __future__ import annotations

__all__ = ["run_meta_analysis"]


def run_meta_analysis(*args, **kwargs):
    from qsarena.meta_analysis.pipeline import run_meta_analysis as _run

    return _run(*args, **kwargs)
