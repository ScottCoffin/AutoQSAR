"""portable_colab_qsar_bundle — QSARena workflow scripts and shared libraries.

This file only marks the directory as a Python package so that the modules stay
importable after ``pip install qsarena``::

    from portable_colab_qsar_bundle.qsar_workflow_core import build_feature_matrix_from_smiles
    from portable_colab_qsar_bundle.benchmark_registry import TDC_QSAR_OPTIONS

Nothing is imported eagerly. ``qsar_workflow_core`` pulls in RDKit and
scikit-learn and ``run_qsarena_benchmarks`` is a ~10k-line CLI, so importing
either here would make every ``import portable_colab_qsar_bundle`` expensive.
Import the submodule you actually need.

The original "clone the repo and run the script" workflow is unchanged: each
module is still runnable directly (``python
portable_colab_qsar_bundle/run_qsarena_benchmarks.py --help``). The
cross-module imports try the package path first and fall back to inserting the
repository root on ``sys.path``, so both styles work.
"""

from __future__ import annotations

__all__: list[str] = []
