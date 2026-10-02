"""Feature-expansion training harness: honest CV, metric parity with the benchmark."""

from __future__ import annotations

import numpy as np
import pytest

from qsarena.feature_expansion import train as tr


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_evaluate_one_reports_cv_and_test_metrics(task):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 8)).astype(np.float32)
    signal = X[:, 0] * 2 + rng.normal(0, 0.3, 120)
    y = (signal > 0).astype(float) if task == "classification" else signal
    smiles = [f"C{'C' * (i % 7)}O" for i in range(100)]
    context = {"cv_split_strategy": "random", "primary_metric": "roc_auc" if task == "classification" else "mae"}
    row, oof, test_pred = tr.evaluate_one("random_forest", X, y[:100], y[100:], 100, task, smiles, context, 1, "cpu")
    assert np.isfinite(oof).all() and len(test_pred) == 20
    key = "cv_roc_auc" if task == "classification" else "cv_mae"
    assert row["cv_primary"] == row[key]
    if task == "classification":
        assert row["test_roc_auc"] > 0.8
    else:
        assert row["test_r2"] > 0.5


def test_cv_strategy_mapping():
    assert tr._cv_strategy_for_folds("scaffold") == "scaffold"
    assert tr._cv_strategy_for_folds("seed_ensemble_no_cv") == "random"
    assert tr._cv_strategy_for_folds("target_quartiles") == "target_quartiles"


def test_dataset_context_reads_benchmark_metric():
    context = tr.dataset_context("tdc_caco2_wang")
    assert context["primary_metric"] == "mae" and context["cv_split_strategy"] in {"random", "scaffold"}


def test_scaffold_cv_only_mode_skips_test_and_uses_scaffold_folds():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 5)).astype(np.float32)
    y = X[:, 0] + rng.normal(0, 0.1, 60)
    rings = ["c1ccccc1", "c1ccncc1", "C1CCCCC1", "c1ccc2ccccc2c1", "c1ccoc1", "C1CCNCC1"]
    smiles = [rings[i % 6] + "C" * (1 + i // 6) for i in range(50)]  # six Murcko scaffold groups
    context = {"cv_split_strategy": "random", "primary_metric": "mae"}
    row, oof, test_pred = tr.evaluate_one(
        "random_forest",
        X,
        y[:50],
        y[50:],
        50,
        "regression",
        smiles,
        context,
        1,
        "cpu",
        cv_strategy="scaffold",
        cv_only=True,
    )
    assert np.isnan(test_pred).all() and np.isnan(row["test_mae"])
    assert np.isfinite(row["cv_primary"]) and np.isfinite(oof).all()
