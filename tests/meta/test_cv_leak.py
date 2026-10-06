"""CV-leak measurement: orientation of the overstatement and the arm reference."""

from __future__ import annotations

import pandas as pd
import pytest

from qsarena.meta_analysis import cv_leak


def test_overstatement_sign_follows_metric_direction():
    rows = pd.DataFrame(
        [
            {"dataset": "a", "model": "m", "primary_metric": "mae", "cv_primary": 0.8, "test_mae": 1.0},
            {"dataset": "b", "model": "m", "primary_metric": "roc_auc", "cv_primary": 0.9, "test_roc_auc": 0.75},
            {"dataset": "c", "model": "m", "primary_metric": "mse", "cv_primary": 0.5, "test_rmse": 1.0},
        ]
    )
    out = cv_leak._overstatement(rows).set_index("dataset")["overstatement"]
    assert out["a"] == pytest.approx(0.2)  # lower MAE in CV than test = optimistic
    assert out["b"] == pytest.approx(0.2)  # higher AUROC in CV than test = optimistic
    assert out["c"] == pytest.approx(0.5)  # mse compared against test_rmse squared


def test_summary_on_deposited_run_shows_nesting_reduces_cv_optimism():
    summary = cv_leak.summarize()
    assert summary["n_benchmark_models"] >= 10
    if summary["paired_median_outer_pct"] is not None:  # nesting selection removes most of the optimism
        assert summary["paired_median_outer_pct"] > summary["paired_median_nested_pct"]
