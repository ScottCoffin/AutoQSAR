"""Feature-expansion evaluation: honest CV pick orientation and head-to-head bookkeeping."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qsarena.feature_expansion import evaluate as ev


def test_cv_pick_respects_metric_direction_and_pool():
    cand = pd.DataFrame(
        [
            {"dataset": "a", "model": "m1", "pool": "benchmark", "primary_metric": "mae", "cv_primary": 0.30},
            {"dataset": "a", "model": "m2", "pool": "arm", "primary_metric": "mae", "cv_primary": 0.20},
            {"dataset": "b", "model": "m1", "pool": "benchmark", "primary_metric": "roc_auc", "cv_primary": 0.80},
            {"dataset": "b", "model": "m2", "pool": "arm", "primary_metric": "roc_auc", "cv_primary": 0.90},
            {"dataset": "b", "model": "m3", "pool": "benchmark", "primary_metric": "roc_auc", "cv_primary": np.nan},
        ]
    )
    old = ev.cv_pick(cand, {"benchmark"}).set_index("dataset")["model"]
    new = ev.cv_pick(cand, {"benchmark", "arm"}).set_index("dataset")["model"]
    assert old.to_dict() == {"a": "m1", "b": "m1"}
    assert new.to_dict() == {"a": "m2", "b": "m2"}


def test_metric_kind_parsing():
    assert ev.metric_kind("ROC_AUC") == "auroc"
    assert ev.metric_kind("Test MAE") == "mae"
    assert ev.metric_kind("Spearman") == "spearman"
    assert ev.metric_kind("AUPRC") == "auprc"


def test_head_to_head_counts():
    datasets = ev.OFFICIAL_TDC
    kinds = pd.Series({d: "mae" for d in datasets})
    refs = pd.DataFrame({"ds": datasets, "model": "Ref", "value": 1.0})
    entry = pd.Series({d: (0.5 if i < 5 else 2.0) for i, d in enumerate(datasets)})
    h2h, rank = ev.tdc_table({"ours": entry}, refs, kinds)
    row = h2h.iloc[0]
    assert (row["qsarena_wins"], row["qsarena_losses"]) == (5, len(datasets) - 5)
    assert set(rank["entry"]) == {"ours", "Ref"}
