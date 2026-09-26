from __future__ import annotations

import numpy as np
import pandas as pd

from portable_colab_qsar_bundle.qsar_workflow_core import (
    build_ensemble,
    fill_oof_predictions,
    is_fusion_member,
    make_oof_folds,
    oof_fold_signature,
)


def _smiles(prefix: str, n: int) -> pd.Series:
    return pd.Series([prefix * (i + 1) for i in range(n)])


def _payload(y, train_pred, oof_pred, test_pred, workflow="Conventional ML"):
    n = len(y)
    payload = {
        "workflow": workflow,
        "train_smiles": _smiles("C", n),
        "test_smiles": _smiles("N", n),
        "train_observed": np.asarray(y, dtype=float),
        "test_observed": np.asarray(y, dtype=float),
        "train": np.asarray(train_pred, dtype=float),
        "test": np.asarray(test_pred, dtype=float),
    }
    if oof_pred is not None:
        payload["oof"] = np.asarray(oof_pred, dtype=float)
    return payload


def test_core_weighted_ensemble_uses_oof_error() -> None:
    rng = np.random.default_rng(0)
    y = np.linspace(0.0, 10.0, 40)
    payloads = {
        "memoriser": _payload(y, y, y + rng.normal(0, 3.0, 40), y + rng.normal(0, 3.0, 40)),
        "honest": _payload(y, y + rng.normal(0, 0.5, 40), y + rng.normal(0, 0.6, 40), y + rng.normal(0, 0.6, 40)),
    }
    build = build_ensemble(
        payloads,
        method="Weighted average (inverse train RMSE)",
        task_type="regression",
        primary_metric="rmse",
        lower_is_better=True,
        selection_split="oof",
        drop_highly_correlated=False,
    )
    weights = dict(zip(build.weights["Model"], build.weights["Weight"]))
    assert build.label == "Weighted average (inverse OOF error)"
    assert weights["honest"] > weights["memoriser"]
    assert any("out-of-fold" in note for note in build.notes)


def test_core_excludes_fusion_outputs_and_missing_oof() -> None:
    y = np.linspace(0.0, 1.0, 12)
    payloads = {
        "a": _payload(y, y, y, y),
        "b": _payload(y, y + 0.1, y + 0.1, y + 0.1),
        "no_oof": _payload(y, y, None, y),
        "CFA (Combinatorial Fusion)": _payload(y, y, y, y, workflow="cfa"),
    }
    build = build_ensemble(
        payloads,
        method="Simple average",
        task_type="regression",
        primary_metric="rmse",
        lower_is_better=True,
        selection_split="oof",
        exclude_nonpositive_r2=False,
    )
    assert sorted(build.members) == ["a", "b"]
    assert is_fusion_member("CFA (Combinatorial Fusion)", "cfa")
    assert any("no_oof" in note and "CFA" in note for note in build.notes)


def test_core_fill_oof_predictions_uses_provider_then_memory_cache() -> None:
    y = np.arange(10, dtype=float)
    payloads = {"m": _payload(y, y, None, y)}
    folds = [(np.arange(5, 10), np.arange(0, 5)), (np.arange(0, 5), np.arange(5, 10))]
    notes = fill_oof_predictions(
        payloads,
        refitters={},
        providers={"m": lambda: (y + 1.0, "provider vector")},
        folds=folds,
        fold_signature="sig",
        n_train=10,
        memory_cache={},
        log=None,
    )
    np.testing.assert_allclose(payloads["m"]["oof"], y + 1.0)
    assert notes == ["m: provider vector"]

    calls = []
    payloads = {"m": _payload(y, y, None, y)}
    cache = {}

    def refit(_fit_idx, val_idx, _fold_dir):
        calls.append(len(val_idx))
        return y[val_idx] + 2.0

    fill_oof_predictions(
        payloads,
        refitters={"m": refit},
        folds=folds,
        fold_signature="sig2",
        n_train=10,
        memory_cache=cache,
        log=None,
    )
    np.testing.assert_allclose(payloads["m"]["oof"], y + 2.0)
    assert calls == [5, 5]
    payloads = {"m": _payload(y, y, None, y)}
    calls.clear()
    fill_oof_predictions(
        payloads,
        refitters={"m": refit},
        folds=folds,
        fold_signature="sig2",
        n_train=10,
        memory_cache=cache,
        log=None,
    )
    assert calls == []
    np.testing.assert_allclose(payloads["m"]["oof"], y + 2.0)


def test_core_oof_folds_signature_is_stable() -> None:
    X = pd.DataFrame({"x": np.arange(12)})
    y = pd.Series(np.linspace(0.0, 1.0, 12))
    folds = make_oof_folds(X, y, _smiles("C", 12), split_strategy="random", n_folds=3, random_seed=7)
    assert len(folds) == 3
    assert oof_fold_signature(folds) == oof_fold_signature(folds)
