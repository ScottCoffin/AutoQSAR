"""Ensemble members must be selected, weighted and stacked on out-of-fold predictions.

Regression test for the defect found in benchmark_results/qsarena_benchmark_chemprop_fixed: with
member selection on in-sample training predictions, a model that memorises the training set (extra
trees, training RMSE ~0.03 on ESOL) received ~100% of the stacking weight and the ensembles fell
behind their own best member.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portable_colab_qsar_bundle import run_qsarena_benchmarks as runner


def _regression_spec():
    return runner.DatasetSpec(
        name="toy",
        source="unit",
        frame=pd.DataFrame({"SMILES": [], "y": []}),
        smiles_column="SMILES",
        target_column="y",
        recommended_metric="rmse",
        task_type="regression",
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
        "train_row_id": np.arange(n),
        "test_row_id": np.arange(n),
    }
    if oof_pred is not None:
        payload["oof"] = np.asarray(oof_pred, dtype=float)
    return payload


def _memoriser_and_honest():
    rng = np.random.default_rng(0)
    y = np.linspace(0.0, 10.0, 40)
    return {
        # Memorises the training set (in-sample error 0) but generalises badly.
        "memoriser": _payload(y, y, y + rng.normal(0, 3.0, 40), y + rng.normal(0, 3.0, 40)),
        # Honest in-sample error, and it generalises.
        "honest": _payload(y, y + rng.normal(0, 0.5, 40), y + rng.normal(0, 0.6, 40), y + rng.normal(0, 0.6, 40)),
    }


def _build(payloads, method, split, **overrides):
    previous = runner.CURRENT_DATASET_SPEC
    runner.CURRENT_DATASET_SPEC = _regression_spec()
    try:
        kwargs = dict(
            payloads=payloads,
            method=method,
            stacking_cv_folds=5,
            random_seed=0,
            drop_highly_correlated_members=False,
            max_train_prediction_correlation=0.995,
            exclude_negative_test_r2_members=True,
            member_selection_split=split,
        )
        kwargs.update(overrides)
        return runner.build_ensemble_result(**kwargs)
    finally:
        runner.CURRENT_DATASET_SPEC = previous


def test_train_split_weighting_rewards_memorisation() -> None:
    # Documents the defect that the oof mode fixes.
    _results, weight_df, *_ = _build(_memoriser_and_honest(), "Weighted average (inverse train RMSE)", "train")
    weights = dict(zip(weight_df["Model"], weight_df["Weight"]))
    assert weights["memoriser"] > 0.9


def test_oof_weighting_prefers_the_model_that_generalises() -> None:
    results, weight_df, *_ = _build(_memoriser_and_honest(), "Weighted average (inverse train RMSE)", "oof")
    weights = dict(zip(weight_df["Model"], weight_df["Weight"]))
    assert weights["honest"] > 0.7
    assert "Ensemble (Weighted average (inverse OOF error))" in set(results["model"])
    assert runner.ensemble_model_name_matches_method(
        "Ensemble (Weighted average (inverse OOF error))", "Weighted average (inverse train RMSE)"
    )


def test_oof_stacking_meta_model_is_fitted_on_oof_predictions() -> None:
    _results, weight_df, *_ = _build(_memoriser_and_honest(), "OOF Stacking (RidgeCV)", "oof")
    weights = dict(zip(weight_df["Model"], weight_df["Abs normalized contribution"]))
    assert weights["honest"] > weights["memoriser"]


def test_oof_mode_excludes_members_without_oof_and_fusion_outputs() -> None:
    payloads = _memoriser_and_honest()
    y = payloads["honest"]["train_observed"]
    payloads["no_oof"] = _payload(y, y, None, y)
    payloads["CFA (Combinatorial Fusion)"] = _payload(y, y, y, y, workflow="cfa")
    _results, _weights, _train, _test, active, notes, _meta = _build(
        payloads, "Simple average", "oof", exclude_negative_test_r2_members=False
    )
    assert sorted(active) == ["honest", "memoriser"]
    assert any("no_oof" in note and "CFA" in note for note in notes)


def test_ensure_oof_predictions_refits_per_fold_and_resumes_from_cache(tmp_path) -> None:
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": np.arange(30, dtype=float), "b": rng.normal(size=30)})
    y = pd.Series(2.0 * X["a"] + 1.0)
    payloads = {
        "lin": _payload(y, y, None, y),
        "CFA (Combinatorial Fusion)": _payload(y, y, None, y, workflow="cfa"),
    }
    calls: list[int] = []

    def refit(fit_idx, val_idx, fold_dir):
        calls.append(len(val_idx))
        model = LinearRegression().fit(X.iloc[fit_idx], y.iloc[fit_idx])
        return model.predict(X.iloc[val_idx])

    folds = runner.ensemble_oof_folds(
        X_train=X, y_train=y, smiles_train=_smiles("C", 30), split_strategy="random", cv_folds=5, random_seed=0
    )
    signature = runner.ensemble_oof_fold_signature(folds)
    persisted: dict[str, np.ndarray] = {}
    notes = runner.ensure_ensemble_oof_predictions(
        payloads=payloads,
        refitters={"lin": refit},
        folds=folds,
        fold_signature=signature,
        cache_root=tmp_path,
        n_train=30,
        on_model_done=lambda name, oof: persisted.__setitem__(name, oof),
    )
    assert notes == []
    assert len(calls) == 5 and sum(calls) == 30
    np.testing.assert_allclose(payloads["lin"]["oof"], y.to_numpy(), atol=1e-8)
    assert "oof" not in payloads["CFA (Combinatorial Fusion)"]
    assert set(persisted) == {"lin"}

    # A fresh payload for the same folds resumes from the on-disk fold cache without refitting.
    payloads["lin"].pop("oof")
    calls.clear()
    runner.ensure_ensemble_oof_predictions(
        payloads=payloads, refitters={"lin": refit}, folds=folds, fold_signature=signature,
        cache_root=tmp_path, n_train=30,
    )
    assert calls == []
    np.testing.assert_allclose(payloads["lin"]["oof"], y.to_numpy(), atol=1e-8)


def test_missing_refitter_is_reported_not_fatal(tmp_path) -> None:
    y = np.arange(10, dtype=float)
    payloads = {"gpu_model": _payload(y, y, None, y)}
    folds = [(np.arange(5, 10), np.arange(0, 5)), (np.arange(0, 5), np.arange(5, 10))]
    notes = runner.ensure_ensemble_oof_predictions(
        payloads=payloads, refitters={}, folds=folds, fold_signature="sig", cache_root=tmp_path, n_train=10
    )
    assert "oof" not in payloads["gpu_model"]
    assert notes and "gpu_model" in notes[0]


def test_oof_rows_round_trip_through_prediction_tables() -> None:
    smiles = pd.Series(["C", "CC", "CCC"])
    observed = pd.Series([1.0, 2.0, 3.0])
    train = runner.prediction_frame("toy", "m", "conventional", "train", smiles, observed, np.array([1.0, 2.0, 3.0]))
    test = runner.prediction_frame("toy", "m", "conventional", "test", pd.Series(["N", "NN"]), pd.Series([1.0, 2.0]), np.array([1.1, 2.1]))
    oof = runner.prediction_frame("toy", "m", "conventional", "oof", smiles, observed, np.array([0.9, 2.2, 2.8]))
    oof["oof_signature"] = "abc"
    payloads = runner.rebuild_prediction_payloads([pd.concat([train, test, oof], ignore_index=True)])
    np.testing.assert_allclose(payloads["m"]["oof"], [0.9, 2.2, 2.8])
    assert payloads["m"]["oof_signature"] == "abc"
    np.testing.assert_allclose(payloads["m"]["train"], [1.0, 2.0, 3.0])


def _write_unimol_dir(path, scaled_pred, scaler=None):
    import joblib

    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(np.asarray(scaled_pred, dtype=float).reshape(-1, 1), path / "cv.data")
    if scaler is not None:
        joblib.dump(scaler, path / "target_scaler.ss")
    return path


def test_unimol_saved_oof_is_unscaled_back_to_the_target(tmp_path) -> None:
    from sklearn.preprocessing import StandardScaler

    y = np.linspace(-3.0, 5.0, 20)
    scaler = StandardScaler().fit(y.reshape(-1, 1))
    oof_true = y + 0.3
    model_dir = _write_unimol_dir(tmp_path / "v1", scaler.transform(oof_true.reshape(-1, 1)), scaler)
    oof, note = runner.load_unimol_saved_oof(model_dir, n_train=20, reference_train_pred=y)
    np.testing.assert_allclose(oof, oof_true, atol=1e-10)
    assert "cv.data" in note


def test_unimol_saved_oof_classification_probabilities_pass_through(tmp_path) -> None:
    probs = np.linspace(0.05, 0.95, 12)
    model_dir = _write_unimol_dir(tmp_path / "cls", probs)
    oof, _note = runner.load_unimol_saved_oof(model_dir, n_train=12, reference_train_pred=probs)
    np.testing.assert_allclose(oof, probs)


def test_unimol_saved_oof_rejects_wrong_length_and_misaligned_rows(tmp_path) -> None:
    y = np.linspace(0.0, 1.0, 30)
    model_dir = _write_unimol_dir(tmp_path / "v1", y)
    oof, reason = runner.load_unimol_saved_oof(model_dir, n_train=29)
    assert oof is None and "rows" in reason
    oof, reason = runner.load_unimol_saved_oof(model_dir, n_train=30, reference_train_pred=y[::-1])
    assert oof is None and "line up" in reason
    oof, reason = runner.load_unimol_saved_oof(tmp_path / "missing", n_train=30)
    assert oof is None and "no cv.data" in reason


def test_saved_oof_provider_is_used_before_any_refit(tmp_path) -> None:
    y = np.arange(10, dtype=float)
    payloads = {"Uni-Mol V1": _payload(y, y, None, y, workflow="Uni-Mol")}
    folds = [(np.arange(5, 10), np.arange(0, 5)), (np.arange(0, 5), np.arange(5, 10))]

    def refit(*_args):
        raise AssertionError("must not refit when saved OOF predictions exist")

    notes = runner.ensure_ensemble_oof_predictions(
        payloads=payloads,
        refitters={"Uni-Mol V1": refit},
        providers={"Uni-Mol V1": lambda: (y + 0.5, "read saved cv.data")},
        folds=folds,
        fold_signature="sig",
        cache_root=tmp_path,
        n_train=10,
    )
    np.testing.assert_allclose(payloads["Uni-Mol V1"]["oof"], y + 0.5)
    assert any("cv.data" in note for note in notes)


def test_classification_oof_weights_do_not_collapse_onto_a_perfect_rank_score() -> None:
    y = np.array([0, 1] * 20, dtype=float)
    rng = np.random.default_rng(3)
    # Perfect ranking (AUC = 1) but unconfident probabilities: 1 / (1 - AUC) would give it ~all weight.
    perfect_rank = np.where(y == 1, 0.6, 0.4)
    # Confident and nearly as good (AUC < 1).
    confident = np.clip(np.where(y == 1, 0.9, 0.1) + rng.normal(0, 0.15, len(y)), 0.0, 1.0)
    payloads = {
        "perfect_rank": _payload(y, perfect_rank, perfect_rank, perfect_rank),
        "confident": _payload(y, confident, confident, confident),
    }
    previous = runner.CURRENT_DATASET_SPEC
    runner.CURRENT_DATASET_SPEC = runner.DatasetSpec(
        name="toy_cls",
        source="unit",
        frame=pd.DataFrame({"SMILES": [], "y": []}),
        smiles_column="SMILES",
        target_column="y",
        recommended_metric="roc_auc",
        task_type="classification",
    )
    try:
        _results, weight_df, _train, _test, _active, notes, _meta = runner.build_ensemble_result(
            payloads=payloads,
            method="Weighted average (inverse train RMSE)",
            stacking_cv_folds=5,
            random_seed=0,
            drop_highly_correlated_members=False,
            max_train_prediction_correlation=0.995,
            exclude_negative_test_r2_members=True,
            member_selection_split="oof",
        )
    finally:
        runner.CURRENT_DATASET_SPEC = previous
    weights = dict(zip(weight_df["Model"], weight_df["Weight"]))
    assert 0.2 < weights["perfect_rank"] < 0.8
    assert any("RMS probability error" in note for note in notes)
