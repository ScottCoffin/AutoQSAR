"""Opt-in models added from the feature-expansion arm: XGBoost (ADMETboost features) and the CheMeleon Chemprop variant.

Both are off by default, so the canonical benchmark configuration (and every resume signature) is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portable_colab_qsar_bundle import run_qsarena_benchmarks as runner
from qsarena import config as qc


def _args(*argv: str):
    return runner.build_arg_parser().parse_args(["--dataset", "x.csv", *argv])


def test_both_models_are_off_by_default():
    args = _args()
    assert args.run_admetboost_xgboost is False
    assert args.run_chemprop_chemeleon is False
    assert runner.ADMETBOOST_XGB_LABEL not in runner.selected_conventional_model_names(args)
    assert not any(s["variant_tag"] == "chemeleon" for s in runner.chemprop_variant_specs(args))


def test_chemeleon_variant_spec_alone_and_alongside_others():
    alone = runner.chemprop_variant_specs(_args("--no-run-chemprop-attentivefp", "--no-run-chemprop-selected-features",
                                                "--run-chemprop-chemeleon", "--chemprop-ensemble-size", "3"))
    assert [s["variant_tag"] for s in alone] == ["chemeleon"]
    spec = alone[0]
    assert spec["train_args"] == ["--from-foundation", "CHEMELEON"]
    assert spec["label"] == "Chemprop v2 (CheMeleon fine-tuned, ensemble=3)"
    assert qc.model_family(spec["label"]) == "graph_nn"
    both = runner.chemprop_variant_specs(_args("--run-chemprop-dmpnn", "--run-chemprop-chemeleon"))
    tags = [s["variant_tag"] for s in both]
    assert "dmpnn" in tags and tags[-1] == "chemeleon"


@pytest.mark.skipif(runner.XGBRegressor is None, reason="xgboost not installed")
def test_admetboost_xgboost_is_listed_and_uses_the_arm_settings():
    from qsarena.feature_expansion.train import XGB_PARAMS

    args = _args("--run-admetboost-xgboost")
    assert runner.ADMETBOOST_XGB_LABEL in runner.selected_conventional_model_names(args)
    assert qc.model_family(runner.ADMETBOOST_XGB_LABEL) == "gradient_boosting"
    est = runner.admetboost_xgboost_estimator(args, n_jobs=1)
    params = est.get_params()
    for key, value in XGB_PARAMS.items():
        assert params[key] == value
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 4)), columns=list("abcd"))
    models = runner.conventional_models(args, X, pd.Series(np.arange(30.0)), pd.Series(["C"] * 30))
    assert runner.ADMETBOOST_XGB_LABEL in models


def test_new_flags_do_not_enter_existing_family_signatures():
    """Adding these flags must not invalidate cached rows of any existing family on resume."""
    for keys in runner._FAMILY_SIGNATURE_ARGS.values():
        assert "run_admetboost_xgboost" not in keys and "run_chemprop_chemeleon" not in keys
    base, on = _args(), _args("--run-admetboost-xgboost", "--run-chemprop-chemeleon")
    for family in runner._FAMILY_SIGNATURE_ARGS:
        assert runner.family_arg_signature(base, family) == runner.family_arg_signature(on, family)


def test_runconfig_round_trips_the_new_options():
    cfg = qc.RunConfig()
    cfg.models.admetboost_xgboost = True
    cfg.deep.chemprop.variants = ["dmpnn", "chemeleon"]
    values = cfg.to_arg_values(explicit_only=False)
    assert values["run_admetboost_xgboost"] is True
    assert values["run_chemprop_chemeleon"] is True
    assert values["run_chemprop_attentivefp"] is False


def test_explicit_disable_model_families_survives_the_profile():
    argv = ["--dataset", "x.csv", "--benchmark-profile", "quick", "--disable-model-families", "graph_nn"]
    args = runner.build_arg_parser().parse_args(argv)
    runner.apply_benchmark_profile_defaults(args, argv)
    assert args.disabled_model_families == ["graph_nn"]
    argv = ["--dataset", "x.csv", "--benchmark-profile", "quick"]
    args = runner.build_arg_parser().parse_args(argv)
    runner.apply_benchmark_profile_defaults(args, argv)
    assert "gradient_boosting" in args.disabled_model_families


def test_filtered_run_keeps_stale_rows_it_will_not_recompute():
    """A --only-model-names run must not delete stale rows of models it excludes (they are not recomputed)."""
    args = _args("--only-model-names", "XGBoost (ADMETboost features)")
    rows = [
        {"model": "TabPFNClassifier", "stage_config_signature": "old-signature"},
        {"model": "Ensemble (OOF Stacking (RidgeCV, 5-fold))", "stage_config_signature": "old-signature"},
        {"model": "XGBoost (ADMETboost features)", "stage_config_signature": "old-signature"},
    ]
    kept, stale, _legacy = runner.split_stale_metric_rows(rows, args, "stage23")
    assert stale == {"XGBoost (ADMETboost features)"}
    assert {r["model"] for r in kept} == {"TabPFNClassifier", "Ensemble (OOF Stacking (RidgeCV, 5-fold))"}
    unfiltered_kept, unfiltered_stale, _ = runner.split_stale_metric_rows(rows, _args(), "stage23")
    assert len(unfiltered_stale) == 3 and not unfiltered_kept  # an unfiltered run recomputes everything stale


def test_classification_task_with_regression_metric_returns_probabilities(monkeypatch):
    """Binary datasets catalogued with "rmse" used to save hard 0/1 labels as predictions (AUROC on labels)."""
    from sklearn.linear_model import LogisticRegression

    X = np.random.default_rng(1).normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int)
    clf = LogisticRegression().fit(X, y)
    monkeypatch.setattr(runner, "current_dataset_task_type", lambda: "classification")
    pred = runner.predict_values_for_metric(clf, X, "rmse")
    assert len(np.unique(pred)) > 2 and pred.min() >= 0 and pred.max() <= 1
    monkeypatch.setattr(runner, "current_dataset_task_type", lambda: "regression")
    assert set(np.unique(runner.predict_values_for_metric(clf, X, "rmse"))) <= {0.0, 1.0}


def test_local_tabpfn_predicts_in_chunks(monkeypatch):
    from sklearn.base import clone
    from sklearn.linear_model import LinearRegression, LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(23, 3))
    y_reg = X @ np.array([1.0, -2.0, 0.5])
    y_cls = (y_reg > 0).astype(int)
    reg = runner.ChunkedTabPFNRegressor(LinearRegression(), chunk_rows=5).fit(X, y_reg)
    np.testing.assert_allclose(reg.predict(X), LinearRegression().fit(X, y_reg).predict(X))
    cls = clone(runner.ChunkedTabPFNClassifier(LogisticRegression(), chunk_rows=4)).fit(X, y_cls)
    np.testing.assert_allclose(cls.predict_proba(X), LogisticRegression().fit(X, y_cls).predict_proba(X))
    assert list(cls.classes_) == [0, 1]
    from sklearn.base import is_classifier, is_regressor
    assert is_classifier(cls) and is_regressor(reg)
    monkeypatch.setattr(runner, "TABPFN_REGRESSOR_SOURCE", "tabpfn_client")
    monkeypatch.setattr(runner, "TabPFNClassifier", LogisticRegression)
    assert isinstance(runner.tabpfn_estimator(classification=True), LogisticRegression)


def test_chunked_tabpfn_halves_the_chunk_on_out_of_memory():
    class Flaky:
        calls: list[int] = []

        def predict(self, X):
            Flaky.calls.append(len(X))
            if len(X) > 4:
                raise RuntimeError("CUDA out of memory with 8 test samples")
            return np.asarray(X)[:, 0]

    X = np.arange(22, dtype=float).reshape(11, 2)
    reg = runner.ChunkedTabPFNRegressor(None, chunk_rows=8)
    reg.estimator_ = Flaky()
    np.testing.assert_allclose(reg.predict(X), X[:, 0])
    assert Flaky.calls[0] == 8 and max(Flaky.calls[1:]) <= 4


def test_chunked_tabpfn_reuses_the_prediction_for_repeated_scorer_calls():
    class Counting:
        calls = 0

        def predict(self, X):
            Counting.calls += 1
            return np.asarray(X)[:, 0]

    X = np.arange(10, dtype=float).reshape(5, 2)
    reg = runner.ChunkedTabPFNRegressor(None, chunk_rows=8)
    reg.estimator_ = Counting()
    for _ in range(5):
        np.testing.assert_allclose(reg.predict(X.copy()), X[:, 0])  # a new array each time, as a Pipeline passes
    assert Counting.calls == 1
    reg.predict(X + 1.0)
    assert Counting.calls == 2
