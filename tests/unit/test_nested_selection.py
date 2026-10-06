"""--cv-selection nested: per-fold feature selection for CV metrics and ensemble OOF predictions.

See docs/NESTED_SELECTION_CV_PLAN.md and qsarena/feature_expansion/selection_leak.py (the causal test).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portable_colab_qsar_bundle import run_qsarena_benchmarks as runner


def _args(*argv: str):
    return runner.build_arg_parser().parse_args(["--dataset", "x.csv", *argv])


def test_nested_cv_metrics_average_over_folds():
    y = np.arange(10.0)
    folds = [(np.arange(5, 10), np.arange(5)), (np.arange(5), np.arange(5, 10))]
    perfect = runner.nested_cv_metric_columns(y, y.copy(), folds, "rmse", classification=False)
    assert perfect["cv_rmse"] == 0.0 and perfect["cv_r2"] == 1.0 and perfect["cv_primary"] == 0.0
    shifted = runner.nested_cv_metric_columns(y, y + 1.0, folds, "mae", classification=False)
    assert shifted["cv_mae"] == pytest.approx(1.0) and shifted["cv_primary"] == pytest.approx(1.0)


def test_nested_selection_columns_pins_method_and_caches(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 8)), columns=[f"f{i}" for i in range(8)])
    y = pd.Series(3 * X["f0"] + rng.normal(scale=0.1, size=40))
    split = {"X_train": X, "y_train": y, "smiles_train": pd.Series(["C"] * 40)}
    folds = [(np.arange(20, 40), np.arange(20)), (np.arange(20), np.arange(20, 40))]
    seen = []

    def fake_select(X_fit, X_val, y_fit, smiles, args, split_strategy_for_cv=None):
        seen.append(args.selector_method)
        return X_fit[["f0"]], X_val[["f0"]], {"selector_method": args.selector_method, "selected_features": ["f0"]}

    monkeypatch.setattr(runner, "select_features", fake_select)
    meta = {"selector_method": "random_forest_importance_fallback"}
    cols = runner.nested_selection_columns(split=split, selector_meta=meta, args=_args(), folds=folds,
                                           cv_strategy="random", cache_dir=tmp_path, signature="sig1", log=lambda *a, **k: None)
    assert cols == [["f0"], ["f0"]]
    assert seen == ["rf_importance", "rf_importance"]  # the deployed method, not whatever the args default to
    seen.clear()
    again = runner.nested_selection_columns(split=split, selector_meta=meta, args=_args(), folds=folds,
                                            cv_strategy="random", cache_dir=tmp_path, signature="sig1", log=lambda *a, **k: None)
    assert again == cols and seen == []  # served from the per-fold cache
    runner.nested_selection_columns(split=split, selector_meta=meta, args=_args(), folds=folds,
                                    cv_strategy="random", cache_dir=tmp_path, signature="sig2", log=lambda *a, **k: None)
    assert len(seen) == 2  # a new signature (data, split or folds changed) recomputes


def test_profiles_default_to_nested_except_quick():
    for profile, expected in (("full", "nested"), ("cost_optimized", "nested"), ("quick", "outer")):
        argv = ["--dataset", "x.csv", "--benchmark-profile", profile]
        args = runner.build_arg_parser().parse_args(argv)
        runner.apply_benchmark_profile_defaults(args, argv)
        assert args.cv_selection == expected, profile
    argv = ["--dataset", "x.csv", "--benchmark-profile", "full", "--cv-selection", "outer"]
    args = runner.build_arg_parser().parse_args(argv)
    runner.apply_benchmark_profile_defaults(args, argv)
    assert args.cv_selection == "outer"  # an explicit choice survives the profile


def test_only_the_ensemble_signature_sees_nested_and_outer_is_unchanged():
    outer, nested = _args("--cv-selection", "outer"), _args("--cv-selection", "nested")
    legacy = _args()
    del legacy.cv_selection  # a namespace from before the option existed
    for family in runner._FAMILY_SIGNATURE_ARGS:
        assert runner.family_arg_signature(outer, family) == runner.family_arg_signature(legacy, family)
        same = runner.family_arg_signature(outer, family) == runner.family_arg_signature(nested, family)
        assert same == (family != "ensemble"), family


def test_ensemble_exclude_model_is_repeatable_and_only_changes_the_ensemble_signature():
    plain = _args()
    excluded = _args("--ensemble-exclude-model", "TabPFNRegressor", "--ensemble-exclude-model", "Voting Regressor (KNN, SVM)")
    assert runner.ensemble_excluded_models(plain) == set()
    assert runner.ensemble_excluded_models(excluded) == {"TabPFNRegressor", "Voting Regressor (KNN, SVM)"}
    assert runner.family_arg_signature(plain, "ensemble") != runner.family_arg_signature(excluded, "ensemble")
    assert runner.family_arg_signature(plain, "conventional") == runner.family_arg_signature(excluded, "conventional")


def test_withdrawn_outer_cv_is_kept_aside_and_blanked():
    row = {"model": "TabPFNRegressor", "cv_primary": 0.41, "cv_rmse": 0.41, "cv_r2": 0.7, "cv_folds": 5}
    runner.withdraw_outer_cv_metrics(row, "sig", "too large")
    assert row["cv_primary_outer"] == 0.41
    assert np.isnan(row["cv_primary"]) and np.isnan(row["cv_rmse"]) and np.isnan(row["cv_r2"])
    assert row["cv_folds"] == 5 and row["cv_selection"] == "outer_withdrawn" and row["cv_selection_note"] == "too large"
    assert _args().tabpfn_local_max_cells == 1_500_000
