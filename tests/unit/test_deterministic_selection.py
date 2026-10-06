"""--deterministic-selection and --selected-features-from (revision R1, work order Phase 5.2).

ElasticNetCV selection falls back to random-forest importance after a wall-clock limit, so which datasets fall
back depends on the machine; re-running the benchmark on another machine changed the selection on 6 of 44
datasets. These options make selection reproducible across machines, or skip it in favour of a deposited one.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from portable_colab_qsar_bundle import run_qsarena_benchmarks as runner


def _args(*argv: str):
    return runner.build_arg_parser().parse_args(["--dataset", "x.csv", *argv])


def _spec(name: str = "toy_dataset"):
    return SimpleNamespace(name=name, smiles_column="smiles", target_column="y", predefined_split_column="",
                           task_type="", classification_threshold=None)


def test_default_selection_keeps_wall_clock_limit_and_signature():
    args = _args()
    assert args.deterministic_selection is False and args.selected_features_from is None
    assert runner.selector_run_limits(args) == {"timeout_seconds": 7200.0, "n_jobs": runner.benchmark_n_jobs(args)}
    payload = runner.stage23_args_payload(args, _spec())
    # Existing stage 2/3 caches must keep matching: no new keys unless the options are used.
    assert "deterministic_selection" not in payload and "selected_features_from" not in payload


def test_deterministic_selection_removes_limit_and_enters_signature():
    args = _args("--deterministic-selection")
    assert runner.selector_run_limits(args) == {"timeout_seconds": None, "n_jobs": 1, "single_thread": True}
    assert runner.stage23_args_payload(args, _spec())["deterministic_selection"] is True


def test_untimed_single_thread_elasticnet_fit_runs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = 2.0 * X[:, 0] + rng.normal(scale=0.1, size=60)
    folds = [(np.arange(20, 60), np.arange(20)), (np.arange(40), np.arange(40, 60))]
    fit = runner.run_timed_elasticnet_selector_fit(
        X_scaled=X, y_train=y, l1_ratio=[0.5], alphas=np.array([1e-3, 1e-2]), cv_splits=folds, max_iter=2000,
        random_seed=13, timeout_seconds=None, n_jobs=1, single_thread=True,
    )
    assert fit["ok"] and int(np.argmax(np.abs(fit["coef"]))) == 0


def _deposit(tmp_path, features, method="random_forest_importance_fallback"):
    ds = tmp_path / "deposited" / "toy_dataset"
    ds.mkdir(parents=True)
    pd.DataFrame({"feature": features}).to_csv(ds / "selected_features.csv", index=False)
    pd.DataFrame({"model": ["SVR"], "selector_method": [method]}).to_csv(ds / "metrics.csv", index=False)
    return tmp_path / "deposited"


def test_deposited_selection_is_loaded_with_its_method(tmp_path):
    root = _deposit(tmp_path, ["b", "d"])
    args = _args("--selected-features-from", str(root))
    path = runner.deposited_selection_path(args, "toy_dataset")
    X = pd.DataFrame(np.ones((3, 4)), columns=list("abcd"))
    X_train, X_test, meta = runner.load_deposited_selection(X, X.iloc[:1], path)
    assert list(X_train.columns) == ["b", "d"] and list(X_test.columns) == ["b", "d"]
    # The recorded method is kept, so nested CV refits the same method inside each fold.
    assert meta["selector_method"] == "random_forest_importance_fallback"
    assert meta["selected_features"] == ["b", "d"]
    assert len(runner.stage23_args_payload(args, _spec())["selected_features_from"]) == 64


def test_deposited_selection_fails_loudly(tmp_path):
    root = _deposit(tmp_path, ["b", "z"])
    args = _args("--selected-features-from", str(root))
    with pytest.raises(FileNotFoundError):
        runner.deposited_selection_path(args, "another_dataset")
    X = pd.DataFrame(np.ones((3, 4)), columns=list("abcd"))
    with pytest.raises(ValueError, match="absent from this run's feature matrix"):
        runner.load_deposited_selection(X, X, runner.deposited_selection_path(args, "toy_dataset"))
