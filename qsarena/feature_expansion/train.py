"""Fixed-configuration tree models on full (unselected) feature concatenations.

For every dataset and model: 5-fold CV on the training partition, using the benchmark's own fold
geometry (``make_oof_folds``, same split strategy and seed), gives an honest CV score and an OOF
vector. Then one fit on the full training partition and one evaluation on the test partition.
Metrics use the benchmark's ``default_metric_fn``, so they are directly comparable with
``cv_primary`` and ``test_*`` in the benchmark's ``metrics.csv``.

Outputs, under ``benchmark_results/qsarena_feature_expansion/<feature_set>/``:
``metrics.csv`` (committed) and ``predictions/<dataset>__<model>.npz`` (oof, test; gitignored).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from portable_colab_qsar_bundle.qsar_workflow_core import default_metric_fn, make_oof_folds
from qsarena.feature_expansion import featurize as fz
from qsarena.meta_analysis import io

OUT_ROOT = io.REPO_ROOT / "benchmark_results" / "qsarena_feature_expansion"
RANDOM_SEED = 13  # the benchmark's run_config random_seed
CV_FOLDS = 5
REGRESSION_METRICS = ["rmse", "mae", "r2", "spearman"]
CLASSIFICATION_METRICS = ["roc_auc", "auprc"]
XGB_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.5,
    min_child_weight=1,
    tree_method="hist",
)
MODELS = ["xgboost", "random_forest", "catboost"]
DEFAULT_MODELS = ["xgboost", "random_forest"]
MODEL_LABELS = {"xgboost": "XGBoost", "random_forest": "Random forest", "catboost": "CatBoost"}


def make_model(name: str, task: str, n_jobs: int, device: str = "cpu"):
    classification = task == "classification"
    if name == "xgboost":
        import xgboost as xgb

        cls = xgb.XGBClassifier if classification else xgb.XGBRegressor
        return cls(**XGB_PARAMS, n_jobs=n_jobs, random_state=RANDOM_SEED, device=device)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cls = RandomForestClassifier if classification else RandomForestRegressor
        return cls(n_estimators=500, max_features="sqrt", n_jobs=n_jobs, random_state=RANDOM_SEED)
    if name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        cls = CatBoostClassifier if classification else CatBoostRegressor
        return cls(iterations=1000, thread_count=n_jobs, random_seed=RANDOM_SEED, verbose=False)
    raise ValueError(name)


def _predict(model, X, task):
    return model.predict_proba(X)[:, 1] if task == "classification" else model.predict(X)


def dataset_context(dataset: str, run_dir: Path = io.DEFAULT_RUN_DIR) -> dict:
    """Task, CV split strategy and primary metric exactly as the benchmark recorded them."""
    frame = pd.read_csv(
        Path(run_dir) / dataset / "metrics.csv",
        usecols=lambda c: c in {"cv_split_strategy", "primary_metric", "cv_primary", "split_strategy"},
    )
    with_cv = frame[frame.get("cv_primary", pd.Series(dtype=float)).notna()] if "cv_primary" in frame else frame
    pick = with_cv if len(with_cv) else frame
    strategy = (
        pick["cv_split_strategy"].dropna().iloc[0]
        if "cv_split_strategy" in pick and pick["cv_split_strategy"].notna().any()
        else "random"
    )
    metric = pick["primary_metric"].dropna().iloc[0] if pick["primary_metric"].notna().any() else "rmse"
    return {"cv_split_strategy": str(strategy), "primary_metric": str(metric)}


def _cv_strategy_for_folds(strategy: str) -> str:
    # The benchmark records e.g. "scaffold_group_kfold" or "random_kfold"; make_oof_folds takes the base name.
    s = strategy.lower()
    for base in ("target_quartiles", "scaffold", "random"):
        if base in s:
            return base
    return "random"


def cv_primary_value(primary_metric: str, y_train: np.ndarray, oof: np.ndarray) -> float:
    """CV score on the benchmark's recorded primary metric, whatever the task.

    The benchmark records e.g. ``rmse`` as the primary metric for some binary tasks (catalog error,
    AGENTS.md), and its ``cv_primary`` for those rows is then RMSE on probabilities. Computing the
    same metric keeps the arm's CV scores comparable with the benchmark's in the honest-selection pool.
    """
    ok = np.isfinite(oof)
    return float(default_metric_fn(primary_metric, y_train[ok], oof[ok]))


def evaluate_one(
    model_name: str,
    X: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_train: int,
    task: str,
    smiles_train: list[str],
    context: dict,
    n_jobs: int,
    device: str,
    cv_strategy: str | None = None,
    cv_only: bool = False,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """``cv_strategy`` overrides the benchmark's fold geometry (e.g. "scaffold"); ``cv_only`` skips the
    full fit and test evaluation (used when re-scoring CV only: the test predictions do not change)."""
    X_train, X_test = X[:n_train], X[n_train:]
    folds = make_oof_folds(
        np.zeros((n_train, 1)),
        y_train,
        pd.Series(smiles_train),
        split_strategy=cv_strategy or _cv_strategy_for_folds(context["cv_split_strategy"]),
        n_folds=CV_FOLDS,
        random_seed=RANDOM_SEED,
    )
    oof = np.full(n_train, np.nan)
    started = time.perf_counter()
    for fit_idx, val_idx in folds:
        model = make_model(model_name, task, n_jobs, device).fit(X_train[fit_idx], y_train[fit_idx])
        oof[val_idx] = _predict(model, X_train[val_idx], task)
    if cv_only:
        test_pred = np.full(len(y_test), np.nan)
    else:
        model = make_model(model_name, task, n_jobs, device).fit(X_train, y_train)
        test_pred = _predict(model, X_test, task)
    metrics = CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS
    row = {"seconds": round(time.perf_counter() - started, 1)}
    for m in metrics:
        ok = np.isfinite(oof)
        row[f"cv_{m}"] = float(default_metric_fn(m, y_train[ok], oof[ok]))
        row[f"test_{m}"] = np.nan if cv_only else float(default_metric_fn(m, y_test, test_pred))
    row["cv_primary"] = cv_primary_value(context["primary_metric"], y_train, oof)
    return row, oof, test_pred


def run(
    feature_set: str,
    datasets=None,
    models=DEFAULT_MODELS,
    n_jobs: int = 2,
    device: str = "cpu",
    cv_strategy: str | None = None,
    cv_only: bool = False,
) -> Path:
    fz._lower_priority()
    families = fz.FEATURE_SETS[feature_set]
    out_dir = OUT_ROOT / (f"{feature_set}__{cv_strategy}cv" if cv_strategy else feature_set)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    done = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame(columns=["dataset", "model_key"])
    partitions = io.load_partitions()
    sizes = partitions.groupby("dataset").size()
    datasets = sorted(datasets or sizes.index, key=lambda d: sizes[d])  # small first: early signal
    config = json.dumps(
        {
            "xgb": XGB_PARAMS,
            "rf": {"n_estimators": 500},
            "cb": {"iterations": 1000},
            "families": families,
            "folds": CV_FOLDS,
            "seed": RANDOM_SEED,
        },
        sort_keys=True,
    )
    for dataset in datasets:
        pending = [m for m in models if not ((done["dataset"] == dataset) & (done["model_key"] == m)).any()]
        if not pending:
            continue
        try:
            X, n_train = fz.load_features(families, dataset, partitions)
        except FileNotFoundError as exc:
            print(f"[skip] {dataset}: {exc}", flush=True)
            continue
        sub = partitions[partitions["dataset"] == dataset]
        train = sub[sub["split"] == "train"].sort_values("row_index")
        test = sub[sub["split"] == "test"].sort_values("row_index")
        y_train, y_test = train["observed"].to_numpy(float), test["observed"].to_numpy(float)
        task = io.infer_task(train["observed"])
        context = dataset_context(dataset)
        for model_name in pending:
            row, oof, test_pred = evaluate_one(
                model_name,
                X,
                y_train,
                y_test,
                n_train,
                task,
                list(train["smiles"]),
                context,
                n_jobs,
                device,
                cv_strategy=cv_strategy,
                cv_only=cv_only,
            )
            record = {
                "dataset": dataset,
                "model_key": model_name,
                "model": f"{MODEL_LABELS[model_name]} [{feature_set}]",
                "feature_set": feature_set,
                "n_features": int(X.shape[1]),
                "task": task,
                "n_train": n_train,
                "n_test": len(y_test),
                "primary_metric": context["primary_metric"],
                "cv_split_strategy": cv_strategy or context["cv_split_strategy"],
                "config": config,
                **row,
            }
            np.savez_compressed(out_dir / "predictions" / f"{dataset}__{model_name}.npz", oof=oof, test=test_pred)
            done = pd.concat([done, pd.DataFrame([record])], ignore_index=True)
            done.to_csv(metrics_path, index=False)
            print(f"{dataset} {model_name}: cv_primary={row['cv_primary']:.4f} ({row['seconds']}s)", flush=True)
    return metrics_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feature-set", default="admetboost", choices=sorted(fz.FEATURE_SETS))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=MODELS)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--device", default="cpu", help="xgboost device: cpu or cuda (use cuda after the GPU run)")
    parser.add_argument(
        "--cv-strategy",
        default=None,
        choices=["scaffold", "random", "target_quartiles"],
        help="Override the benchmark's CV fold geometry; output goes to <feature_set>__<strategy>cv/",
    )
    parser.add_argument("--cv-only", action="store_true", help="CV re-scoring only: skip the full fit and test")
    args = parser.parse_args(argv)
    print(run(args.feature_set, args.datasets, args.models, args.n_jobs, args.device, args.cv_strategy, args.cv_only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
