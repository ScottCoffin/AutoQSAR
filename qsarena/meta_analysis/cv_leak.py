"""How optimistic is the benchmark's model-selection CV? (feature selection fitted outside the CV loop)

Stage 3 of the runner fits the ElasticNetCV feature selector once on all training rows. Each model's
cross-validation then runs on the selected features, so every held-out fold has already influenced
which features were kept. This module measures the consequence from deposited artifacts only: for each
model with a CV score, the relative overstatement of its CV score over its test score on the same
metric, restricted to datasets whose test split is predefined or scaffold. Where the
feature-expansion arm's metrics exist, its models (no feature selection, same CV folds) give the
reference level.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qsarena.meta_analysis import io

LOWER = {"mae", "rmse", "mse"}
ARM_METRICS = io.REPO_ROOT / "benchmark_results" / "qsarena_feature_expansion" / "admetboost" / "metrics.csv"


def _test_on_primary(row: pd.Series, metric: str) -> float:
    if metric == "mse":
        rmse = row.get("test_rmse", np.nan)
        return float(rmse) ** 2 if pd.notna(rmse) else np.nan
    return float(row.get(f"test_{metric}", np.nan))


def _overstatement(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in rows.iterrows():
        metric = str(r["primary_metric"])
        test = _test_on_primary(r, metric)
        if not np.isfinite(test) or test == 0 or not np.isfinite(r["cv_primary"]):
            continue
        sign = -1.0 if metric in LOWER else 1.0
        out.append(
            {"dataset": r["dataset"], "model": r["model"], "overstatement": sign * (r["cv_primary"] - test) / abs(test)}
        )
    return pd.DataFrame(out, columns=["dataset", "model", "overstatement"])


def benchmark_overstatement(run_dir: Path | str = io.DEFAULT_RUN_DIR) -> pd.DataFrame:
    split = io.load_dataset_summary(run_dir).set_index("dataset")["split_strategy"]
    rows = []
    for d in io.dataset_dirs(run_dir):
        m = pd.read_csv(d / "metrics.csv")
        m = m[m["error"].isna() & m["cv_primary"].notna()].drop_duplicates("model", keep="last")
        binary_with_regression_metric = (
            m["test_roc_auc"].notna() & m["test_rmse"].isna() & ~m["primary_metric"].isin(["roc_auc", "auprc"])
            if "test_roc_auc" in m and "test_rmse" in m
            else pd.Series(False, index=m.index)
        )
        m = m[~binary_with_regression_metric]  # cv_primary not comparable to any test column there
        rows.append(m.assign(dataset=d.name))
    frame = pd.concat(rows, ignore_index=True)
    frame = frame[frame["dataset"].map(split).isin(["predefined", "scaffold"])]
    return _overstatement(frame)


def arm_overstatement(path: Path = ARM_METRICS, run_dir: Path | str = io.DEFAULT_RUN_DIR) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=["dataset", "model", "overstatement"])
    split = io.load_dataset_summary(run_dir).set_index("dataset")["split_strategy"]
    arm = pd.read_csv(path)
    arm = arm[arm["dataset"].map(split).isin(["predefined", "scaffold"])]
    arm = arm[~((arm["task"] == "classification") & ~arm["primary_metric"].isin(["roc_auc", "auprc"]))]
    return _overstatement(arm)


def summarize(run_dir: Path | str = io.DEFAULT_RUN_DIR) -> dict:
    bench = benchmark_overstatement(run_dir)
    per_model = bench.groupby("model")["overstatement"].median().sort_values(ascending=False)
    arm = arm_overstatement(run_dir=run_dir)
    return {
        "n_benchmark_model_rows": int(len(bench)),
        "n_benchmark_models": int(per_model.size),
        "median_overstatement_all_benchmark_pct": float(100 * bench["overstatement"].median()),
        "per_model_median_pct": {k: float(100 * v) for k, v in per_model.items()},
        "per_model_median_range_pct": [float(100 * per_model.min()), float(100 * per_model.max())],
        "top_models": list(per_model.index[:3]),
        "arm_median_overstatement_pct": float(100 * arm["overstatement"].median()) if len(arm) else None,
        "n_arm_rows": int(len(arm)),
    }
