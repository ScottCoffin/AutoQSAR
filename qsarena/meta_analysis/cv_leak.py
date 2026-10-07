"""How optimistic is the benchmark's model-selection CV, before and after nesting feature selection?

Stage 3 of the runner fits the feature selector once on all training rows. Cross-validating each model on those
selected features lets every held-out fold influence which features were kept ("outer" selection). With
``--cv-selection nested`` the selector is refitted inside every fold; the metrics rows then carry the nested CV value
in ``cv_primary`` and the old outer value in ``cv_primary_outer``. This module measures, from deposited artifacts
only, the relative overstatement of each model's CV score over its test score on the same metric, restricted to
datasets whose test split is predefined or scaffold, under both protocols. The feature-expansion arm (no feature
selection, same CV folds) gives a reference level, and ``selection_leak.csv`` holds the controlled test (same
features, models and folds, selection outside vs inside the folds).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qsarena.meta_analysis import io

LOWER = {"mae", "rmse", "mse"}
ARM_METRICS = io.REPO_ROOT / "benchmark_results" / "qsarena_feature_expansion" / "admetboost" / "metrics.csv"
CONTROLLED_TEST = io.REPO_ROOT / "benchmark_results" / "qsarena_feature_expansion" / "selection_leak.csv"


def _test_on_primary(row: pd.Series, metric: str) -> float:
    if metric == "mse":
        rmse = row.get("test_rmse", np.nan)
        return float(rmse) ** 2 if pd.notna(rmse) else np.nan
    return float(row.get(f"test_{metric}", np.nan))


def _overstatement(rows: pd.DataFrame, cv_column: str = "cv_primary") -> pd.DataFrame:
    out = []
    for _, r in rows.iterrows():
        metric = str(r["primary_metric"])
        test = _test_on_primary(r, metric)
        cv = pd.to_numeric(r.get(cv_column, np.nan), errors="coerce")
        if not np.isfinite(test) or test == 0 or not np.isfinite(cv):
            continue
        sign = -1.0 if metric in LOWER else 1.0
        out.append({"dataset": r["dataset"], "model": r["model"], "overstatement": sign * (cv - test) / abs(test)})
    return pd.DataFrame(out, columns=["dataset", "model", "overstatement"])


def benchmark_overstatement(run_dir: Path | str = io.DEFAULT_RUN_DIR, cv_column: str = "cv_primary",
                            nested_only: bool = False) -> pd.DataFrame:
    """``cv_column="cv_primary_outer"`` with ``nested_only=True`` gives the same rows scored with the outer CV value."""
    split = io.load_dataset_summary(run_dir).set_index("dataset")["split_strategy"]
    rows = []
    for d in io.dataset_dirs(run_dir):
        m = pd.read_csv(d / "metrics.csv", low_memory=False)
        m = m[m["error"].isna()].drop_duplicates("model", keep="last")
        if nested_only:
            if "cv_selection" not in m:
                continue
            m = m[(m["cv_selection"] == "nested") & m["cv_primary"].notna() & m["cv_primary_outer"].notna()]
        else:
            m = m[m[cv_column].notna()] if cv_column in m else m.iloc[0:0]
        binary_with_regression_metric = (
            m["test_roc_auc"].notna() & m["test_rmse"].isna() & ~m["primary_metric"].isin(["roc_auc", "auprc"])
            if "test_roc_auc" in m and "test_rmse" in m
            else pd.Series(False, index=m.index)
        )
        m = m[~binary_with_regression_metric]  # cv_primary not comparable to any test column there
        rows.append(m.assign(dataset=d.name))
    frame = pd.concat(rows, ignore_index=True)
    frame = frame[frame["dataset"].map(split).isin(["predefined", "scaffold"])]
    return _overstatement(frame, cv_column)


def arm_overstatement(path: Path = ARM_METRICS, run_dir: Path | str = io.DEFAULT_RUN_DIR) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=["dataset", "model", "overstatement"])
    split = io.load_dataset_summary(run_dir).set_index("dataset")["split_strategy"]
    arm = pd.read_csv(path)
    arm = arm[arm["dataset"].map(split).isin(["predefined", "scaffold"])]
    arm = arm[~((arm["task"] == "classification") & ~arm["primary_metric"].isin(["roc_auc", "auprc"]))]
    return _overstatement(arm)


def controlled_test(path: Path = CONTROLLED_TEST) -> dict | None:
    """Selection outside vs inside the CV folds, same features, models and folds (``selection_leak.py``)."""
    if not Path(path).exists():
        return None
    from scipy.stats import binomtest

    d = pd.read_csv(path)
    d["leak"] = d["overstatement_outer_pct"] - d["overstatement_nested_pct"]
    per_model = d.groupby("model")["leak"].agg(["median", lambda s: int((s > 0).sum()), "count"])
    per_model.columns = ["median_points", "n_positive", "n"]
    n_pos = int((d["leak"] > 0).sum())
    return {
        "n_datasets": int(d["dataset"].nunique()),
        "n_pairs": int(len(d)),
        "n_positive": n_pos,
        "sign_test_p": float(binomtest(n_pos, len(d)).pvalue),
        "per_model": {m: {k: (float(v) if k == "median_points" else int(v)) for k, v in r.items()}
                      for m, r in per_model.sort_values("median_points", ascending=False).iterrows()},
    }


def summarize(run_dir: Path | str = io.DEFAULT_RUN_DIR) -> dict:
    bench = benchmark_overstatement(run_dir)
    per_model = bench.groupby("model")["overstatement"].median().sort_values(ascending=False)
    arm = arm_overstatement(run_dir=run_dir)
    paired_nested = benchmark_overstatement(run_dir, "cv_primary", nested_only=True)
    paired_outer = benchmark_overstatement(run_dir, "cv_primary_outer", nested_only=True)
    paired = paired_outer.merge(paired_nested, on=["dataset", "model"], suffixes=("_outer", "_nested"))
    return {
        "n_paired_rows": int(len(paired)),
        "paired_median_outer_pct": float(100 * paired["overstatement_outer"].median()) if len(paired) else None,
        "paired_median_nested_pct": float(100 * paired["overstatement_nested"].median()) if len(paired) else None,
        "paired_share_lower_nested_pct": (
            float(100 * (paired["overstatement_nested"] < paired["overstatement_outer"]).mean()) if len(paired) else None
        ),
        "controlled_test": controlled_test(),
        "n_benchmark_model_rows": int(len(bench)),
        "n_benchmark_models": int(per_model.size),
        "median_overstatement_all_benchmark_pct": float(100 * bench["overstatement"].median()),
        "per_model_median_pct": {k: float(100 * v) for k, v in per_model.items()},
        "per_model_median_range_pct": [float(100 * per_model.min()), float(100 * per_model.max())],
        "top_models": list(per_model.index[:3]),
        "arm_median_overstatement_pct": float(100 * arm["overstatement"].median()) if len(arm) else None,
        "n_arm_rows": int(len(arm)),
    }
