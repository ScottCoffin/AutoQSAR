"""Exploratory (not pre-specified): does scaffold-fold CV track held-out test performance better than
the benchmark's own CV geometry?

Inputs: ``<feature_set>/metrics.csv`` (benchmark fold geometry, with test metrics) and
``<feature_set>__scaffoldcv/metrics.csv`` (the same models re-scored with scaffold 5-fold CV).
Per dataset this reports (a) whether each CV geometry picks the arm model (XGBoost vs random forest)
that actually tests better, and (b) each geometry's optimism, i.e. how much better the CV score
looks than the test score, relative to the test score. Results are grouped by the dataset's real test
split (predefined/scaffold vs random/target-quartile).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qsarena.feature_expansion.train import OUT_ROOT
from qsarena.meta_analysis import io

LOWER = {"mae", "rmse", "mse"}


def test_value(row: pd.Series, metric: str) -> float:
    if metric == "mse":
        return float(row["test_rmse"]) ** 2
    return float(row.get(f"test_{metric}", np.nan))


def _metric(task: str, primary: str) -> str:
    """ROC-AUC for every classification task (some are catalogued with a regression metric); else primary."""
    return "roc_auc" if task == "classification" else primary


def _cv(row: pd.Series, metric: str, primary: str) -> float:
    return float(row["cv_primary"]) if metric == primary else float(row.get(f"cv_{metric}", np.nan))


def compare(feature_set: str = "admetboost", out_root: Path = OUT_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(out_root / feature_set / "metrics.csv")
    scaf = pd.read_csv(out_root / f"{feature_set}__scaffoldcv" / "metrics.csv")
    keys = ["dataset", "model_key"]
    merged = base.merge(scaf, on=keys, suffixes=("", "_scaf"))
    split = io.load_dataset_summary().set_index("dataset")["split_strategy"]
    merged["test_split"] = merged["dataset"].map(split)
    merged["metric"] = [_metric(t, pm) for t, pm in zip(merged["task"], merged["primary_metric"])]
    merged["cv_benchmark"] = [_cv(r, r["metric"], r["primary_metric"]) for _, r in merged.iterrows()]
    merged["cv_scaffold"] = [
        float(r["cv_primary_scaf"]) if r["metric"] == r["primary_metric"] else float(r[f"cv_{r['metric']}_scaf"])
        for _, r in merged.iterrows()
    ]
    merged["test"] = [test_value(r, r["metric"]) for _, r in merged.iterrows()]
    sign = np.where(merged["metric"].isin(LOWER), -1.0, 1.0)  # +1: higher is better
    for geo in ("benchmark", "scaffold"):
        merged[f"optimism_{geo}"] = sign * (merged[f"cv_{geo}"] - merged["test"]) / merged["test"].abs()
    rows = []
    for dataset, g in merged.groupby("dataset"):
        if len(g) < 2:
            continue
        s = -1.0 if g["metric"].iloc[0] in LOWER else 1.0
        best_test = g.loc[(s * g["test"]).idxmax(), "model_key"]
        rec = {"dataset": dataset, "test_split": g["test_split"].iloc[0], "best_test": best_test}
        for geo in ("benchmark", "scaffold"):
            rec[f"pick_{geo}"] = g.loc[(s * g[f"cv_{geo}"]).idxmax(), "model_key"]
            rec[f"correct_{geo}"] = rec[f"pick_{geo}"] == best_test
        rows.append(rec)
    picks = pd.DataFrame(rows)
    return merged, picks


def summarize(merged: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
    def group(name: str, mask_m: pd.Series, mask_p: pd.Series) -> dict:
        m, p = merged[mask_m], picks[mask_p]
        return {
            "group": name,
            "datasets": int(p["dataset"].nunique()),
            "pick_correct_benchmark_cv": f"{int(p['correct_benchmark'].sum())}/{len(p)}",
            "pick_correct_scaffold_cv": f"{int(p['correct_scaffold'].sum())}/{len(p)}",
            "median_optimism_benchmark_cv_pct": round(100 * m["optimism_benchmark"].median(), 1),
            "median_optimism_scaffold_cv_pct": round(100 * m["optimism_scaffold"].median(), 1),
            "median_abs_error_benchmark_cv_pct": round(100 * m["optimism_benchmark"].abs().median(), 1),
            "median_abs_error_scaffold_cv_pct": round(100 * m["optimism_scaffold"].abs().median(), 1),
        }

    scaffold_like = {"predefined", "scaffold"}
    out = [
        group("all", merged["dataset"].notna(), picks["dataset"].notna()),
        group(
            "test split predefined/scaffold",
            merged["test_split"].isin(scaffold_like),
            picks["test_split"].isin(scaffold_like),
        ),
        group(
            "test split random/target-quartile",
            ~merged["test_split"].isin(scaffold_like),
            ~picks["test_split"].isin(scaffold_like),
        ),
    ]
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feature-set", default="admetboost")
    args = parser.parse_args(argv)
    merged, picks = compare(args.feature_set)
    summary = summarize(merged, picks)
    out = OUT_ROOT / f"{args.feature_set}__scaffoldcv"
    merged.to_csv(out / "cv_geometry_by_model.csv", index=False)
    picks.to_csv(out / "cv_geometry_picks.csv", index=False)
    summary.to_csv(out / "cv_geometry_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
