"""Controlled test of the feature-selection leak in the benchmark's model CV (2026-10-03).

Holds everything fixed except WHERE supervised feature selection is fitted:

- ``outer``: ElasticNetCV selection fitted once on all training rows, then 5-fold CV on the selected matrix
  (what the benchmark runner does: ``select_features`` then ``cross_validate``);
- ``nested``: selection refitted inside every CV fold on that fold's training rows only.

Same features (the full ADMETboost set from the feature-expansion cache), same models, same folds, same test
split. The test score is identical for both protocols (selection on all training rows, fit, score on test), so
any difference in CV-vs-test overstatement is caused by the selection leak alone.

    python -m qsarena.feature_expansion.selection_leak            # -> benchmark_results/qsarena_feature_expansion/selection_leak.csv
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from qsarena.feature_expansion import featurize
from qsarena.feature_expansion.train import OUT_ROOT, RANDOM_SEED
from qsarena.meta_analysis import io

MODELS = {
    "SVR": lambda: make_pipeline(StandardScaler(), SVR()),
    "Random forest": lambda: RandomForestRegressor(n_estimators=300, max_features="sqrt", n_jobs=4, random_state=RANDOM_SEED),
    "ElasticNetCV": lambda: make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, max_iter=5000,
                                                                         n_jobs=4, random_state=RANDOM_SEED)),
}


def select(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Runner-style supervised selection: ElasticNetCV on imputed, standardised features; keep the largest |coef|
    up to 10% of the training rows (the runner's default cap)."""
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, max_iter=5000, n_jobs=4, random_state=RANDOM_SEED).fit(Xs, y)
    coef = np.abs(enet.coef_)
    cap = max(1, math.ceil(0.10 * len(y)))
    order = np.argsort(-coef)
    keep = order[: min(cap, int((coef > 0).sum()) or cap)]
    return np.sort(keep)


def rmse(a, b) -> float:
    return float(math.sqrt(mean_squared_error(a, b)))


def run_dataset(dataset: str, partitions: pd.DataFrame) -> list[dict]:
    X, n_train = featurize.load_features(featurize.FEATURE_SETS["admetboost"], dataset, partitions)
    X = X.astype(np.float64)
    keep_var = X[:n_train].std(axis=0) > 0  # unsupervised, label-free filter (same as the runner's variance filter)
    X = X[:, keep_var]
    sub = partitions[partitions["dataset"] == dataset]
    y_tr = sub[sub["split"] == "train"].sort_values("row_index")["observed"].to_numpy(float)
    y_te = sub[sub["split"] == "test"].sort_values("row_index")["observed"].to_numpy(float)
    X_tr, X_te = X[:n_train], X[n_train:]
    folds = list(KFold(5, shuffle=True, random_state=RANDOM_SEED).split(X_tr))
    full_keep = select(X_tr, y_tr)
    fold_keeps = [select(X_tr[fit], y_tr[fit]) for fit, _ in folds]
    out = []
    for name, factory in MODELS.items():
        t0 = time.perf_counter()
        test = rmse(y_te, clone(factory()).fit(X_tr[:, full_keep], y_tr).predict(X_te[:, full_keep]))
        oof_outer, oof_nested = np.zeros_like(y_tr), np.zeros_like(y_tr)
        for (fit, val), keep in zip(folds, fold_keeps):
            oof_outer[val] = clone(factory()).fit(X_tr[fit][:, full_keep], y_tr[fit]).predict(X_tr[val][:, full_keep])
            oof_nested[val] = clone(factory()).fit(X_tr[fit][:, keep], y_tr[fit]).predict(X_tr[val][:, keep])
        cv_outer, cv_nested = rmse(y_tr, oof_outer), rmse(y_tr, oof_nested)
        out.append({
            "dataset": dataset, "model": name, "n_train": int(n_train), "n_selected": int(len(full_keep)),
            "test_rmse": test, "cv_rmse_outer": cv_outer, "cv_rmse_nested": cv_nested,
            # positive = CV looks better than the test result (optimistic)
            "overstatement_outer_pct": 100 * (test - cv_outer) / test,
            "overstatement_nested_pct": 100 * (test - cv_nested) / test,
            "seconds": round(time.perf_counter() - t0, 1),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-train", type=int, default=1500, help="Only datasets with at most this many training rows.")
    parser.add_argument("--out", default=str(OUT_ROOT / "selection_leak.csv"))
    args = parser.parse_args(argv)
    partitions = io.load_partitions()
    summary = io.load_dataset_summary(io.DEFAULT_RUN_DIR).set_index("dataset")
    n_train = partitions[partitions["split"] == "train"].groupby("dataset").size()
    out_path = Path(args.out)
    rows: list[dict] = pd.read_csv(out_path).to_dict("records") if out_path.exists() else []  # resume
    done = {r["dataset"] for r in rows}
    for dataset in sorted(n_train.index):
        split = str(summary["split_strategy"].get(dataset, ""))
        y = partitions.loc[partitions["dataset"] == dataset, "observed"]
        if split not in ("predefined", "scaffold") or n_train[dataset] > args.max_train or y.nunique() <= 2:
            continue  # regression with a predefined or scaffold test split only
        if dataset in done:
            continue
        rows.extend(run_dataset(dataset, partitions))
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"{dataset}: done ({len(rows)} rows)", flush=True)
    frame = pd.DataFrame(rows)
    print(frame.groupby("model")[["overstatement_outer_pct", "overstatement_nested_pct"]].median().round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
