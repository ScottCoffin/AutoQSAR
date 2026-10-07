"""Phase 7 (optional, off by default): learning curves on a fixed test set.

The spec suggested the runner's ``--row-limit``, but that samples the whole dataset *before* the split,
so every curve point would get a different, smaller test set and the curve would confound training
size with test-set composition. Instead, this module writes one user CSV per (dataset, n) from the
committed, hash-verified partitions:

- **training rows:** a nested subset. The training partition is shuffled once with a fixed seed and
  the first n rows are kept, so smaller sets are contained in larger ones.
- **test rows:** the dataset's full benchmark test partition, identical at every n.

``tools/run_meta_phase7_rtx.ps1`` feeds them to the unchanged runner with
``--split-strategy predefined --predefined-split-col split``. Targets are already on the model scale,
so ``--target-transform raw`` is passed. The run is GPU work (Uni-Mol V1) and is never started by
``run_meta_analysis``; the analysis only reads its exported metrics when explicitly enabled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qsarena.meta_analysis import io

PLAN_PATH = io.REPO_ROOT / "docs" / "meta_analysis" / "RTX_PHASE7_LEARNING_CURVE_PLAN.csv"
INPUT_DIR = io.REPO_ROOT / "data" / "meta_analysis" / "phase7_inputs"
PATCH_METRICS = io.REPO_ROOT / "meta_phase7_gpu_patch" / "phase7_learning_curve_metrics.csv"
SEED = 0


def load_plan(path: Path | str = PLAN_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def curve_frame(partitions: pd.DataFrame, dataset: str, n_train: int, seed: int = SEED) -> pd.DataFrame:
    """SMILES/TARGET/split rows for one learning-curve point (n_train = 0 means the full training set)."""
    sub = partitions[partitions["dataset"] == dataset]
    if sub.empty:
        raise KeyError(f"no partition for {dataset}")
    train = sub[sub["split"] == "train"].sort_values("row_index").reset_index(drop=True)
    test = sub[sub["split"] == "test"].sort_values("row_index").reset_index(drop=True)
    order = np.random.default_rng(seed).permutation(len(train))
    n = len(train) if n_train <= 0 else min(int(n_train), len(train))
    chosen = train.iloc[np.sort(order[:n])]
    if io.infer_task(train["observed"]) == "classification" and chosen["observed"].nunique() < 2:
        raise ValueError(f"{dataset} n={n}: training subset has a single class")
    frame = pd.concat(
        [chosen.assign(split="train"), test.assign(split="test")],
        ignore_index=True,
    )
    return frame.rename(columns={"smiles": "SMILES", "observed": "TARGET"})[["SMILES", "TARGET", "split"]]


def build_inputs(
    out_dir: Path | str = INPUT_DIR,
    plan: pd.DataFrame | None = None,
    partitions: pd.DataFrame | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Write one CSV per plan row; return a manifest (dataset, label, n_train, n_test, task, path)."""
    plan = load_plan() if plan is None else plan
    partitions = io.load_partitions() if partitions is None else partitions
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in plan.itertuples(index=False):
        frame = curve_frame(partitions, record.dataset, int(record.row_limit), seed=seed)
        # <label>/<dataset>.csv: the runner names a user-CSV dataset after the file stem, so run outputs land
        # at <run>/<dataset>_<label>/<dataset>/metrics.csv, which is what export_meta_phase7_patch.py expects.
        path = out_dir / str(record.label) / f"{record.dataset}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
        train = frame[frame["split"] == "train"]
        rows.append(
            {
                "dataset": record.dataset,
                "label": record.label,
                "n_train": int(len(train)),
                "n_test": int((frame["split"] == "test").sum()),
                "task": io.infer_task(train["TARGET"]),
                "path": str(path.relative_to(io.REPO_ROOT)) if path.is_relative_to(io.REPO_ROOT) else str(path),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False, lineterminator="\n")
    return manifest


CURVE_MODELS = ["Random forest", "XGBoost", "ChemML MLP (PyTorch)", "Uni-Mol V1"]


def load_curves(path: Path | str = PATCH_METRICS) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run tools/run_meta_phase7_rtx.ps1 on a GPU machine first.")
    return pd.read_csv(path)


def _metric_column(frame: pd.DataFrame) -> str:
    return "test_roc_auc" if frame.get("test_roc_auc", pd.Series(dtype=float)).notna().any() else "test_r2"


def curve_table(patch: pd.DataFrame, run_dir: Path | str = io.DEFAULT_RUN_DIR) -> pd.DataFrame:
    """One row per (dataset, model, n_train) with the held-out metric (ROC-AUC or R^2).

    The full-training-set point is not re-run: the benchmark already trained these four models on the
    identical split, so it is read from the run's metrics.csv (valid rows only).
    """
    rows = []
    for dataset, group in patch.groupby("phase7_dataset"):
        metric = _metric_column(group)
        for record in group.itertuples(index=False):
            error = getattr(record, "error", np.nan)
            if isinstance(error, str) and error.strip():
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "model": record.model,
                    "n_train": int(record.n_train),
                    "metric": metric,
                    "value": float(getattr(record, metric)),
                    "source": "phase7",
                }
            )
        full = pd.read_csv(Path(run_dir) / dataset / "metrics.csv", low_memory=False)
        full = full[full["model"].isin(CURVE_MODELS) & (full["error"].isna() | (full["error"].astype(str) == ""))]
        for record in full.drop_duplicates("model", keep="last").itertuples(index=False):
            rows.append(
                {
                    "dataset": dataset,
                    "model": record.model,
                    "n_train": int(record.n_train),
                    "metric": metric,
                    "value": float(getattr(record, metric)),
                    "source": "benchmark_full",
                }
            )
    table = pd.DataFrame(rows)
    return table.sort_values(["dataset", "model", "n_train"]).reset_index(drop=True)


def summarize(curves: pd.DataFrame) -> dict:
    """Per dataset: the interpolated n where Uni-Mol V1 first matches the better tree model (RF / XGBoost).

    Linear interpolation in log10(n) between the first adjacent pair of sizes where the sign of
    (Uni-Mol - best tree) flips from negative to non-negative. NaN if the curves never cross.
    """
    out = {}
    for dataset, group in curves.groupby("dataset"):
        pivot = group.pivot_table(index="n_train", columns="model", values="value", aggfunc="last").sort_index()
        trees = [m for m in ("Random forest", "XGBoost") if m in pivot.columns]
        record = {"metric": str(group["metric"].iloc[0]), "sizes": [int(n) for n in pivot.index], "crossover_n": None}
        if "Uni-Mol V1" in pivot.columns and trees:
            diff = (pivot["Uni-Mol V1"] - pivot[trees].max(axis=1)).dropna()
            logn = np.log10(diff.index.to_numpy(dtype=float))
            values = diff.to_numpy()
            for i in range(1, len(values)):
                if values[i - 1] < 0 <= values[i]:
                    frac = -values[i - 1] / (values[i] - values[i - 1])
                    record["crossover_n"] = float(10 ** (logn[i - 1] + frac * (logn[i] - logn[i - 1])))
                    break
            record["unimol_minus_best_tree"] = {int(n): float(v) for n, v in diff.items()}
        out[str(dataset)] = record
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build Phase 7 learning-curve input CSVs (fixed test set).")
    parser.add_argument("--out-dir", default=str(INPUT_DIR))
    args = parser.parse_args(argv)
    manifest = build_inputs(args.out_dir)
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
