"""Export the optional Phase 7 RTX learning-curve metrics as a small patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


DEFAULT_KEEP_COLUMNS = [
    "dataset",
    "model",
    "workflow",
    "status",
    "error",
    "n_molecules",
    "n_train",
    "n_test",
    "split_strategy",
    "target_transform",
    "selected_feature_count",
    "primary_metric",
    "primary_metric_value",
    "cv_primary",
    "cv_r2",
    "cv_rmse",
    "cv_mae",
    "train_r2",
    "train_rmse",
    "train_mae",
    "train_spearman",
    "test_r2",
    "test_rmse",
    "test_mae",
    "test_spearman",
    "cv_roc_auc",
    "cv_auprc",
    "train_roc_auc",
    "test_roc_auc",
    "test_auprc",
    "test_balanced_accuracy",
    "test_mcc",
    "elapsed_seconds",
    "dataset_elapsed_seconds",
    "cost_scope",
    "stage_duration_seconds",
    "unimol_internal_split",
    "unimol_epochs",
    "unimol_learning_rate",
    "unimol_batch_size",
    "unimol_early_stopping",
    "unimol_num_workers",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_remove_tree(path: Path, cwd: Path) -> None:
    resolved = path.resolve()
    allowed_roots = [
        (cwd / "meta_phase7_gpu_patch").resolve(),
        (cwd / "handoff" / "meta_phase7_gpu_patch").resolve(),
    ]
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise SystemExit(f"Refusing to remove unexpected patch directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def parse_models(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(";") if part.strip()]


def run_subdir(dataset: str, label: str) -> str:
    return f"{dataset}_{label}"


def collect_phase7_metrics(
    run_root: Path,
    manifest_path: Path,
    patch_dir: Path,
    *,
    overwrite: bool = False,
    allow_incomplete: bool = False,
) -> dict:
    run_root = run_root.resolve()
    manifest_path = manifest_path.resolve()
    patch_dir = patch_dir.resolve()
    cwd = Path.cwd().resolve()

    if not manifest_path.exists():
        raise SystemExit(f"Manifest does not exist: {manifest_path}")
    if not run_root.exists():
        raise SystemExit(f"Run root does not exist: {run_root}")
    if patch_dir.exists():
        if not overwrite:
            raise SystemExit(f"Patch directory already exists; pass --overwrite to replace: {patch_dir}")
        checked_remove_tree(patch_dir, cwd)
    patch_dir.mkdir(parents=True, exist_ok=True)

    plan = pd.read_csv(manifest_path)
    required = {"dataset", "row_limit", "label", "models"}
    missing_columns = required.difference(plan.columns)
    if missing_columns:
        raise SystemExit(f"Manifest is missing required columns: {sorted(missing_columns)}")

    metric_frames: list[pd.DataFrame] = []
    status_rows: list[dict] = []
    incomplete: list[str] = []

    for _, plan_row in plan.iterrows():
        dataset = str(plan_row["dataset"])
        label = str(plan_row["label"])
        expected_models = parse_models(str(plan_row["models"]))
        row_limit = int(plan_row["row_limit"])
        subdir = run_subdir(dataset, label)
        metrics_path = run_root / subdir / dataset / "metrics.csv"
        status = {
            "dataset": dataset,
            "row_limit": row_limit,
            "label": label,
            "run_dir": str(run_root / subdir),
            "metrics_path": str(metrics_path),
            "expected_models": ";".join(expected_models),
            "found_models": "",
            "selected_rows": 0,
            "status": "missing_metrics",
            "message": "",
        }
        if not metrics_path.exists():
            status["message"] = "metrics.csv not found"
            incomplete.append(f"{dataset}/{label}: missing metrics.csv")
            status_rows.append(status)
            continue

        metrics = pd.read_csv(metrics_path, low_memory=False)
        if "model" not in metrics.columns:
            status["status"] = "invalid_metrics"
            status["message"] = "metrics.csv has no model column"
            incomplete.append(f"{dataset}/{label}: metrics.csv has no model column")
            status_rows.append(status)
            continue

        selected = metrics.loc[metrics["model"].astype(str).isin(expected_models)].copy()
        found_models = sorted(selected["model"].astype(str).unique())
        missing_models = sorted(set(expected_models).difference(found_models))
        status["found_models"] = ";".join(found_models)
        status["selected_rows"] = int(len(selected))
        if missing_models:
            status["status"] = "missing_models"
            status["message"] = "Missing models: " + ";".join(missing_models)
            incomplete.append(f"{dataset}/{label}: missing {', '.join(missing_models)}")
        else:
            status["status"] = "ok"

        keep_columns = [column for column in DEFAULT_KEEP_COLUMNS if column in selected.columns]
        selected = selected.loc[:, keep_columns]
        selected.insert(0, "phase7_dataset", dataset)
        selected.insert(1, "phase7_row_limit", row_limit)
        selected.insert(2, "phase7_label", label)
        selected.insert(3, "phase7_run_dir", str(run_root / subdir))
        selected.insert(4, "phase7_metrics_path", str(metrics_path))
        metric_frames.append(selected)
        status_rows.append(status)

    if incomplete and not allow_incomplete:
        detail = "\n  - ".join(incomplete)
        raise SystemExit(f"Phase 7 export is incomplete:\n  - {detail}")

    metrics_out = patch_dir / "phase7_learning_curve_metrics.csv"
    status_out = patch_dir / "phase7_learning_curve_manifest.csv"
    if metric_frames:
        pd.concat(metric_frames, ignore_index=True).to_csv(metrics_out, index=False)
    else:
        pd.DataFrame().to_csv(metrics_out, index=False)
    pd.DataFrame(status_rows).to_csv(status_out, index=False)

    manifest = {
        "source_run_root": str(run_root),
        "source_manifest": str(manifest_path),
        "patch_dir": str(patch_dir),
        "planned_runs": int(len(plan)),
        "complete_runs": int(sum(row["status"] == "ok" for row in status_rows)),
        "selected_metric_rows": int(sum(row["selected_rows"] for row in status_rows)),
        "metrics_sha256": sha256_file(metrics_out),
        "manifest_sha256": sha256_file(status_out),
        "allow_incomplete": allow_incomplete,
    }
    (patch_dir / "PHASE7_GPU_PATCH_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("benchmark_results/qsarena_meta_phase7_gpu"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/meta_analysis/RTX_PHASE7_LEARNING_CURVE_PLAN.csv"),
    )
    parser.add_argument("--patch-dir", type=Path, default=Path("meta_phase7_gpu_patch"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write available rows even if one or more planned runs are missing.",
    )
    args = parser.parse_args()

    manifest = collect_phase7_metrics(
        args.run_root,
        args.manifest,
        args.patch_dir,
        overwrite=args.overwrite,
        allow_incomplete=args.allow_incomplete,
    )
    print(
        "Exported Phase 7 GPU metrics patch: "
        f"{manifest['complete_runs']}/{manifest['planned_runs']} runs, "
        f"{manifest['selected_metric_rows']} metric rows -> {args.patch_dir}"
    )
    if manifest["selected_metric_rows"] == 0:
        raise SystemExit("No Phase 7 metric rows found; do not commit an empty patch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
