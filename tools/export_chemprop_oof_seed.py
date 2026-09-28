"""Export a compact Chemprop-only seed run for GPU OOF refits.

The full QSARena OOF run keeps predictions.csv files out of git because they are
large. Chemprop OOF refits only need the Chemprop train/test prediction payloads
as member seeds, plus matching Chemprop metric rows for provenance. This script
copies just those rows into a small run-shaped directory that can be pushed to
GitHub and used on another GPU machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_remove_tree(path: Path, cwd: Path) -> None:
    resolved = path.resolve()
    allowed_roots = [
        (cwd / "chemprop_oof_seed").resolve(),
        (cwd / "handoff" / "chemprop_oof_seed").resolve(),
    ]
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise SystemExit(f"Refusing to remove unexpected output directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def chemprop_model_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.startswith("Chemprop")


def export_seed(source_run: Path, output_dir: Path, *, overwrite: bool = False) -> dict:
    source_run = source_run.resolve()
    output_dir = output_dir.resolve()
    cwd = Path.cwd().resolve()
    if not source_run.exists():
        raise SystemExit(f"Source run does not exist: {source_run}")
    if output_dir.exists():
        if not overwrite:
            raise SystemExit(f"Output already exists; pass --overwrite to replace: {output_dir}")
        checked_remove_tree(output_dir, cwd)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    for dataset_dir in sorted(path for path in source_run.iterdir() if path.is_dir()):
        metrics_path = dataset_dir / "metrics.csv"
        predictions_path = dataset_dir / "predictions.csv"
        if not metrics_path.exists() or not predictions_path.exists():
            continue
        predictions = pd.read_csv(predictions_path, low_memory=False)
        if "model" not in predictions.columns or "split" not in predictions.columns:
            continue
        pred_mask = chemprop_model_mask(predictions["model"]) & predictions["split"].astype(str).str.lower().isin(
            {"train", "test"}
        )
        chemprop_predictions = predictions.loc[pred_mask].copy()
        if chemprop_predictions.empty:
            continue

        models_with_predictions = set(chemprop_predictions["model"].astype(str))
        metrics = pd.read_csv(metrics_path, low_memory=False)
        if "model" in metrics.columns:
            metric_mask = metrics["model"].astype(str).isin(models_with_predictions)
            chemprop_metrics = metrics.loc[metric_mask].copy()
        else:
            chemprop_metrics = pd.DataFrame()

        out_dataset = output_dir / dataset_dir.name
        out_dataset.mkdir(parents=True, exist_ok=True)
        out_metrics = out_dataset / "metrics.csv"
        out_predictions = out_dataset / "predictions.csv"
        chemprop_metrics.to_csv(out_metrics, index=False)
        chemprop_predictions.to_csv(out_predictions, index=False)

        manifest_rows.append(
            {
                "dataset": dataset_dir.name,
                "metric_rows": int(len(chemprop_metrics)),
                "prediction_rows": int(len(chemprop_predictions)),
                "chemprop_members": int(len(models_with_predictions)),
                "metrics_sha256": sha256_file(out_metrics),
                "predictions_sha256": sha256_file(out_predictions),
            }
        )

    manifest = {
        "source_run": str(source_run),
        "output_dir": str(output_dir),
        "datasets": len(manifest_rows),
        "chemprop_members": int(sum(row["chemprop_members"] for row in manifest_rows)),
        "prediction_rows": int(sum(row["prediction_rows"] for row in manifest_rows)),
        "rows": manifest_rows,
    }
    (output_dir / "CHEMPROP_OOF_SEED_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(manifest_rows).to_csv(output_dir / "CHEMPROP_OOF_SEED_MANIFEST.csv", index=False)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path("benchmark_results") / "qsarena_benchmark_oof_ensemble",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("chemprop_oof_seed"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = export_seed(args.source_run, args.output_dir, overwrite=args.overwrite)
    print(
        "Exported Chemprop OOF seed: "
        f"{manifest['datasets']} datasets, {manifest['chemprop_members']} members, "
        f"{manifest['prediction_rows']} train/test prediction rows -> {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
