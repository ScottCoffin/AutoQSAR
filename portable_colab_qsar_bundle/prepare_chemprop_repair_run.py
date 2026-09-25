#!/usr/bin/env python3
"""Seed a repair run from completed metrics and predictions without copying models."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Canonical benchmark run to preserve")
    parser.add_argument("destination", type=Path, help="New, initially absent repair run directory")
    return parser.parse_args()


def successful_chemprop_pairs(metrics: pd.DataFrame) -> set[tuple[str, str]]:
    if metrics.empty or not {"dataset", "model", "error"}.issubset(metrics.columns):
        return set()
    chemprop = metrics[metrics["model"].astype(str).str.startswith("Chemprop v2")].copy()
    error = chemprop["error"].fillna("").astype(str).str.strip().str.lower()
    successful = chemprop[error.isin({"", "nan", "none"})]
    return set(zip(successful["dataset"].astype(str), successful["model"].astype(str)))


def seed_repair_run(source: Path, destination: Path) -> dict[str, int]:
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source run does not exist: {source}")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    destination.mkdir(parents=True)
    copied_datasets = 0
    metric_frames: list[pd.DataFrame] = []
    try:
        for dataset_dir in sorted(path for path in source.iterdir() if path.is_dir()):
            metrics_path = dataset_dir / "metrics.csv"
            predictions_path = dataset_dir / "predictions.csv"
            status_path = dataset_dir / "run_status.json"
            if not (metrics_path.exists() and predictions_path.exists() and status_path.exists()):
                continue

            target_dir = destination / dataset_dir.name
            target_dir.mkdir()
            for artifact in dataset_dir.iterdir():
                if artifact.is_file():
                    shutil.copy2(artifact, target_dir / artifact.name)
            metric_frames.append(pd.read_csv(metrics_path, low_memory=False))
            copied_datasets += 1
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise

    combined = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    chemprop_models = sorted(
        combined.loc[
            combined.get("model", pd.Series(dtype=str)).astype(str).str.startswith("Chemprop v2"),
            "model",
        ].dropna().astype(str).unique()
    )
    all_pairs = {
        (str(dataset), model)
        for dataset in combined.get("dataset", pd.Series(dtype=str)).dropna().astype(str).unique()
        for model in chemprop_models
    }
    successful_pairs = successful_chemprop_pairs(combined)
    return {
        "datasets": copied_datasets,
        "chemprop_models": len(chemprop_models),
        "successful_chemprop_pairs": len(successful_pairs),
        "missing_chemprop_pairs": len(all_pairs - successful_pairs),
    }


def main() -> int:
    args = parse_args()
    summary = seed_repair_run(args.source, args.destination)
    print(f"Seeded repair run: {args.destination}")
    print(
        "Coverage: "
        f"datasets={summary['datasets']}, "
        f"chemprop_models={summary['chemprop_models']}, "
        f"successful_pairs={summary['successful_chemprop_pairs']}, "
        f"missing_pairs={summary['missing_chemprop_pairs']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
