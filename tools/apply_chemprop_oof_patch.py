"""Merge a Chemprop OOF patch into the full local QSARena OOF run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def validate_patch_for_dataset(target_predictions: pd.DataFrame, patch: pd.DataFrame, dataset: str) -> None:
    required = {"model", "split", "row_index", "predicted"}
    missing = required.difference(patch.columns)
    if missing:
        raise SystemExit(f"{dataset}: patch is missing required columns: {sorted(missing)}")
    non_oof = patch.loc[~patch["split"].astype(str).str.lower().eq("oof")]
    if not non_oof.empty:
        raise SystemExit(f"{dataset}: patch contains non-OOF rows")
    non_chemprop = patch.loc[~patch["model"].astype(str).str.startswith("Chemprop")]
    if not non_chemprop.empty:
        raise SystemExit(f"{dataset}: patch contains non-Chemprop rows")

    for model_name, model_patch in patch.groupby(patch["model"].astype(str), sort=False):
        train_rows = target_predictions.loc[
            target_predictions["model"].astype(str).eq(model_name)
            & target_predictions["split"].astype(str).str.lower().eq("train")
        ]
        if train_rows.empty:
            raise SystemExit(f"{dataset}: target run has no train rows for {model_name}")
        patch_rows = model_patch.drop_duplicates(subset=["row_index"], keep="last")
        if len(patch_rows) != len(train_rows):
            raise SystemExit(
                f"{dataset}: {model_name} OOF row count {len(patch_rows)} != train row count {len(train_rows)}"
            )
        predicted = pd.to_numeric(patch_rows["predicted"], errors="coerce")
        if predicted.isna().any():
            raise SystemExit(f"{dataset}: {model_name} patch contains non-numeric OOF predictions")


def align_columns(target: pd.DataFrame, patch: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = list(target.columns)
    for col in patch.columns:
        if col not in columns:
            columns.append(col)
    return target.reindex(columns=columns), patch.reindex(columns=columns)


def apply_patch(patch_dir: Path, target_run: Path, *, dry_run: bool = False) -> dict:
    patch_dir = patch_dir.resolve()
    target_run = target_run.resolve()
    if not patch_dir.exists():
        raise SystemExit(f"Patch directory does not exist: {patch_dir}")
    if not target_run.exists():
        raise SystemExit(f"Target run does not exist: {target_run}")

    results: list[dict] = []
    for patch_path in sorted(patch_dir.glob("*/chemprop_oof_predictions.csv")):
        dataset = patch_path.parent.name
        target_path = target_run / dataset / "predictions.csv"
        if not target_path.exists():
            raise SystemExit(f"{dataset}: missing target predictions.csv at {target_path}")
        target_predictions = pd.read_csv(target_path, low_memory=False)
        patch_predictions = pd.read_csv(patch_path, low_memory=False)
        validate_patch_for_dataset(target_predictions, patch_predictions, dataset)

        patch_models = set(patch_predictions["model"].astype(str))
        stale_mask = target_predictions["model"].astype(str).isin(patch_models) & target_predictions[
            "split"
        ].astype(str).str.lower().eq("oof")
        kept = target_predictions.loc[~stale_mask].copy()
        kept, patch_aligned = align_columns(kept, patch_predictions)
        merged = pd.concat([kept, patch_aligned], ignore_index=True)
        if not dry_run:
            merged.to_csv(target_path, index=False)
        results.append(
            {
                "dataset": dataset,
                "chemprop_members": int(len(patch_models)),
                "removed_stale_oof_rows": int(stale_mask.sum()),
                "added_oof_rows": int(len(patch_predictions)),
            }
        )

    report = {
        "patch_dir": str(patch_dir),
        "target_run": str(target_run),
        "dry_run": bool(dry_run),
        "datasets": len(results),
        "chemprop_members": int(sum(row["chemprop_members"] for row in results)),
        "added_oof_rows": int(sum(row["added_oof_rows"] for row in results)),
        "rows": results,
    }
    out_path = target_run / "chemprop_oof_patch_apply_report.json"
    if not dry_run:
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-dir", type=Path, default=Path("chemprop_oof_patch"))
    parser.add_argument(
        "--target-run",
        type=Path,
        default=Path("benchmark_results") / "qsarena_benchmark_oof_ensemble",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = apply_patch(args.patch_dir, args.target_run, dry_run=args.dry_run)
    print(
        "Applied Chemprop OOF patch"
        + (" (dry run)" if args.dry_run else "")
        + f": {report['datasets']} datasets, {report['chemprop_members']} members, "
        f"{report['added_oof_rows']} OOF rows"
    )
    if report["datasets"] == 0:
        raise SystemExit("No patch files were found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
