"""Export Chemprop OOF rows from a GPU seed run as a small Git patch tree."""

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
        (cwd / "chemprop_oof_patch").resolve(),
        (cwd / "handoff" / "chemprop_oof_patch").resolve(),
    ]
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise SystemExit(f"Refusing to remove unexpected patch directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def export_patch(run_dir: Path, patch_dir: Path, *, overwrite: bool = False) -> dict:
    run_dir = run_dir.resolve()
    patch_dir = patch_dir.resolve()
    cwd = Path.cwd().resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    if patch_dir.exists():
        if not overwrite:
            raise SystemExit(f"Patch directory already exists; pass --overwrite to replace: {patch_dir}")
        checked_remove_tree(patch_dir, cwd)
    patch_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for predictions_path in sorted(run_dir.glob("*/predictions.csv")):
        dataset = predictions_path.parent.name
        predictions = pd.read_csv(predictions_path, low_memory=False)
        if "model" not in predictions.columns or "split" not in predictions.columns:
            continue
        mask = predictions["model"].astype(str).str.startswith("Chemprop") & predictions["split"].astype(
            str
        ).str.lower().eq("oof")
        oof = predictions.loc[mask].copy()
        if oof.empty:
            continue
        out_dataset = patch_dir / dataset
        out_dataset.mkdir(parents=True, exist_ok=True)
        out_path = out_dataset / "chemprop_oof_predictions.csv"
        oof.to_csv(out_path, index=False)
        rows.append(
            {
                "dataset": dataset,
                "oof_rows": int(len(oof)),
                "chemprop_members": int(oof["model"].astype(str).nunique()),
                "sha256": sha256_file(out_path),
            }
        )

    manifest = {
        "source_run": str(run_dir),
        "patch_dir": str(patch_dir),
        "datasets": len(rows),
        "chemprop_members": int(sum(row["chemprop_members"] for row in rows)),
        "oof_rows": int(sum(row["oof_rows"] for row in rows)),
        "rows": rows,
    }
    (patch_dir / "CHEMPROP_OOF_PATCH_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(patch_dir / "CHEMPROP_OOF_PATCH_MANIFEST.csv", index=False)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("chemprop_oof_seed"))
    parser.add_argument("--patch-dir", type=Path, default=Path("chemprop_oof_patch"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = export_patch(args.run_dir, args.patch_dir, overwrite=args.overwrite)
    print(
        "Exported Chemprop OOF patch: "
        f"{manifest['datasets']} datasets, {manifest['chemprop_members']} members, "
        f"{manifest['oof_rows']} OOF rows -> {args.patch_dir}"
    )
    if manifest["datasets"] == 0:
        raise SystemExit("No Chemprop OOF rows found; do not commit an empty patch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
