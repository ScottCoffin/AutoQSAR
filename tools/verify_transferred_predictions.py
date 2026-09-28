"""Check transferred predictions.csv files against a run's committed artifact_manifest.csv.

The per-row predictions are gitignored and move between machines by SSD. This confirms every
predictions.csv listed in the manifest is present and byte-identical (SHA-256) to the file the
run produced, so a stale or partial copy is caught before any ensemble work reads it.

Usage:
    python tools/verify_transferred_predictions.py benchmark_results/qsarena_benchmark_oof_ensemble
Exit code 0 only if every listed predictions.csv matches.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--name", default="predictions.csv", help="File name to verify (default: predictions.csv).")
    args = parser.parse_args()

    manifest = args.run_dir / "artifact_manifest.csv"
    if not manifest.exists():
        print(f"FAIL: no manifest at {manifest}")
        return 2
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if Path(row["path"].replace("\\", "/")).name == args.name]

    missing, mismatched, ok = [], [], 0
    for row in rows:
        target = args.run_dir / row["path"].replace("\\", "/")
        if not target.exists():
            missing.append(row["path"])
        elif target.stat().st_size != int(row["bytes"]) or sha256_of(target) != row["sha256"]:
            mismatched.append(row["path"])
        else:
            ok += 1

    print(f"{args.name}: {ok}/{len(rows)} match the manifest in {args.run_dir}")
    for label, paths in (("missing", missing), ("mismatched", mismatched)):
        for path in paths:
            print(f"  {label}: {path}")
    return 0 if rows and not missing and not mismatched else 1


if __name__ == "__main__":
    sys.exit(main())
