"""Build the supplementary Zenodo upload: the per-molecule predictions that .gitignore keeps out of the repo.

    python tools/build_zenodo_bundle.py                 # default runs -> dist/zenodo/
    python tools/build_zenodo_bundle.py --runs qsarena_benchmark_oof_ensemble

The GitHub->Zenodo webhook archives only the tagged tree, so the gitignored ``predictions.csv`` files (which
the paper's Availability of data and materials promises) must be uploaded by hand. Per run this writes:

- ``<run>_predictions.tar.gz``: ``benchmark_results/<run>/<dataset>/predictions.csv`` for every dataset;
- ``<run>_predictions_manifest.csv``: path, bytes, sha256 and row count for every file (also inside the tarball).

Default runs: ``qsarena_benchmark_oof_ensemble`` (every reported result; seeded from the A100 runs, so its files
hold all base models' train/test predictions plus the out-of-fold rows) and ``benchmark_name_date`` (the
consumer-GPU comparison run of §3.10). The run directory's committed ``artifact_manifest.csv``, if present, is
compared: a mismatch means the local files are not the ones the committed run describes, so investigate before uploading.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ["qsarena_benchmark_oof_ensemble", "benchmark_name_date"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def build(run: str, out_dir: Path) -> dict:
    run_dir = REPO / "benchmark_results" / run
    files = sorted(run_dir.glob("*/predictions.csv"))
    if not files:
        raise FileNotFoundError(f"{run_dir}: no */predictions.csv (they are gitignored; copy them from the machine that ran it)")
    rows = [
        {"path": f.relative_to(REPO).as_posix(), "bytes": f.stat().st_size, "sha256": sha256(f), "rows": row_count(f)}
        for f in files
    ]
    old = {}
    old_manifest = run_dir / "artifact_manifest.csv"
    if old_manifest.exists():
        with old_manifest.open(encoding="utf-8") as handle:
            old = {r["path"]: r["sha256"] for r in csv.DictReader(handle)}
    rel = {r["path"]: Path(r["path"]).relative_to(f"benchmark_results/{run}").as_posix() for r in rows}
    compared = [r for r in rows if rel[r["path"]] in old]
    changed = [r["path"] for r in compared if old[rel[r["path"]]] != r["sha256"]]

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["path", "bytes", "sha256", "rows"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    manifest_text = buffer.getvalue()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{run}_predictions_manifest.csv"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    tar_path = out_dir / f"{run}_predictions.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for r in rows:
            tar.add(REPO / r["path"], arcname=r["path"])
        info = tarfile.TarInfo(f"benchmark_results/{run}/predictions_manifest.csv")
        data = manifest_text.encode("utf-8")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return {"run": run, "files": len(rows), "raw_mb": sum(r["bytes"] for r in rows) / 1e6,
            "tar_mb": tar_path.stat().st_size / 1e6, "tar": tar_path, "compared": len(compared), "changed": len(changed)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--out-dir", default=str(REPO / "dist" / "zenodo"))
    args = parser.parse_args(argv)
    for run in args.runs:
        s = build(run, Path(args.out_dir))
        print(f"{s['run']}: {s['files']} files, {s['raw_mb']:.0f} MB raw -> {s['tar']} ({s['tar_mb']:.0f} MB); "
              f"artifact_manifest.csv: {s['compared']} compared, {s['changed']} differ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
