"""
qsarena.examples — the small synthetic datasets used by the tutorial (Additional file 2).

``qsarena-examples DIR`` copies them, a batch manifest and a starter ``run.yaml`` into ``DIR`` so every
tutorial command can be run exactly as printed after ``pip install qsarena``.

The molecules are real structures; every target value is synthetic (a fixed formula of RDKit
descriptors plus seeded noise, see ``tests/fixtures/tutorial/make_tutorial_data.py``). They exist to
exercise the software, not to model any real property.
"""

from __future__ import annotations

import argparse
import shutil
from importlib import resources
from pathlib import Path
from typing import Sequence

__all__ = ["DATA_FILES", "copy_examples", "main"]

#: Files shipped in ``qsarena/examples/data`` (paths relative to that directory).
DATA_FILES = (
    "solubility.csv",
    "bbb.csv",
    "broken.csv",
    "batch_manifest.csv",
    "batch_dir/permeability.csv",
    "batch_dir/lipophilicity.csv",
)

#: Starter configuration written as ``run.yaml`` next to the data.
TUTORIAL_RUN_YAML = """\
# Starter run.yaml for the QSARena tutorial (qsarena-examples wrote this file).
# Run it with:   qsarena-benchmark --config run.yaml
# Paths are relative to this file. Every other key keeps its default; the full list with
# comments is configs/run.example.yaml / docs/options_reference.md.
run:
  output_dir: runs/solubility_config
input:
  path: solubility.csv
  smiles_col: smiles
  target_col: logS
  id_col: compound_id
standardize:
  strip_salts: true
  deduplicate: canonical_smiles
models:
  profile: quick
split:
  strategy: random
  seed: 13
selection:
  protocol: both
"""


def _data_root():
    return resources.files("qsarena.examples").joinpath("data")


def copy_examples(destination: str | Path, *, overwrite: bool = False) -> list[Path]:
    """Copy the example files into ``destination``; returns the written paths."""
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    root = _data_root()
    for relative in DATA_FILES:
        out = target / relative
        if out.exists() and not overwrite:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with resources.as_file(root.joinpath(relative)) as source:
            shutil.copyfile(source, out)
        written.append(out)
    run_yaml = target / "run.yaml"
    if overwrite or not run_yaml.exists():
        run_yaml.write_text(TUTORIAL_RUN_YAML, encoding="utf-8", newline="\n")
        written.append(run_yaml)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qsarena-examples",
        description="Copy the QSARena tutorial example data (synthetic targets) into a directory.",
    )
    parser.add_argument("destination", nargs="?", default="qsarena_tutorial", help="Target directory (default: ./qsarena_tutorial).")
    parser.add_argument("--overwrite", action="store_true", help="Replace files that already exist.")
    args = parser.parse_args(argv)
    written = copy_examples(args.destination, overwrite=args.overwrite)
    print(f"Wrote {len(written)} file(s) to {Path(args.destination).resolve()}")
    for path in written:
        print(f"  {path.relative_to(Path(args.destination))}".replace("\\", "/"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
