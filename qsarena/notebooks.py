"""
qsarena.notebooks - copy the generated QSARena tutorial notebooks from an installed package.

``qsarena-notebooks DIR`` materializes the Colab/local Jupyter notebooks and the Molab
launcher/runtime files in ``DIR``. The notebook JSON is generated from
``portable_colab_qsar_bundle/build_colab_qsar_tutorial.py`` in the source tree; this command
does not regenerate it, it only makes the packaged artifacts available to pip users.
"""

from __future__ import annotations

import argparse
import shutil
from importlib import resources
from pathlib import Path
from typing import Sequence

__all__ = ["NOTEBOOK_FILES", "copy_notebooks", "main"]

NOTEBOOK_FILES = (
    "colab_qsar_tutorial.ipynb",
    "local_qsar_tutorial.ipynb",
    "molab_qsar_tutorial.py",
    "molab_qsar_runtime.py",
    "colab_qsar_workflow_map.png",
)


def _bundle_root():
    return resources.files("portable_colab_qsar_bundle")


def copy_notebooks(destination: str | Path, *, overwrite: bool = False) -> list[Path]:
    """Copy generated tutorial notebook artifacts into ``destination``."""
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    root = _bundle_root()
    for relative in NOTEBOOK_FILES:
        out = target / relative
        if out.exists() and not overwrite:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with resources.as_file(root.joinpath(relative)) as source:
            shutil.copyfile(source, out)
        written.append(out)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qsarena-notebooks",
        description="Copy the generated QSARena Colab/local/Molab tutorial notebooks into a directory.",
    )
    parser.add_argument("destination", nargs="?", default="qsarena_notebooks", help="Target directory (default: ./qsarena_notebooks).")
    parser.add_argument("--overwrite", action="store_true", help="Replace files that already exist.")
    args = parser.parse_args(argv)
    written = copy_notebooks(args.destination, overwrite=args.overwrite)
    print(f"Wrote {len(written)} file(s) to {Path(args.destination).resolve()}")
    for path in written:
        print(f"  {path.relative_to(Path(args.destination))}".replace("\\", "/"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
