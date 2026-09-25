"""Re-execute benchmark_results_summary.ipynb headlessly and save it with fresh outputs.

This regenerates manuscript_assets/ (figures, tables, manuscript_numbers.json) and the
publication_*.csv files inside the benchmark run directory, then refreshes every
<!-- TABLE:<stem> --> ... <!-- /TABLE --> block in manuscript.md. Run from the repository root:

    python portable_colab_qsar_bundle/render_manuscript_assets.py [--run-dir benchmark_results/<run>]

Only pandas, numpy, matplotlib, plotly, scipy, nbformat and nbclient are needed (RDKit is optional);
the full autoqsar-py311 conda environment is not required. Takes well under a minute.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "portable_colab_qsar_bundle" / "benchmark_results_summary.ipynb"
MANUSCRIPT = REPO_ROOT / "manuscript.md"
TABLE_DIR = REPO_ROOT / "manuscript_assets" / "tables"
TABLE_BLOCK = re.compile(r"(<!-- TABLE:(?P<stem>[A-Za-z0-9_]+) -->\n)(?P<body>.*?)(<!-- /TABLE -->)", re.DOTALL)


def sync_manuscript_tables() -> list[str]:
    """Replace each <!-- TABLE:stem --> ... <!-- /TABLE --> block in manuscript.md with tables/<stem>.md."""
    if not MANUSCRIPT.exists():
        return []
    text = MANUSCRIPT.read_text(encoding="utf-8")
    missing: list[str] = []

    def replace(match: re.Match) -> str:
        table_path = TABLE_DIR / f"{match['stem']}.md"
        if not table_path.exists():
            missing.append(match["stem"])
            return match.group(0)
        return match.group(1) + table_path.read_text(encoding="utf-8") + match.group(4)

    MANUSCRIPT.write_text(TABLE_BLOCK.sub(replace, text), encoding="utf-8")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default=None, help="Override benchmark_run_dir (repo-relative or absolute).")
    parser.add_argument("--kernel", default="python3", help="Jupyter kernel name used for execution.")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-cell timeout in seconds.")
    args = parser.parse_args()

    nb = nbformat.read(NOTEBOOK, as_version=4)
    saved_kernelspec = nb.metadata.get("kernelspec")
    if args.run_dir:
        for cell in nb.cells:
            if cell.cell_type == "code" and "benchmark_run_dir = " in cell.source:
                cell.source = re.sub(r'benchmark_run_dir = r?"[^"]*"', f'benchmark_run_dir = r"{args.run_dir}"', cell.source, count=1)
                break

    client = NotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name=args.kernel,
        allow_errors=True,
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()

    errors = [
        (index, output.ename, str(output.evalue)[:300])
        for index, cell in enumerate(nb.cells)
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    if saved_kernelspec:
        nb.metadata["kernelspec"] = saved_kernelspec
    nbformat.write(nb, NOTEBOOK)
    for index, name, value in errors:
        print(f"cell {index}: {name}: {value}", file=sys.stderr)
    print(f"Executed {NOTEBOOK.name}; {len(errors)} cell error(s). Assets in {REPO_ROOT / 'manuscript_assets'}")
    missing = sync_manuscript_tables()
    if missing:
        print(f"manuscript.md references missing tables: {', '.join(missing)}", file=sys.stderr)
    print("Refreshed table blocks in manuscript.md. Prose numbers still need checking against manuscript_numbers.json.")

    # Keep the LaTeX submission package in step with the Markdown: both are generated from the same CSVs.
    try:
        import subprocess
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "portable_colab_qsar_bundle" / "render_latex_tables.py")],
            cwd=REPO_ROOT, check=True,
        )
    except Exception as exc:  # pragma: no cover - the LaTeX package is optional
        print(f"Could not refresh LaTeX tables: {exc}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
