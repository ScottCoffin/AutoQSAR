"""Build Additional file 2 (the guided tutorial) from docs/tutorial.md.

    python submission/build_additional_file_2.py          # writes the .tex and compiles the PDF
    python submission/build_additional_file_2.py --tex-only

Steps: the tutorial's SVG figures are converted to PDF (cairosvg) under
submission/figures/additional_file_2/, the Markdown is converted to LaTeX with pandoc using a preamble
that matches Additional file 1, and the result is compiled twice with pdflatex. The Markdown is the
source of truth: docs/tutorial.md is tested command by command (tests/docs/test_tutorial_runs.py), so
edit it there, never the generated .tex.

Requires pandoc (>= 2.19), a TeX distribution with pdflatex and fvextra, and cairosvg.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SUBMISSION = Path(__file__).resolve().parent
REPO = SUBMISSION.parent
TUTORIAL = REPO / "docs" / "tutorial.md"
ASSETS = REPO / "docs" / "tutorial_assets"
FIGURES = SUBMISSION / "figures" / "additional_file_2"
STEM = "additional_file_2_qsarena_tutorial"

TITLE = (
    "Additional file 2: Guided installation and usage tutorial for QSARena"
)
SUBTITLE = (
    "No Single Model Family Dominates: Ensembles and Conventional Machine Learning Perform Comparably "
    "to Pretrained Molecular Models Across 44 Property-Prediction Benchmarks"
)

HEADER = r"""
\usepackage[htt]{hyphenat}
\usepackage{fvextra}
\fvset{breaklines=true,breakanywhere=true,fontsize=\small}
\RecustomVerbatimEnvironment{verbatim}{Verbatim}{breaklines=true,breakanywhere=true,fontsize=\small}
\setlength{\emergencystretch}{3em}
\renewcommand{\arraystretch}{1.15}
\usepackage{float}
\floatplacement{figure}{H}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\small}
"""

NOTE = (
    "This file is generated from \\texttt{docs/tutorial.md} of the QSARena repository by "
    "\\texttt{submission/build\\_additional\\_file\\_2.py}. Every command shown is executed on the "
    "example data and every output excerpt is compared with the real output by the automated test "
    "\\texttt{tests/docs/test\\_tutorial\\_runs.py}, so the tutorial and the software cannot drift apart."
)


def _highlight_args() -> list[str]:
    """pandoc >= 3.8 renamed --highlight-style to --syntax-highlighting."""
    version = subprocess.run(["pandoc", "--version"], capture_output=True, text=True, check=True).stdout.split()[1]
    major, minor = (int(part) for part in version.split(".")[:2])
    return ["--syntax-highlighting", "tango"] if (major, minor) >= (3, 8) else ["--highlight-style", "tango"]


def convert_figures() -> dict[str, str]:
    import cairosvg

    FIGURES.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for svg in sorted(ASSETS.glob("*.svg")):
        target = FIGURES / f"{svg.stem}.pdf"
        cairosvg.svg2pdf(url=str(svg), write_to=str(target))
        mapping[f"tutorial_assets/{svg.name}"] = f"figures/additional_file_2/{target.name}"
    return mapping


def prepare_markdown(text: str, figure_map: dict[str, str]) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)  # front matter (title set below)
    text = re.sub(r"\A\s*# [^\n]*\n", "", text)  # the H1 duplicates the title
    text = re.sub(r"<!--.*?-->\n?", "", text, flags=re.DOTALL)  # doctest markers
    for source, target in figure_map.items():
        text = text.replace(f"]({source})", f"]({target}){{width=95%}}")
    # Section numbers are part of the headings already.
    return text


_TEXTTT = re.compile(r"\\texttt\{([^{}]*)\}")


def allow_breaks_in_code_spans(tex: str) -> str:
    """Long option names (``--enable-shared-feature-matrix-cache``) are single tokens that overflow
    narrow table columns; allow a line break after each '-', '_' and '.' inside inline code."""

    def fix(match: "re.Match[str]") -> str:
        body = match.group(1).replace("-\\/-", "\0")
        body = body.replace("-", "-\\allowbreak{}").replace("\\_", "\\_\\allowbreak{}").replace(".", ".\\allowbreak{}")
        return "\\texttt{" + body.replace("\0", "-\\/-") + "}"

    return _TEXTTT.sub(fix, tex)


def build_tex() -> Path:
    if shutil.which("pandoc") is None:
        raise SystemExit("pandoc not found on PATH")
    figure_map = convert_figures()
    markdown = prepare_markdown(TUTORIAL.read_text(encoding="utf-8"), figure_map)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "tutorial.md"
        source.write_text(markdown, encoding="utf-8")
        header = tmp_path / "header.tex"
        header.write_text(HEADER, encoding="utf-8")
        before = tmp_path / "before.tex"
        before.write_text("\\noindent " + NOTE + "\n\n\\tableofcontents\n\\clearpage\n", encoding="utf-8")
        out = SUBMISSION / f"{STEM}.tex"
        subprocess.run(
            [
                "pandoc", str(source), "--from", "markdown-implicit_figures+pipe_tables",
                "--to", "latex", "--standalone", "--output", str(out),
                *_highlight_args(),
                "--columns", "100",
                # "## 1. Overview" becomes \section; the numbers are already in the headings.
                "--shift-heading-level-by", "-1",
                "-V", "documentclass=article",
                "-V", "classoption=11pt,a4paper",
                "-V", "geometry:margin=2cm",
                "-V", "colorlinks=true", "-V", "linkcolor=blue", "-V", "urlcolor=blue",
                "-M", f"title={TITLE}",
                "-M", f"subtitle={SUBTITLE}",
                "-M", "author=Scott Coffin",
                "-M", "date=",
                "--include-in-header", str(header),
                "--include-before-body", str(before),
            ],
            check=True,
        )
    text = allow_breaks_in_code_spans(out.read_text(encoding="utf-8"))
    banner = (
        "%% Additional file 2 -- Guided installation and usage tutorial (GENERATED; do not edit).\n"
        "%% Source: docs/tutorial.md. Rebuild: python submission/build_additional_file_2.py\n"
        "%% Compiles standalone:  pdflatex " + STEM + "\n"
    )
    out.write_text(banner + text.replace("\r\n", "\n"), encoding="utf-8", newline="\n")
    return out


def compile_pdf(tex: Path) -> Path:
    engine = shutil.which("pdflatex")
    if engine is None:
        raise SystemExit("pdflatex not found on PATH")
    for _ in range(2):
        subprocess.run([engine, "-interaction=nonstopmode", "-halt-on-error", tex.name], cwd=SUBMISSION, check=True,
                       stdout=subprocess.DEVNULL)
    for suffix in (".aux", ".log", ".out", ".toc"):
        (SUBMISSION / f"{STEM}{suffix}").unlink(missing_ok=True)
    return SUBMISSION / f"{STEM}.pdf"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tex-only", action="store_true", help="Write the .tex but do not run pdflatex.")
    args = parser.parse_args(argv)
    tex = build_tex()
    print(f"wrote {tex.relative_to(REPO)}")
    if not args.tex_only:
        pdf = compile_pdf(tex)
        print(f"wrote {pdf.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
