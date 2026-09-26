#!/usr/bin/env python
"""Guard against LaTeX backslash-escapes destroyed by a Python non-raw string.

Why this exists. Editing `body.tex` from a Python patch script that uses a normal (non-raw)
string turns `"\texttt"` into TAB + "exttt" and `"\ref"` into CR + "ef", silently. The file still
compiles, so `pdflatex` reports no error; the damage only shows up in the rendered PDF as
`exttttdc_cyp2c9_veith` or `Section efsec:platforms`. One such pass put 12 tabs and one mangled
cross-reference into the submission file, and it took a reviewer reading the proof to catch them.

Run this after ANY programmatic edit to the submission sources:

    python submission/check_tex_escapes.py

Exits non-zero and prints file:line for each hit. Prevention is better: use raw strings (r"...")
or double backslashes in patch scripts, and never write LaTeX through a bash heredoc.
"""

from __future__ import annotations

import pathlib
import re
import sys

# Control characters that a mangled escape leaves behind. A literal tab is legal LaTeX but never
# appears in these hand-written sources, so treating it as damage is correct here.
CONTROL = {
    "\t": r"\t (probably ate \texttt / \textbf / \table)",
    "\r": r"\r (probably ate \ref / \rightarrow)",
    "\f": r"\f (probably ate \frac / \footnote)",
    "\v": r"\v (probably ate \vspace)",
    "\a": r"\a (probably ate \alpha / \angle)",
    "\b": r"\b (probably ate \begin / \bf)",
}

# Command names left orphaned once their leading backslash-letter was consumed.
ORPHANS = [
    (r"(?<![A-Za-z\\])ef\{(?:sec|tab|fig|eq):", r"\ref{...}"),
    (r"(?<![A-Za-z\\])exttt\{", r"\texttt{...}"),
    (r"(?<![A-Za-z\\])extbf\{", r"\textbf{...}"),
    (r"(?<![A-Za-z\\])extit\{", r"\textit{...}"),
    (r"(?<![A-Za-z\\])oindent", r"\noindent"),
    (r"(?<![A-Za-z\\])egin\{", r"\begin{...}"),
    (r"(?<![A-Za-z\\])ightarrow", r"\rightarrow"),
    (r"(?<![A-Za-z\\])rac\{", r"\frac{...}"),
]


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent
    targets = sorted(root.rglob("*.tex"))
    md = root.parent / "manuscript.md"
    if md.exists():
        targets.append(md)

    hits = 0
    for path in targets:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"  SKIP {path}: {exc}")
            continue

        for ch, why in CONTROL.items():
            for m in re.finditer(re.escape(ch), text):
                line = text.count("\n", 0, m.start()) + 1
                ctx = text[max(0, m.start() - 40): m.start() + 40].replace("\n", " ")
                print(f"  {path.name}:{line}  control {why}\n      ...{ctx!r}")
                hits += 1

        for pattern, meant in ORPHANS:
            for m in re.finditer(pattern, text):
                line = text.count("\n", 0, m.start()) + 1
                ctx = text[max(0, m.start() - 40): m.start() + 40].replace("\n", " ")
                print(f"  {path.name}:{line}  orphaned command, meant {meant}\n      ...{ctx!r}")
                hits += 1

    if hits:
        print(f"\nFAIL: {hits} escape problem(s) found in {len(targets)} file(s).")
        return 1
    print(f"OK: no escape damage in {len(targets)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
