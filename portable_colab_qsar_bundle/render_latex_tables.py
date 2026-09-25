"""Convert manuscript_assets/tables/*.csv into LaTeX fragments for the submission package.

Run from the repository root, after render_manuscript_assets.py:

    python portable_colab_qsar_bundle/render_latex_tables.py

Writes submission/tables/<stem>.tex, one booktabs tabular per table, with the same numbers as the
Markdown tables (both are generated from the same CSVs, so the two manuscript formats cannot drift).
Wide tables are emitted as sidewaystable (landscape) environments.

Per Springer Nature LaTeX guidance, all non-ASCII characters are converted to TeX commands.
"""

from __future__ import annotations

import csv
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV_DIR = REPO_ROOT / "manuscript_assets" / "tables"
OUT_DIR = REPO_ROOT / "submission" / "tables"

# Springer: "convert special characters ... into the appropriate TeX code".
UNICODE_TO_TEX = {
    "–": "--", "—": "---", "−": "$-$", "×": "$\\times$", "≈": "$\\approx$",
    "±": "$\\pm$", "≥": "$\\ge$", "≤": "$\\le$", "→": "$\\rightarrow$",
    "²": "$^{2}$", "³": "$^{3}$", "⁻": "$^{-}$", "·": "$\\cdot$", "★": "$\\star$",
    "“": "``", "”": "''", "‘": "`", "’": "'", "…": "\\ldots", " ": "~",
    "α": "$\\alpha$", "β": "$\\beta$", "ε": "$\\epsilon$", "μ": "$\\mu$", "σ": "$\\sigma$",
    "é": "\\'e", "í": "\\'i", "á": "\\'a", "ó": "\\'o", "ú": "\\'u",
    "ä": '\\"a', "ö": '\\"o', "ü": '\\"u', "ñ": "\\~n", "ç": "\\c{c}",
    "č": "\\v{c}", "š": "\\v{s}", "ž": "\\v{z}", "ė": "\\.e", "ą": "\\k{a}",
    "ū": "\\={u}", "ī": "\\={i}", "ł": "\\l{}", "ø": "\\o{}", "å": "\\aa{}",
}
LATEX_SPECIALS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}

# stem -> (label, landscape?, rounding digits, caption)
TABLE_SPEC = {
    "table1_model_inventory": (
        "tab:models", False, 3,
        "Model inventory. ``Valid'' datasets are those on which the model produced a metric without an error; "
        "differences from 44 reflect task-type applicability (regression-only or classification-only estimators), "
        "dataset-size guardrails, or backend failures (Section~\\ref{sec:coverage}).",
    ),
    "table2_dataset_catalog": (
        "tab:datasets", True, 3,
        "Dataset catalog. ``Ranking metric'' is the metric used for cross-dataset model ranking (RMSE for regression; "
        "the dataset's designated primary classification metric otherwise); leaderboard comparisons in "
        "Table~\\ref{tab:leaderboard} use the leaderboard's own metric, which differs for several TDC regression tasks.",
    ),
    "table3_architecture_families": (
        "tab:families", False, 1,
        "Model-family coverage and consistency. Gaps are relative to the per-dataset best primary metric. "
        "``Within 5\\% of best'' counts datasets where the family's best member fell within 5\\% of the dataset winner. "
        "Families evaluated on fewer than 44 datasets were limited by task applicability, size guardrails or backend "
        "failures (Additional file~1, Table~S4); their percentages are computed over the datasets on which they ran. "
        "All values derive from a single split and seed, so adjacent rows are not separated (see Limitations).",
    ),
    "table4_leaderboard_comparison": (
        "tab:leaderboard", True, 3,
        "Per-dataset leaderboard comparison, sorted by estimated rank. ``References (n)'' is the number of published "
        "values available for that dataset and metric; ranks from sparse reference sets are correspondingly uncertain. "
        "The \\texttt{tdc\\_ppbr\\_az} top-1 reference (MAE 0.679) is inconsistent in scale with the rest of that "
        "dataset's references (top-10 cutoff 7.914) and its top-1 gap should be disregarded.",
    ),
    "table5_ensemble_value_add": (
        "tab:ensemble", False, 3,
        "Ensemble and fusion value-add against the best single base model available on the same dataset, under each "
        "dataset's primary metric.",
    ),
    "table6_cost": (
        "tab:cost", False, 1,
        "Per-family computational cost and model size in this benchmark, with published comparators. Wall-clock times "
        "are per model per dataset on the run hardware.",
    ),
    "tableS1_dataset_winners": (
        "tab:s1", True, 3,
        "Per-dataset winning model under the ranking metric, with the corresponding cross-validation-selected model "
        "and its relative gap to the dataset best (\\%).",
    ),
    "tableS2_component_ablation": (
        "tab:s2", False, 3,
        "Staged component ablation: datasets whose best achievable primary metric improved when each pipeline stage "
        "was added, computed from the existing model results.",
    ),
    "tableS3_feature_families": (
        "tab:s3", False, 2,
        "Feature-family selection summary across all datasets, sorted by per-feature enrichment relative to a "
        "uniform-selection baseline.",
    ),
    "tableS5_run_comparison": (
        "tab:s5", True, 3,
        "Run-to-run comparison: the canonical NSF ACCESS Jetstream2 A100 benchmark against the earlier "
        "consumer-GPU (RTX 4060) run, analysed identically. ``Same split'' marks datasets where both runs "
        "used the identical held-out partition; the remainder were re-split to scaffold splits in the A100 "
        "run and are therefore not directly comparable. Change is relative and metric-direction aware "
        "(positive favours the A100 run).",
    ),
    "tableS4_model_coverage": (
        "tab:s4", False, 3,
        "Model coverage: datasets attempted and datasets yielding a valid metric, per model.",
    ),
}


def tex_escape(text: str) -> str:
    out = []
    for ch in str(text):
        if ch in LATEX_SPECIALS:
            out.append(LATEX_SPECIALS[ch])
        elif ch in UNICODE_TO_TEX:
            out.append(UNICODE_TO_TEX[ch])
        elif ord(ch) < 128:
            out.append(ch)
        else:
            # Never silently emit a byte pdflatex may reject.
            out.append("?")
            print(f"  warning: unmapped character {ch!r} (U+{ord(ch):04X})", file=sys.stderr)
    return "".join(out)


def cell_escape(text: str) -> str:
    """Escape a table cell and allow line breaks inside long identifiers.

    Dataset and model names such as ``polaris_adme_fang_rclint_1`` are single unbreakable words once
    the underscores are escaped, so they overflow narrow columns. Adding \\allowbreak after each
    underscore, slash and hyphen lets them wrap without changing the printed characters.
    """
    escaped = tex_escape(text)
    for token in (r"\_", "/", "-"):
        escaped = escaped.replace(token, token + r"\allowbreak{}")
    # CamelCase model names (LogisticRegression, HistGradientBoosting) are also unbreakable; allow a
    # break at each internal capital of a long word.
    escaped = re.sub(r"(?<=[a-z])(?=[A-Z][a-z])", r"\\allowbreak{}", escaped)
    return escaped


def is_numeric(value: str) -> bool:
    try:
        float(str(value).replace(",", ""))
        return True
    except ValueError:
        return False


def format_cell(value: str, digits: int) -> str:
    """Round floats the same way the Markdown tables do, so the two formats agree cell for cell."""
    text = str(value).strip()
    if not text or not is_numeric(text):
        return text
    number = float(text.replace(",", ""))
    if number == int(number) and abs(number) < 1e15 and "." not in text:
        return f"{int(number):,}" if abs(number) >= 1000 else str(int(number))
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.{digits}f}"


LONG_TABLE_ROWS = 16  # beyond this, use a page-breaking xltabular instead of a float


def build_table(rows: list[list[str]], landscape: bool, label: str, digits: int, caption: str) -> str:
    """Emit a width-constrained table.

    Long text columns become tabularx `X` columns so the table is always exactly \\textwidth and the
    text wraps instead of overflowing; numeric columns keep their natural width. Tables longer than
    LONG_TABLE_ROWS use xltabular (longtable + tabularx) so they break across pages rather than
    being dropped by the float placer.
    """
    header, body = rows[0], rows[1:]
    formatted = [[format_cell(c, digits) for c in (list(r) + [""] * (len(header) - len(r)))[: len(header)]] for r in body]

    widths, is_num = [], []
    for i, head in enumerate(header):
        column = [r[i] for r in formatted if str(r[i]).strip()]
        is_num.append(bool(column) and all(is_numeric(v) for v in column))
        widths.append(max([len(str(v)) for v in column] + [len(str(head))]))

    # Every text column becomes a wrapping X column. Leaving even a few text columns at natural
    # width starves the X columns and makes each row overflow individually.
    wrap_idx = {i for i in range(len(header)) if not is_num[i]}

    # Weight X columns by content length. tabularx requires the multipliers to sum to the number of
    # X columns, otherwise the table no longer measures \textwidth and overflows again.
    weights = {}
    if wrap_idx:
        total = sum(widths[i] for i in wrap_idx)
        n = len(wrap_idx)
        for i in wrap_idx:
            weights[i] = max(0.35, round(n * widths[i] / total, 3))
        scale = n / sum(weights.values())
        weights = {i: round(w * scale, 3) for i, w in weights.items()}

    spec = []
    for i in range(len(header)):
        if i in wrap_idx:
            if len(wrap_idx) > 1:
                spec.append(">{\\raggedright\\arraybackslash\\hsize=%.3f\\hsize}X" % weights[i])
            else:
                spec.append(">{\\raggedright\\arraybackslash}X")
        else:
            spec.append("r" if is_num[i] else "l")
    colspec = "".join(spec)

    size = "\\scriptsize" if len(header) >= 9 or sum(widths) > 130 else "\\footnotesize"

    def header_cell(index: int, text: str) -> str:
        """Wrap long headers. A narrow numeric column cannot wrap its own header, and an unwrapped
        long header is what actually pushes these tables past \\textwidth."""
        escaped = tex_escape(text)
        if index in wrap_idx or len(text) <= 16:
            return f"\\textbf{{{escaped}}}"
        words, lines_out, current = escaped.split(" "), [], ""
        # More columns means less room per header, so wrap into more, shorter lines.
        divisor = 4 if len(header) >= 8 else (3 if len(escaped) >= 40 else 2)
        target = max(9, len(escaped) // divisor)
        for word in words:
            if current and len(current) + 1 + len(word) > target:
                lines_out.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            lines_out.append(current)
        alignment = "r" if is_num[index] else "l"
        return "\\makecell[%s]{%s}" % (alignment, "\\\\".join(f"\\textbf{{{ln}}}" for ln in lines_out))

    header_row = " & ".join(header_cell(i, h) for i, h in enumerate(header)) + " \\\\"
    body_rows = [" & ".join(cell_escape(c) for c in row) + " \\\\" for row in formatted]

    # A landscape table is rotated onto its own PDF page (pdflscape), which is the only way a wide
    # table fits without shrinking it to illegibility. Seven or more columns never fits portrait
    # here: the numeric columns' multi-line headers eat the width and the text column collapses to
    # one character per line.
    #
    # Inside a pdflscape `landscape`, \textwidth is NOT updated but \linewidth is, so \linewidth is
    # the only correct target. This previously hard-coded \paperheight-5cm, which happened to fit
    # the article-class proof and overflowed the real Springer class by exactly 150pt
    # (702.8pt requested against 552.7pt available in sn-jnl). The proof could never reveal it:
    # sn-jnl's text block is 372.0pt against the proof's 472.3pt.
    landscape = landscape or len(header) >= 7
    width = "\\linewidth" if landscape else "\\textwidth"

    lines = []
    if len(body) > LONG_TABLE_ROWS:
        if landscape:
            lines.append("\\begin{landscape}")
        lines.append(f"\\begingroup{size}")
        lines.append(f"\\begin{{xltabular}}{{{width}}}{{@{{}}{colspec}@{{}}}}")
        lines.append(f"\\caption{{{caption}}}\\label{{{label}}}\\\\")
        lines.append("\\toprule")
        lines.append(header_row)
        lines.append("\\midrule")
        lines.append("\\endfirsthead")
        lines.append("\\toprule")
        lines.append(header_row)
        lines.append("\\midrule")
        lines.append("\\endhead")
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{len(header)}}}{{r@{{}}}}{{\\itshape continued on next page}}\\\\")
        lines.append("\\endfoot")
        lines.append("\\bottomrule")
        lines.append("\\endlastfoot")
        lines.extend(body_rows)
        lines.append("\\end{xltabular}")
        lines.append("\\endgroup")
        if landscape:
            lines.append("\\end{landscape}")
    else:
        if landscape:
            lines.append("\\begin{landscape}")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(size)
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{{label}}}")
        lines.append(f"\\begin{{tabularx}}{{{width}}}{{@{{}}{colspec}@{{}}}}")
        lines.append("\\toprule")
        lines.append(header_row)
        lines.append("\\midrule")
        lines.extend(body_rows)
        lines.append("\\bottomrule")
        lines.append("\\end{tabularx}")
        lines.append("\\end{table}")
        if landscape:
            lines.append("\\end{landscape}")
    return "\n".join(lines) + "\n"


def main() -> int:
    if not CSV_DIR.exists():
        print(f"No tables at {CSV_DIR}; run render_manuscript_assets.py first.", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for stem, (label, landscape, digits, caption) in TABLE_SPEC.items():
        csv_path = CSV_DIR / f"{stem}.csv"
        if not csv_path.exists():
            print(f"  missing {csv_path.name}", file=sys.stderr)
            continue
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = [r for r in csv.reader(handle) if any(str(c).strip() for c in r)]
        if not rows:
            continue
        (OUT_DIR / f"{stem}.tex").write_text(build_table(rows, landscape, label, digits, caption), encoding="utf-8")
        written += 1
    print(f"Wrote {written} LaTeX tables to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
