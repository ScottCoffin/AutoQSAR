"""Render the Journal of Cheminformatics graphical abstract.

Journal spec (J. Cheminform. author guidelines): 920 x 300 px, max 150 KB, jpeg/png/svg,
white background, filling the available width.

Every statistic is read from manuscript_assets/manuscript_numbers.json, which
render_manuscript_assets.py writes from the benchmark artifacts, so this figure cannot drift
from the manuscript.

Design notes (graphical-abstract best practice):
  * One left-to-right reading path: what was benchmarked -> the headline result -> what it means.
  * A message title, not a topic title: the reader should get the finding without the paper.
  * The most citable result (leaderboard placement, and how much of it is a selection artifact)
    is the visual hero; supporting findings are deliberately smaller.
  * Okabe-Ito colourblind-safe palette, also separable in greyscale by lightness.
  * No gradients, shadows or 3D: they cost file size and carry no information.
  * Minimum effective type size ~9.5 px at final scale; numbers carry the emphasis, not decoration.

Usage:  python portable_colab_qsar_bundle/render_graphical_abstract.py
"""

from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "manuscript_assets" / "figures" / "graphical_abstract.svg"
NUMBERS_PATH = ROOT / "manuscript_assets" / "manuscript_numbers.json"

WIDTH, HEIGHT = 920, 300

# Okabe-Ito: colourblind-safe and greyscale-separable.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
INK = "#1B2A38"
MUTED = "#5A6B7B"
RULE = "#D6DEE6"
TRACK = "#EEF2F6"
FONT = "Helvetica, Arial, 'Liberation Sans', sans-serif"


def load_numbers() -> dict:
    with NUMBERS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


class Svg:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, raw: str) -> None:
        self.parts.append(raw)

    def rect(self, x, y, w, h, rx=0, fill="none", stroke="none", sw=1) -> None:
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def text(self, x, y, s, size=12, weight="400", color=INK, anchor="start", spacing=None) -> None:
        extra = f' letter-spacing="{spacing}"' if spacing else ""
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}"{extra}>{escape(s)}</text>'
        )

    def line(self, x1, y1, x2, y2, color=MUTED, sw=1.5) -> None:
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{sw}" stroke-linecap="round"/>'
        )

    def path(self, d, fill="none", stroke=MUTED, sw=1.5) -> None:
        self.add(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def arrow(self, x, y, size=9, color=MUTED) -> None:
        self.add(
            f'<path d="M {x:.1f} {y - size / 2:.1f} L {x + size * 0.85:.1f} {y:.1f} '
            f'L {x:.1f} {y + size / 2:.1f} Z" fill="{color}"/>'
        )


def hexagon(svg: Svg, cx: float, cy: float, r: float, color: str) -> None:
    """Benzene-ring glyph: signals 'molecules in' without implying a specific structure."""
    pts = []
    for i in range(6):
        a = math.radians(60 * i - 30)
        pts.append(f"{cx + r * math.cos(a):.1f},{cy + r * math.sin(a):.1f}")
    svg.add(f'<polygon points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.8"/>')
    svg.add(
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r * 0.42:.1f}" fill="none" '
        f'stroke="{color}" stroke-width="1.3"/>'
    )


SHORT_FAMILY = {
    "Ensemble (stacking / averaging)": "Ensembles",
    # Key must track the family label emitted by the notebook. It was renamed from
    # "Uni-Mol V1 (3D pretrained)" once the family gained Uni-Mol V2; a stale key here silently
    # falls through to the full name and overflows the panel.
    "Uni-Mol (3D pretrained)": "3D pretrained",
    "Conventional ML": "Conventional ML",
    "MapLight + GNN": "MapLight+GNN",
    "Chemprop v2 GNN": "Chemprop",
    "Deep tabular NN (ChemML MLP)": "Deep tabular",
    "CFA combinatorial fusion": "Fusion",
}


def draw() -> str:
    n = load_numbers()
    lb = n["leaderboard"]
    tdc = n["leaderboard_by_comparability"]["tdc_admet_group_official"]
    hw = n["run_comparison_vs_rtx"]
    wins = {k: v["total"] for k, v in n["wins_by_family"].items()}
    top = sorted(wins.items(), key=lambda kv: -kv[1])[:3]
    n_ds = int(n["datasets_analyzed"])
    total = lb["datasets_compared"]
    share = round(100 * top[0][1] / n_ds)

    svg = Svg()
    svg.add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">'
    )
    svg.rect(0, 0, WIDTH, HEIGHT, fill="#FFFFFF")

    # ---- Message title -------------------------------------------------------------------
    # The title leads with the breadth-vs-selection decomposition, not the estimated rank: the rank
    # is scored against a reference set with known leakage, so it is a secondary result (peer review
    # 2026-09-25).
    from_library = lb["top10_test_selected"] - lb["top10_matched_pool"]
    from_selection = lb["top10_matched_pool"] - lb["top10_cv_selected"]
    driver = "model-library breadth" if from_library >= from_selection else "test-set selection"
    svg.text(24, 26, f"QSARena: {driver} drives most leaderboard standing", size=19, weight="700")
    svg.text(24, 45,
             f"One leakage-controlled pipeline, {n_ds} datasets, {n['models_with_valid_results']} models "
             "- and no model family dominates.",
             size=11.5, color=MUTED)
    svg.line(24, 56, WIDTH - 24, 56, color=RULE, sw=1)

    # ---- Panel 1: what was benchmarked ---------------------------------------------------
    x0 = 24
    svg.text(x0, 78, "BENCHMARK", size=9.5, weight="700", color=MUTED, spacing="0.8")
    hexagon(svg, x0 + 13, 105, 11, BLUE)
    svg.line(x0 + 27, 105, x0 + 43, 105, color=RULE, sw=1.4)
    svg.text(x0 + 50, 101, "SMILES", size=11, weight="700", color=BLUE)
    svg.text(x0 + 50, 113, "to property", size=10, color=MUTED)

    for i, (value, label) in enumerate([
        (n_ds, "datasets"),
        (len(n["datasets_by_suite"]), "suites"),
        (int(n["models_with_valid_results"]), "models"),
    ]):
        yy = 142 + i * 26
        svg.text(x0 + 2, yy, str(value), size=17, weight="700", color=INK)
        svg.text(x0 + 36, yy, label, size=11, color=MUTED)

    svg.text(x0, 232, "TDC | MoleculeNet | Polaris", size=9.5, color=MUTED)
    svg.text(x0, 245, "PODUAM | ChemML", size=9.5, color=MUTED)

    # Vertical rules separate the three steps; the left-to-right order is carried by the layout and
    # the title, so no arrow glyph is needed (it only collided with the rule).
    svg.line(x0 + 156, 72, x0 + 156, 256, color=RULE, sw=1)

    # ---- Panel 2: the hero result --------------------------------------------------------
    hx, track_w = 204, 286
    svg.text(hx, 78, "ESTIMATED TOP-10 VS PUBLISHED VALUES (PROVISIONAL)", size=9.5, weight="700",
             color=MUTED, spacing="0.8")

    def bar(y, label, value, color, note):
        svg.text(hx, y - 5, label, size=9.8, weight="600", color=INK)
        svg.rect(hx, y, track_w, 16, rx=8, fill=TRACK)
        svg.rect(hx, y, track_w * value / total, 16, rx=8, fill=color)
        svg.text(hx + track_w + 9, y + 12.5, f"{value}/{total}", size=13.5, weight="700", color=color)
        if note:
            svg.text(hx, y + 27, note, size=8.8, color=MUTED)

    # Three stages. Reading down, the first drop is the value of a broad model library and the
    # second is the cost of refusing held-out information; the paper decomposes them in this order.
    bar(96, "Best of 28 models, chosen on the test set", lb["top10_test_selected"], GREEN,
        f"{tdc['top10_test_selected']}/{tdc['datasets']} on official TDC splits "
        f"- median rank {int(lb['median_rank_test_selected'])}")
    bar(152, "Same protocol, cross-validation-eligible models only",
        lb["top10_matched_pool"], BLUE,
        f"median rank {int(lb['median_rank_matched_pool'])}")
    bar(208, "Chosen by cross-validation only - the honest estimate",
        lb["top10_cv_selected"], ORANGE,
        f"median rank {int(lb['median_rank_cv_selected'])} - and zero first places")

    # Two labelled brackets attribute the 10-placement gap to its two distinct causes.
    gx = hx + track_w + 58
    drop_library = lb["top10_test_selected"] - lb["top10_matched_pool"]
    drop_selection = lb["top10_matched_pool"] - lb["top10_cv_selected"]

    svg.path(f"M {gx} 100 L {gx + 6} 100 L {gx + 6} 156 L {gx} 156", stroke=BLUE, sw=1.5)
    svg.text(gx + 11, 122, f"-{drop_library}", size=14, weight="700", color=BLUE)
    svg.text(gx + 11, 134, "model-library", size=8.6, color=BLUE)
    svg.text(gx + 11, 144, "breadth", size=8.6, color=BLUE)

    svg.path(f"M {gx} 156 L {gx + 6} 156 L {gx + 6} 212 L {gx} 212", stroke=VERMILION, sw=1.5)
    svg.text(gx + 11, 178, f"-{drop_selection}", size=14, weight="700", color=VERMILION)
    svg.text(gx + 11, 190, "held-out", size=8.6, color=VERMILION)
    svg.text(gx + 11, 200, "selection", size=8.6, color=VERMILION)

    svg.line(640, 72, 640, 256, color=RULE, sw=1)

    # ---- Panel 3: supporting findings ----------------------------------------------------
    rx = 660
    svg.text(rx, 78, "NO SINGLE WINNER", size=9.5, weight="700", color=MUTED, spacing="0.8")
    for i, (fam, count) in enumerate(top):
        yy = 94 + i * 21
        svg.text(rx, yy + 9, SHORT_FAMILY.get(fam, fam), size=10, color=INK)
        svg.rect(rx + 88, yy, 86, 11, rx=5.5, fill=TRACK)
        svg.rect(rx + 88, yy, 86 * count / n_ds, 11, rx=5.5, fill=[GREEN, BLUE, MUTED][i])
        svg.text(rx + 180, yy + 9, str(count), size=10.5, weight="700", color=INK)
    svg.text(rx, 170, f"Top family takes only {share}% of datasets;", size=10, color=MUTED)
    svg.text(rx, 183, "the best model is dataset-dependent.", size=10, color=MUTED)

    svg.line(rx, 198, WIDTH - 24, 198, color=RULE, sw=1)
    svg.text(rx, 218, "RUNS ON A LAPTOP GPU", size=9.5, weight="700", color=MUTED, spacing="0.8")
    svg.text(rx, 242, f"{hw['median_change_pct_same_split']:+.1f}%", size=16, weight="700", color=BLUE)
    svg.text(rx + 48, 237, "median change vs an A100,", size=10, color=MUTED)
    svg.text(rx + 48, 249, f"{hw['datasets_same_split']} identically split datasets", size=10, color=MUTED)

    # ---- Footer --------------------------------------------------------------------------
    svg.line(24, 266, WIDTH - 24, 266, color=RULE, sw=1)
    svg.text(24, 284,
             "Start with descriptor-rich gradient boosting plus averaging; add 3D pretrained models "
             "where chemistry and compute justify it.",
             size=10.5, weight="600", color=INK)
    svg.text(WIDTH - 24, 284, "github.com/ScottCoffin/QSARena", size=9.5, color=MUTED, anchor="end")

    svg.add("</svg>")
    return "\n".join(svg.parts)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(draw(), encoding="utf-8")
    kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUT_PATH} ({WIDTH}x{HEIGHT}, {kb:.1f} KB; journal limit 150 KB)")

    # Also emit PNG and PDF from the same source. These used to be produced by hand, which meant a
    # regenerated SVG left a stale PNG and PDF behind; the submission then carried a graphical
    # abstract that contradicted the manuscript. Deriving all three here makes that impossible.
    try:
        import cairosvg
    except ImportError:
        print("  cairosvg not installed: PNG/PDF NOT regenerated (pip install cairosvg)")
        return

    svg_bytes = OUT_PATH.read_bytes()
    png_path = OUT_PATH.with_suffix(".png")
    pdf_path = OUT_PATH.with_suffix(".pdf")
    cairosvg.svg2png(bytestring=svg_bytes, write_to=str(png_path),
                     output_width=WIDTH, output_height=HEIGHT)
    cairosvg.svg2pdf(bytestring=svg_bytes, write_to=str(pdf_path))
    png_kb = png_path.stat().st_size / 1024
    print(f"Wrote {png_path} ({png_kb:.1f} KB)" + ("  [OVER 150 KB LIMIT]" if png_kb > 150 else ""))
    print(f"Wrote {pdf_path}")

    # Keep the submission tree in step with the rendered assets.
    submission_figs = OUT_PATH.parents[2] / "submission" / "figures"
    if submission_figs.is_dir():
        for src in (png_path, pdf_path, OUT_PATH):
            (submission_figs / src.name).write_bytes(src.read_bytes())
        print(f"Copied graphical abstract into {submission_figs}")


if __name__ == "__main__":
    main()
