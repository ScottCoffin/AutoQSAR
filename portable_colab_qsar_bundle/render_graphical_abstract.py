from __future__ import annotations

import math
from html import escape
from pathlib import Path
from textwrap import wrap


OUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "manuscript_assets"
    / "figures"
    / "graphical_abstract.svg"
)

WIDTH = 1800
HEIGHT = 900


def attrs(**kwargs: object) -> str:
    parts: list[str] = []
    for key, value in kwargs.items():
        if value is None:
            continue
        key = key.rstrip("_").replace("_", "-")
        parts.append(f'{key}="{escape(str(value), quote=True)}"')
    return " ".join(parts)


class Svg:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, raw: str) -> None:
        self.parts.append(raw)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        rx: float = 20,
        fill: str = "#ffffff",
        stroke: str = "#d7dee8",
        sw: float = 2,
        extra: str = "",
    ) -> None:
        self.add(
            "<rect "
            + attrs(x=x, y=y, width=w, height=h, rx=rx, fill=fill, stroke=stroke, stroke_width=sw)
            + (f" {extra}" if extra else "")
            + "/>"
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = "#617083",
        sw: float = 3,
        arrow: bool = False,
        dash: str | None = None,
    ) -> None:
        self.add(
            "<line "
            + attrs(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                stroke=color,
                stroke_width=sw,
                stroke_linecap="round",
                marker_end="url(#arrow)" if arrow else None,
                stroke_dasharray=dash,
            )
            + "/>"
        )

    def path(
        self,
        d: str,
        *,
        fill: str = "none",
        stroke: str = "#617083",
        sw: float = 3,
        arrow: bool = False,
        dash: str | None = None,
    ) -> None:
        self.add(
            "<path "
            + attrs(
                d=d,
                fill=fill,
                stroke=stroke,
                stroke_width=sw,
                stroke_linecap="round",
                stroke_linejoin="round",
                marker_end="url(#arrow)" if arrow else None,
                stroke_dasharray=dash,
            )
            + "/>"
        )

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        *,
        fill: str = "#ffffff",
        stroke: str = "#d7dee8",
        sw: float = 2,
    ) -> None:
        self.add(
            "<circle "
            + attrs(cx=cx, cy=cy, r=r, fill=fill, stroke=stroke, stroke_width=sw)
            + "/>"
        )

    def polygon(
        self,
        points: list[tuple[float, float]],
        *,
        fill: str = "none",
        stroke: str = "#617083",
        sw: float = 3,
    ) -> None:
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        self.add(
            "<polygon "
            + attrs(points=point_text, fill=fill, stroke=stroke, stroke_width=sw, stroke_linejoin="round")
            + "/>"
        )

    def text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        size: float = 24,
        weight: str = "400",
        color: str = "#18202c",
        anchor: str = "start",
        max_chars: int | None = None,
        line_height: float = 1.22,
        italic: bool = False,
    ) -> None:
        lines = text.split("\n")
        if max_chars:
            wrapped: list[str] = []
            for line in lines:
                wrapped.extend(wrap(line, max_chars) or [""])
            lines = wrapped
        style = f"font-size:{size}px;font-weight:{weight};fill:{color};font-family:Arial,Helvetica,sans-serif;"
        if italic:
            style += "font-style:italic;"
        self.add(f'<text {attrs(x=x, y=y, text_anchor=anchor, style=style)}>')
        for idx, line in enumerate(lines):
            dy = 0 if idx == 0 else size * line_height
            self.add(f'<tspan {attrs(x=x, dy=dy)}>{escape(line)}</tspan>')
        self.add("</text>")

    def pill(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        *,
        fill: str,
        stroke: str,
        color: str = "#18202c",
        size: float = 17,
    ) -> None:
        self.rect(x, y, w, h, rx=h / 2, fill=fill, stroke=stroke, sw=1.6)
        self.text(x + w / 2, y + h / 2 + size * 0.34, label, size=size, weight="700", color=color, anchor="middle")


def draw_molecule(svg: Svg, cx: float, cy: float, r: float) -> None:
    points = [
        (cx + r * math.cos(math.pi / 6 + i * math.pi / 3), cy + r * math.sin(math.pi / 6 + i * math.pi / 3))
        for i in range(6)
    ]
    svg.polygon(points, stroke="#2f7ebc", sw=4)
    for i, j in [(0, 1), (2, 3), (4, 5)]:
        svg.line(points[i][0], points[i][1], points[j][0], points[j][1], color="#2f7ebc", sw=3)
    svg.circle(cx + r * 1.42, cy - r * 0.58, 10, fill="#ffffff", stroke="#2f7ebc", sw=3)
    svg.line(cx + r * 0.83, cy - r * 0.50, cx + r * 1.23, cy - r * 0.56, color="#2f7ebc", sw=3)


def stat_card(svg: Svg, x: float, y: float, w: float, h: float, big: str, label: str, color: str) -> None:
    svg.rect(x, y, w, h, rx=18, fill="#ffffff", stroke="#d9e2ec", sw=1.6)
    svg.text(x + w / 2, y + 42, big, size=34, weight="800", color=color, anchor="middle")
    svg.text(x + w / 2, y + 72, label, size=16, weight="700", color="#465568", anchor="middle", max_chars=15)


def model_box(svg: Svg, x: float, y: float, w: float, h: float, label: str, color: str) -> None:
    svg.rect(x, y, w, h, rx=14, fill="#ffffff", stroke=color, sw=2)
    svg.circle(x + 20, y + h / 2, 7, fill=color, stroke=color, sw=1)
    svg.text(x + 36, y + h / 2 + 6, label, size=16, weight="700", color="#263241")


def win_bar(svg: Svg, x: float, y: float, label: str, value: int, color: str) -> None:
    svg.text(x, y + 16, label, size=16, weight="700", color="#263241")
    svg.rect(x + 140, y, 210, 19, rx=9.5, fill="#eef2f7", stroke="#dce3ec", sw=1)
    svg.rect(x + 140, y, 210 * value / 45, 19, rx=9.5, fill=color, stroke=color, sw=1)
    svg.text(x + 365, y + 16, f"{value}/45", size=16, weight="800", color="#263241")


def top10_bar(svg: Svg, x: float, y: float, label: str, value: int, color: str) -> None:
    svg.text(x, y + 18, label, size=17, weight="800", color="#263241")
    svg.rect(x, y + 32, 410, 26, rx=13, fill="#eef2f7", stroke="#dce3ec", sw=1)
    svg.rect(x, y + 32, 410 * value / 37, 26, rx=13, fill=color, stroke=color, sw=1)
    svg.text(x + 430, y + 53, f"{value} of 37", size=19, weight="800", color=color)
    svg.text(x + 430, y + 78, "datasets in published-reference top 10", size=13, weight="700", color="#6a7788")


def draw_svg() -> str:
    svg = Svg()
    svg.add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">')
    svg.add(
        """
<defs>
  <marker id="arrow" markerWidth="14" markerHeight="14" refX="10" refY="5" orient="auto" markerUnits="strokeWidth">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#617083"/>
  </marker>
  <linearGradient id="engineGrad" x1="0" x2="1" y1="0" y2="1">
    <stop offset="0%" stop-color="#e8f4ff"/>
    <stop offset="100%" stop-color="#f2fbf6"/>
  </linearGradient>
  <filter id="softShadow" x="-5%" y="-5%" width="110%" height="120%">
    <feDropShadow dx="0" dy="8" stdDeviation="8" flood-color="#1b2a3a" flood-opacity="0.11"/>
  </filter>
</defs>
"""
    )
    svg.rect(0, 0, WIDTH, HEIGHT, rx=0, fill="#f7fafc", stroke="#f7fafc", sw=0)
    svg.text(64, 64, "AutoQSAR: no single architecture wins", size=34, weight="800", color="#172033")
    svg.text(
        64,
        100,
        "A leakage-controlled benchmark maps when conventional, ensemble, and pretrained molecular models are worth using.",
        size=21,
        color="#4a586b",
    )

    # Left panel.
    left_x, left_y, left_w, left_h = 60, 145, 360, 640
    svg.rect(left_x, left_y, left_w, left_h, rx=26, fill="#ffffff", stroke="#cfd9e6", sw=2.2, extra='filter="url(#softShadow)"')
    svg.text(left_x + 28, left_y + 48, "Benchmark set", size=28, weight="800", color="#1b3045")
    draw_molecule(svg, left_x + 91, left_y + 120, 43)
    svg.text(left_x + 160, left_y + 104, "SMILES", size=24, weight="800", color="#2f7ebc")
    svg.text(left_x + 160, left_y + 134, "to molecular\nproperty targets", size=18, color="#526173", max_chars=20)
    stat_card(svg, left_x + 28, left_y + 205, 140, 94, "45", "datasets", "#2f7ebc")
    stat_card(svg, left_x + 192, left_y + 205, 140, 94, "5", "suites", "#2f7ebc")
    stat_card(svg, left_x + 28, left_y + 323, 140, 94, "23", "regression", "#7b54b8")
    stat_card(svg, left_x + 192, left_y + 323, 140, 94, "22", "classification", "#1f8f69")
    stat_card(svg, left_x + 110, left_y + 441, 140, 94, "25", "model variants", "#d28a0b")
    for idx, label in enumerate(["TDC", "Polaris", "MoleculeNet", "PODUAM", "ChemML"]):
        x = left_x + 31 + (idx % 2) * 156
        y = left_y + 565 + (idx // 2) * 36
        svg.pill(x, y, 132, 28, label, fill="#eef6ff", stroke="#b9d4ee", color="#2f587e", size=14)

    # Middle panel.
    mid_x, mid_y, mid_w, mid_h = 485, 145, 590, 640
    svg.rect(mid_x, mid_y, mid_w, mid_h, rx=26, fill="#ffffff", stroke="#cfd9e6", sw=2.2, extra='filter="url(#softShadow)"')
    svg.text(mid_x + 30, mid_y + 48, "AutoQSAR comparison engine", size=28, weight="800", color="#1b3045")
    svg.rect(mid_x + 28, mid_y + 78, mid_w - 56, 330, rx=24, fill="url(#engineGrad)", stroke="#bdd7ee", sw=2)
    svg.text(mid_x + 48, mid_y + 116, "Base model library", size=22, weight="800", color="#1b3045")
    base_models = [
        ("Conventional ML", "#1f8f69"),
        ("Deep tabular NN", "#7b54b8"),
        ("Chemprop GNN", "#7b54b8"),
        ("TabPFN", "#7b54b8"),
        ("Uni-Mol V1", "#7b54b8"),
        ("MapLight + GNN", "#7b54b8"),
    ]
    for idx, (label, color) in enumerate(base_models):
        col = idx % 2
        row = idx // 2
        model_box(svg, mid_x + 50 + col * 252, mid_y + 145 + row * 70, 220, 44, label, color)
    svg.line(mid_x + 295, mid_y + 362, mid_x + 295, mid_y + 440, color="#617083", sw=3, arrow=True)
    svg.rect(mid_x + 78, mid_y + 440, 434, 78, rx=18, fill="#ffffff", stroke="#c4d8ec", sw=2)
    svg.text(mid_x + 295, mid_y + 473, "Aligned train/test predictions", size=22, weight="800", color="#263241", anchor="middle")
    svg.text(mid_x + 295, mid_y + 499, "same splits and metrics for every dataset", size=16, color="#526173", anchor="middle")
    svg.line(mid_x + 295, mid_y + 518, mid_x + 295, mid_y + 560, color="#617083", sw=3, arrow=True)
    svg.rect(mid_x + 58, mid_y + 560, 210, 70, rx=18, fill="#fff8e9", stroke="#e7bd62", sw=2)
    svg.text(mid_x + 163, mid_y + 590, "CFA fusion", size=21, weight="800", color="#7b4d00", anchor="middle")
    svg.text(mid_x + 163, mid_y + 615, "post-model", size=15, weight="700", color="#7b4d00", anchor="middle")
    svg.rect(mid_x + 322, mid_y + 560, 210, 70, rx=18, fill="#eefaf5", stroke="#a9d7c5", sw=2)
    svg.text(mid_x + 427, mid_y + 590, "Ensemble layer", size=21, weight="800", color="#176c50", anchor="middle")
    svg.text(mid_x + 427, mid_y + 615, "post-model", size=15, weight="700", color="#176c50", anchor="middle")
    svg.path(f"M {mid_x + 295} {mid_y + 538} C {mid_x + 240} {mid_y + 548}, {mid_x + 190} {mid_y + 548}, {mid_x + 163} {mid_y + 560}", stroke="#617083", sw=2.5, arrow=True)
    svg.path(f"M {mid_x + 295} {mid_y + 538} C {mid_x + 350} {mid_y + 548}, {mid_x + 400} {mid_y + 548}, {mid_x + 427} {mid_y + 560}", stroke="#617083", sw=2.5, arrow=True)
    svg.text(mid_x + 295, mid_y + 682, "Model choice is evaluated as a pipeline output.", size=18, weight="700", color="#405268", anchor="middle")

    svg.line(left_x + left_w + 25, 465, mid_x - 26, 465, color="#617083", sw=4, arrow=True)

    # Right panel.
    right_x, right_y, right_w, right_h = 1138, 145, 602, 640
    svg.rect(right_x, right_y, right_w, right_h, rx=26, fill="#ffffff", stroke="#cfd9e6", sw=2.2, extra='filter="url(#softShadow)"')
    svg.text(right_x + 30, right_y + 48, "Benchmark findings", size=28, weight="800", color="#1b3045")

    card_x, card_y, card_w = right_x + 28, right_y + 82, 546
    svg.rect(card_x, card_y, card_w, 174, rx=20, fill="#fbfdff", stroke="#d7e3ef", sw=1.8)
    svg.text(card_x + 22, card_y + 37, "1. Winner distribution is broad", size=22, weight="800", color="#1b3045")
    win_bar(svg, card_x + 22, card_y + 62, "Ensemble", 15, "#1f8f69")
    win_bar(svg, card_x + 22, card_y + 92, "Conventional", 11, "#55a37f")
    win_bar(svg, card_x + 22, card_y + 122, "Uni-Mol V1", 7, "#7b54b8")
    svg.text(card_x + 390, card_y + 158, "No family exceeds one-third of datasets.", size=16, weight="800", color="#7b4d00", anchor="middle")

    card_y2 = card_y + 196
    svg.rect(card_x, card_y2, card_w, 154, rx=20, fill="#fbfdff", stroke="#d7e3ef", sw=1.8)
    svg.text(card_x + 22, card_y2 + 37, "2. Best default depends on task", size=22, weight="800", color="#1b3045")
    svg.rect(card_x + 24, card_y2 + 59, 238, 66, rx=16, fill="#eefaf5", stroke="#b8dece", sw=1.5)
    svg.text(card_x + 143, card_y2 + 86, "Classification", size=19, weight="800", color="#176c50", anchor="middle")
    svg.text(card_x + 143, card_y2 + 111, "20/22 wins: ensemble or conventional", size=14, weight="700", color="#405268", anchor="middle")
    svg.rect(card_x + 286, card_y2 + 59, 238, 66, rx=16, fill="#f4f0fb", stroke="#d5c8eb", sw=1.5)
    svg.text(card_x + 405, card_y2 + 86, "Regression", size=19, weight="800", color="#664399", anchor="middle")
    svg.text(card_x + 405, card_y2 + 111, "heterogeneous; 3D wins selectively", size=14, weight="700", color="#405268", anchor="middle")
    svg.text(card_x + 274, card_y2 + 142, "Uni-Mol V1: about 115x median conventional-model cost", size=15, weight="800", color="#7b4d00", anchor="middle")

    card_y3 = card_y2 + 176
    svg.rect(card_x, card_y3, card_w, 186, rx=20, fill="#fbfdff", stroke="#d7e3ef", sw=1.8)
    svg.text(card_x + 22, card_y3 + 37, "3. Published-reference top-10 placement", size=22, weight="800", color="#1b3045")
    top10_bar(svg, card_x + 24, card_y3 + 62, "Best model chosen with test set", 35, "#1f8f69")
    top10_bar(svg, card_x + 24, card_y3 + 122, "Model chosen by cross-validation", 26, "#d28a0b")
    svg.text(card_x + 274, card_y3 + 176, "Honest model selection costs about 9 top-10 placements.", size=15, weight="800", color="#7b4d00", anchor="middle")

    svg.line(mid_x + mid_w + 25, 465, right_x - 26, 465, color="#617083", sw=4, arrow=True)

    svg.text(
        900,
        846,
        "Practical takeaway: start with descriptor-rich gradient boosting plus averaging; add 3D pretrained models only for regression tasks where chemistry and compute justify it.",
        size=20,
        weight="700",
        color="#263241",
        anchor="middle",
        max_chars=128,
    )
    svg.add("</svg>")
    return "\n".join(svg.parts)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(draw_svg(), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
