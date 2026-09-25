from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_PATH = Path(__file__).with_name("colab_qsar_workflow_map.png")


def draw_box(ax, xy, text, *, width=2.4, height=0.62, face="#eef6ff", edge="#2f5d8c", fontsize=8.5):
    x, y = xy
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.035,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color="#182533", wrap=True)


def draw_arrow(ax, start, end, *, color="#65758b", rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.3,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(15.5, 7.5), dpi=180)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.text(
        0.35,
        7.12,
        "AutoQSAR Colab Workflow",
        fontsize=18,
        weight="bold",
        color="#172033",
        ha="left",
        va="center",
    )
    ax.text(
        0.35,
        6.78,
        "Run the core path first. Prediction and advanced branches are optional after the baseline plots look reasonable.",
        fontsize=10,
        color="#475569",
        ha="left",
        va="center",
    )

    core_face = "#eaf4ff"
    core_edge = "#2563a9"
    diag_face = "#f1f7ed"
    diag_edge = "#4f7f39"
    optional_face = "#fff6df"
    optional_edge = "#a66a00"

    core_nodes = [
        ((1.25, 5.85), "0\nSetup\n2-12 min"),
        ((4.0, 5.85), "1A-1C\nLoad, clean,\ntransform target\n15 sec-6 min"),
        ((6.75, 5.85), "2A-2B\nPreview and\nbuild features\n1-10 min"),
        ((9.5, 5.85), "4A-4B\nSplit and optional\nfeature selection\n1-25 min"),
        ((12.25, 5.85), "4C-4D\nTrain baseline and\ninspect predictions\n2-30 min"),
    ]
    for pos, label in core_nodes:
        draw_box(ax, pos, label, face=core_face, edge=core_edge, width=2.25)
    for (start, _), (end, _) in zip(core_nodes, core_nodes[1:]):
        draw_arrow(ax, (start[0] + 1.14, start[1]), (end[0] - 1.14, end[1]), color=core_edge)

    draw_box(ax, (6.75, 4.55), "3A\nSimilarity map\n30 sec-8 min\nrecommended diagnostic", face=diag_face, edge=diag_edge)
    draw_arrow(ax, (6.75, 5.53), (6.75, 4.98), color=diag_edge)

    optional_nodes = [
        ((1.65, 2.75), "4E-4G\nGA tuning\n10 min-2+ hr"),
        ((4.2, 2.75), "5A-5C\nDeep learning\n5-60+ min"),
        ((6.75, 2.75), "6A-6H\nUni-Mol and\nChemprop\n10 min-3+ hr"),
        ((9.3, 2.75), "7A-7B\nEnsembles\n1-10 min"),
        ((11.85, 2.75), "8A-8C\nExplanation\n1-20 min"),
        ((14.4, 2.75), "9A-9D\nPredict, map,\nand AD\n1-60+ min"),
    ]
    for pos, label in optional_nodes:
        draw_box(ax, pos, label, face=optional_face, edge=optional_edge, width=2.18, height=0.82, fontsize=8.2)

    ax.text(
        0.65,
        3.62,
        "Optional / advanced branches",
        fontsize=12,
        weight="bold",
        color="#7c4a00",
        ha="left",
    )
    draw_arrow(ax, (12.25, 5.5), (1.65, 3.18), color=optional_edge, rad=0.18)
    draw_arrow(ax, (12.25, 5.5), (4.2, 3.18), color=optional_edge, rad=0.12)
    draw_arrow(ax, (12.25, 5.5), (6.75, 3.18), color=optional_edge, rad=0.06)
    draw_arrow(ax, (12.25, 5.5), (9.3, 3.18), color=optional_edge, rad=-0.03)
    draw_arrow(ax, (12.25, 5.5), (11.85, 3.18), color=optional_edge, rad=-0.08)
    draw_arrow(ax, (12.25, 5.5), (14.4, 3.18), color=optional_edge, rad=-0.12)

    # Redraw the diagnostic layer above optional arrows so its label stays readable.
    draw_box(ax, (6.75, 4.55), "3A\nSimilarity map\n30 sec-8 min\nrecommended diagnostic", face=diag_face, edge=diag_edge)
    draw_arrow(ax, (6.75, 5.53), (6.75, 4.98), color=diag_edge)

    ax.text(
        0.65,
        0.48,
        "Ranges are approximate Colab runtimes. Blue = necessary core path   Green = recommended diagnostic   Gold = optional or resource-heavy",
        fontsize=9.5,
        color="#475569",
        ha="left",
    )

    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
