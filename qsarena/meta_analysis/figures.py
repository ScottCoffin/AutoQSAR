"""Meta-analysis figures, style-matched to the manuscript export cell (fonts, sizes, family colours).

Every figure is written as PDF (vector, TrueType fonts) and 600-dpi PNG. Each function returns the
numbers it annotates, and those go into ``meta_numbers.json`` under ``figures``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qsarena.meta_analysis import stats as st  # noqa: E402
from qsarena.meta_analysis.gap_matrix import FAMILY_COLORS, GROUP_ORDER  # noqa: E402

#: Same rcParams as the notebook's MANUSCRIPT_FIGURE_EXPORT cell.
MANUSCRIPT_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.hashsalt": "qsarena-meta",
}
DPI = 600
GAP_CAP = 50.0
SINGLE_SEED_NOTE = (
    "Single split, single seed per dataset; bands are dataset-bootstrap 95% intervals, not seed variance."
)


def _save(fig, fig_dir: Path, stem: str) -> list[str]:
    paths = []
    for fmt in ("pdf", "png"):
        path = fig_dir / f"{stem}.{fmt}"
        metadata = {"CreationDate": None} if fmt == "pdf" else {"Software": None}
        fig.savefig(path, dpi=DPI, metadata=metadata)
        paths.append(str(path.name))
    plt.close(fig)
    return paths


def figure_m1(results: dict, fig_dir: Path) -> dict:
    gap = results["gap_pct"]
    logn = results["catalog"].set_index("dataset").loc[gap.index, "log10_n_train"].astype(float)
    families = results["trend_families"]
    excluded = [f for f in gap.columns if f not in families]
    grid = np.linspace(logn.min(), logn.max(), 60)
    notes = {"slopes_pp_per_decade": {}}
    with plt.rc_context(MANUSCRIPT_RC):
        fig = plt.figure(figsize=(10.0, 5.2))
        outer = fig.add_gridspec(1, 2, width_ratios=[2.25, 1.0], wspace=0.28)
        inner = outer[0].subgridspec(2, 4, hspace=0.45, wspace=0.12)
        for k, family in enumerate(families):
            ax = fig.add_subplot(inner[k // 4, k % 4])
            y = gap[family]
            color = FAMILY_COLORS.get(family, "#555555")
            band = st.trend_band(logn, y, grid, seed=0)
            notes["slopes_pp_per_decade"][family] = band["slope"]
            ax.fill_between(
                grid, np.clip(band["low"], 0, GAP_CAP), np.clip(band["high"], 0, GAP_CAP), color=color, alpha=0.25, lw=0
            )
            ax.plot(grid, np.clip(band["fit"], 0, GAP_CAP), color=color, lw=1.4)
            ax.scatter(logn, y.clip(upper=GAP_CAP), s=9, color=color, edgecolor="black", linewidth=0.3, zorder=3)
            ax.set_ylim(-2, GAP_CAP + 3)
            short = family.split(" (")[0]
            ax.set_title(f"{short}\n{band['slope']:+.1f} pp per decade", fontsize=7.2, fontweight="normal")
            if k % 4:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("Gap to best (%, cap 50)", fontsize=7.5)
            if k // 4 == 1 or k + 4 >= len(families):
                ax.set_xlabel("log10(n train)", fontsize=7.5)
            ax.tick_params(labelsize=6.5)
        fig.text(0.02, 0.97, "a", fontsize=12, fontweight="bold")
        if excluded:
            fig.text(0.02, 0.005, f"Excluded (valid on < 50% of datasets): {', '.join(excluded)}", fontsize=6.5)

        ax = fig.add_subplot(outer[1])
        cross = results["crossover_primary"]
        a_name, b_name = "Conventional ML", "Uni-Mol (3D pretrained)"
        for name in (a_name, b_name):
            color = FAMILY_COLORS[name]
            band = st.trend_band(logn, gap[name], grid, seed=0)
            ax.fill_between(grid, band["low"], band["high"], color=color, alpha=0.22, lw=0)
            ax.plot(grid, band["fit"], color=color, lw=1.8, label=name)
        if np.isfinite(cross["crossover_n"]):
            x0 = np.log10(cross["crossover_n"])
            ax.axvline(x0, color="black", lw=1, ls="--")
            if np.isfinite(cross["ci_low_n"]):
                ax.axvspan(np.log10(cross["ci_low_n"]), np.log10(cross["ci_high_n"]), color="#999999", alpha=0.18)
            label = f"crossover ≈ {cross['crossover_n']:,.0f}"
        else:
            label = "no crossover in range"
        ax.set_title(
            f"{label}\n({100 * cross['p_cross_in_range']:.0f}% of bootstrap replicates cross)",
            fontsize=7.5,
            fontweight="normal",
        )
        ax.set_xlabel("log10(n train)")
        ax.set_ylabel("Fitted gap to best (%)")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13))
        fig.text(0.70, 0.97, "b", fontsize=12, fontweight="bold")
        fig.suptitle(SINGLE_SEED_NOTE, y=-0.08, x=0.5, fontsize=6.5, fontweight="normal", style="italic")
        notes["files"] = _save(fig, fig_dir, "figureM1_size_crossover")
    notes["crossover_n"] = cross["crossover_n"]
    return notes


def figure_m2(results: dict, fig_dir: Path) -> dict:
    diff = results["difficulty"]
    frame = diff["frame"]
    nat = results["natural_experiment"]
    notes = {}
    split_colors = {"predefined": "#0072B2", "scaffold": "#D55E00", "target_quartiles": "#009E73", "random": "#CC79A7"}
    with plt.rc_context(MANUSCRIPT_RC):
        fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.5], "wspace": 0.35})
        for ax, task, metric in [
            (axes[0], "classification", "Best test ROC-AUC"),
            (axes[1], "regression", "Best test R²"),
        ]:
            sub = frame[frame["task"] == task]
            for split, group in sub.groupby("split_strategy"):
                ax.scatter(
                    group["mean_snn"],
                    group["achievable_best"],
                    s=16,
                    label=split,
                    color=split_colors.get(split, "#777777"),
                    edgecolor="black",
                    linewidth=0.3,
                )
            rec = diff["per_feature"]["mean_snn"][task]
            notes[f"rho_snn_{task}"] = rec["rho"]
            ax.set_title(
                f"{task.capitalize()}: ρ = {rec['rho']:.2f} [{rec['ci'][0]:.2f}, {rec['ci'][1]:.2f}]",
                fontsize=7.5,
                fontweight="normal",
            )
            ax.set_xlabel("Mean SNN (test → train Tanimoto)")
            ax.set_ylabel(metric)
        handles = [
            plt.Line2D([], [], ls="", marker="o", ms=4, mfc=c, mec="black", mew=0.3, label=k.replace("_", " "))
            for k, c in split_colors.items()
            if k in set(frame["split_strategy"])
        ]
        axes[0].legend(handles=handles, fontsize=6.5, title="split", title_fontsize=6.5, loc="lower left")
        fig.text(0.07, 0.95, "a", fontsize=12, fontweight="bold")

        ax = axes[2]
        if isinstance(nat, pd.DataFrame) and not nat.empty:
            pair_counts = nat["family"].value_counts()
            # Same rule as the text: families valid on fewer than 3 re-split datasets in both runs are left out.
            order = [f for f in FAMILY_COLORS if pair_counts.get(f, 0) >= 3]
            for i, family in enumerate(order):
                g = nat[nat["family"] == family]
                color = FAMILY_COLORS[family]
                for _, row in g.iterrows():
                    ax.plot(
                        [i - 0.18, i + 0.18],
                        [min(row["gap_low_shift_pct"], GAP_CAP), min(row["gap_scaffold_pct"], GAP_CAP)],
                        color=color,
                        alpha=0.5,
                        lw=0.8,
                    )
                med_before, med_after = g["gap_low_shift_pct"].median(), g["gap_scaffold_pct"].median()
                ax.plot(
                    [i - 0.18, i + 0.18],
                    [min(med_before, GAP_CAP), min(med_after, GAP_CAP)],
                    color="black",
                    lw=2.0,
                    marker="o",
                    ms=3,
                )
                notes.setdefault("natural_experiment_median_delta_pp", {})[family] = float(med_after - med_before)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([f.split(" (")[0] for f in order], rotation=60, ha="right", fontsize=6.5)
            ax.set_ylabel("Gap to best (%, cap 50)")
            n = nat["dataset"].nunique()
            ax.set_title(
                f"Same chemistry, low-shift split (left) → scaffold split (right); {n} datasets",
                fontsize=7.2,
                fontweight="normal",
            )
        else:
            ax.text(0.5, 0.5, "comparison run not available", ha="center", transform=ax.transAxes)
        fig.text(0.49, 0.95, "b", fontsize=12, fontweight="bold")
        notes["files"] = _save(fig, fig_dir, "figureM2_shift_difficulty")
    return notes


def figure_m3(results: dict, fig_dir: Path) -> dict:
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    from qsarena.meta_analysis.pipeline import RECOMMENDER_FEATURES

    rec = results["recommender"]
    cat = results["catalog"].set_index("dataset")
    target = results["winners"].set_index("dataset")["winner_group"].loc[cat.index]
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=0)
    tree.fit(cat[RECOMMENDER_FEATURES].to_numpy(dtype=float), target.to_numpy())
    notes = {}
    with plt.rc_context(MANUSCRIPT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [2.8, 1], "wspace": 0.2})
        short = [c.split(" (")[0].replace("Descriptor-based ML", "Descriptor ML") for c in tree.classes_]
        plot_tree(
            tree,
            feature_names=RECOMMENDER_FEATURES,
            class_names=short,
            filled=False,
            label="root",
            impurity=False,
            proportion=False,
            fontsize=6,
            ax=axes[0],
        )
        axes[0].set_title(
            f"Depth-3 tree fitted on all datasets (exploratory); value = datasets per class: {', '.join(short)}",
            fontsize=7.5,
            fontweight="normal",
        )
        fig.text(0.02, 0.95, "a", fontsize=12, fontweight="bold")

        ax = axes[1]
        labels, values, lows, highs, colors = [], [], [], [], []
        for name, color in (("decision_tree", "#0072B2"), ("l1_logistic", "#009E73")):
            m = rec["models"][name]
            labels.append({"decision_tree": "decision tree", "l1_logistic": "L1 logistic"}[name])
            values.append(m["lodo_balanced_accuracy"])
            lows.append(m["lodo_balanced_accuracy_ci"][0])
            highs.append(m["lodo_balanced_accuracy_ci"][1])
            colors.append(color)
            notes[f"lodo_bacc_{name}"] = m["lodo_balanced_accuracy"]
        labels.append("majority class")
        values.append(rec["majority_balanced_accuracy"])
        lows.append(np.nan)
        highs.append(np.nan)
        colors.append("#999999")
        x = np.arange(len(labels))
        ax.bar(x, values, color=colors, width=0.6)
        err = np.array(
            [
                [v - lo if np.isfinite(lo) else 0 for v, lo in zip(values, lows)],
                [hi - v if np.isfinite(hi) else 0 for v, hi in zip(values, highs)],
            ]
        )
        ax.errorbar(x, values, yerr=err, fmt="none", ecolor="black", capsize=3, lw=0.8)
        null = np.mean([rec["models"][n]["permutation_null_balanced_accuracy_mean"] for n in rec["models"]])
        ax.axhline(null, color="black", ls=":", lw=1)
        ax.text(len(labels) - 0.5, null + 0.01, "permutation null", ha="right", fontsize=6.5)
        ax.axhline(1 / len(GROUP_ORDER), color="#D55E00", ls="--", lw=0.8)
        ax.text(len(labels) - 0.5, 1 / len(GROUP_ORDER) + 0.01, "chance", ha="right", fontsize=6.5, color="#D55E00")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, fontsize=7)
        ax.set_ylabel("Leave-one-dataset-out balanced accuracy")
        ax.set_ylim(0, 1)
        fig.text(0.68, 0.95, "b", fontsize=12, fontweight="bold")
        notes["majority_bacc"] = rec["majority_balanced_accuracy"]
        notes["files"] = _save(fig, fig_dir, "figureM3_recommender")
    return notes


def figure_s1(results: dict, fig_dir: Path) -> dict:
    frame = results["feature_selection_frame"]
    fs = results["feature_selection"]
    notes = {}
    with plt.rc_context(MANUSCRIPT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharex=True, gridspec_kw={"wspace": 0.35})
        for ax, column, label in [
            (axes[0], "morgan_type_share", "Share of selected features:\nMorgan / ECFP / FCFP"),
            (axes[1], "physchem_share", "Share of selected features:\nRDKit 2D + MapLight descriptors"),
        ]:
            for task, marker in (("classification", "o"), ("regression", "s")):
                sub = frame[frame["task"] == task]
                ax.scatter(
                    sub["internal_diversity"],
                    sub[column],
                    s=14,
                    marker=marker,
                    label=task,
                    color="#0072B2" if task == "classification" else "#E69F00",
                    edgecolor="black",
                    linewidth=0.3,
                )
            r = fs[column]
            ax.set_title(f"ρ = {r['rho']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]", fontsize=7.5, fontweight="normal")
            ax.set_xlabel("Internal diversity")
            ax.set_ylabel(label, fontsize=7.5)
            notes[f"rho_{column}"] = r["rho"]
        axes[0].legend(fontsize=6.5)
        notes["files"] = _save(fig, fig_dir, "figureS_meta_feature_selection_diversity")
    return notes


PHASE7_COLORS = {
    "Random forest": "#0072B2",
    "XGBoost": "#56B4E9",
    "ChemML MLP (PyTorch)": "#999999",
    "Uni-Mol V1": "#009E73",
}


def figure_s2(curves: pd.DataFrame, fig_dir: Path) -> dict:
    """Optional Phase 7 learning curves: held-out metric versus training-set size on a fixed test set."""
    datasets = sorted(curves["dataset"].unique())
    notes = {}
    with plt.rc_context(MANUSCRIPT_RC):
        fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 3.0), squeeze=False)
        for ax, dataset in zip(axes[0], datasets):
            sub = curves[curves["dataset"] == dataset]
            for model, color in PHASE7_COLORS.items():
                line = sub[sub["model"] == model].sort_values("n_train")
                if line.empty:
                    continue
                ax.plot(line["n_train"], line["value"], marker="o", ms=3, color=color, label=model)
                notes[f"{dataset}|{model}|full"] = float(line["value"].iloc[-1])
            ax.set_xscale("log")
            ax.set_title(dataset.replace("tdc_", ""), fontsize=8, fontweight="normal")
            ax.set_xlabel("Training molecules (fixed test set)")
            ax.set_ylabel("Test ROC-AUC" if sub["metric"].iloc[0] == "test_roc_auc" else "Test R²")
        axes[0][0].legend(fontsize=6.5)
        notes["files"] = _save(fig, fig_dir, "figureS_meta_learning_curves")
    return notes


def write_all(results: dict, numbers: dict, fig_dir: Path) -> dict:
    fig_dir = Path(fig_dir)
    return {
        "M1": figure_m1(results, fig_dir),
        "M2": figure_m2(results, fig_dir),
        "M3": figure_m3(results, fig_dir),
        "S1": figure_s1(results, fig_dir),
    }
