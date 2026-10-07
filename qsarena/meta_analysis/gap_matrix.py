"""Dataset x model-family relative-gap matrix, identical in definition to manuscript Fig 6.

The notebook export cell (``# MANUSCRIPT_FIGURE_EXPORT``) computes each model's
``relative_gap_to_best = |score - best| / |best|`` on the primary metric, oriented so lower is better.
It keeps each family's best model per dataset (``family_best``) and pivots that into the Fig 6
heatmap. It writes that exact frame to ``manuscript_assets/tables/figure6_family_best_models.csv``,
which is what this module reads, so the gap is never re-derived independently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qsarena.meta_analysis.io import DEFAULT_RUN_DIR, FAMILY_BEST_PATH, dataset_dirs

NEAR_BEST = 0.05
CHEMPROP = "Chemprop v2 GNN"

#: Family order and colours used by the manuscript figures (notebook ``FAMILY_COLORS``).
FAMILY_COLORS = {
    "Conventional ML": "#0072B2",
    "Ensemble (stacking / averaging)": "#E69F00",
    "CFA combinatorial fusion": "#D55E00",
    "Uni-Mol (3D pretrained)": "#009E73",
    "TabPFN (tabular foundation)": "#CC79A7",
    "Chemprop v2 GNN": "#56B4E9",
    "MapLight + GNN": "#F0E442",
    "Deep tabular NN (ChemML MLP)": "#999999",
    "GA-tuned conventional ML": "#332288",
    "Other": "#BBBBBB",
}

#: Grouped recommender target (at most four classes so each stays populated with n = 44).
FAMILY_GROUPS = {
    "Ensemble (stacking / averaging)": "Fusion (ensemble / CFA)",
    "CFA combinatorial fusion": "Fusion (ensemble / CFA)",
    "Conventional ML": "Descriptor-based ML",
    "GA-tuned conventional ML": "Descriptor-based ML",
    "MapLight + GNN": "Descriptor-based ML",
    "TabPFN (tabular foundation)": "Descriptor-based ML",
    "Deep tabular NN (ChemML MLP)": "Descriptor-based ML",
    "Uni-Mol (3D pretrained)": "Uni-Mol (3D pretrained)",
    "Chemprop v2 GNN": "Chemprop GNN",
}
GROUP_ORDER = ["Fusion (ensemble / CFA)", "Descriptor-based ML", "Uni-Mol (3D pretrained)", "Chemprop GNN"]


def load_family_best(path: Path | str = FAMILY_BEST_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: it is written by the notebook export cell. Run "
            "python portable_colab_qsar_bundle/render_manuscript_assets.py first."
        )
    return pd.read_csv(path)


def load_family_gap_matrix(family_best: pd.DataFrame | None = None) -> pd.DataFrame:
    """Index: dataset; columns: family (manuscript order); values: relative gap in percent (Fig 6)."""
    family_best = load_family_best() if family_best is None else family_best
    heat = family_best.pivot_table(index="dataset", columns="family", values="relative_gap_to_best", aggfunc="min")
    heat = heat.reindex(columns=[f for f in FAMILY_COLORS if f in heat.columns]) * 100.0
    heat.columns.name = None
    return heat.sort_index()


def within5_matrix(gap_pct: pd.DataFrame, threshold: float = NEAR_BEST) -> pd.DataFrame:
    """Boolean matrix (NaN kept as NaN): family-best model within ``threshold`` of the dataset best."""
    out = (gap_pct <= threshold * 100.0 + 1e-9).astype(object)
    return out.where(gap_pct.notna(), np.nan)


def valid_mask(gap_pct: pd.DataFrame) -> pd.DataFrame:
    """True where the family produced a valid result on the dataset."""
    return gap_pct.notna()


def chemprop_valid(gap_pct: pd.DataFrame) -> pd.Series:
    return gap_pct[CHEMPROP].notna() if CHEMPROP in gap_pct.columns else pd.Series(False, index=gap_pct.index)


def trend_families(gap_pct: pd.DataFrame, min_coverage: float = 0.5) -> list[str]:
    """Families valid on at least ``min_coverage`` of datasets; sparser families are excluded from trend fits.

    The spec anticipated Chemprop at 6/44 valid datasets. The Chemprop repair runs made it valid on
    most datasets, so the exclusion rule is applied generically instead of naming Chemprop.
    """
    coverage = gap_pct.notna().mean()
    return [f for f in gap_pct.columns if coverage[f] >= min_coverage]


def winner_family(family_best: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per dataset: task, family of the overall best model, and its grouped class."""
    family_best = load_family_best() if family_best is None else family_best
    ordered = family_best.sort_values(["dataset", "comparison_score", "family"])
    winners = ordered.groupby("dataset", as_index=False).nth(0).reset_index(drop=True)
    winners = winners[["dataset", "task_kind", "family", "model", "analysis_metric"]].rename(
        columns={"family": "winner_family", "model": "winner_model"}
    )
    winners["winner_group"] = winners["winner_family"].map(FAMILY_GROUPS).fillna("Descriptor-based ML")
    return winners.sort_values("dataset").reset_index(drop=True)


def group_gap_matrix(gap_pct: pd.DataFrame) -> pd.DataFrame:
    """Best (minimum) gap within each recommender group."""
    groups = {}
    for group in GROUP_ORDER:
        members = [f for f, g in FAMILY_GROUPS.items() if g == group and f in gap_pct.columns]
        if members:
            groups[group] = gap_pct[members].min(axis=1, skipna=True)
    return pd.DataFrame(groups)


def load_achievable_best(run_dir: Path | str = DEFAULT_RUN_DIR) -> pd.DataFrame:
    """Best held-out test R^2 (regression) or ROC-AUC (classification) across valid model rows.

    A scale-free "how hard is this dataset" measure for Fig M2 (primary metrics mix units).
    """
    rows = []
    for d in dataset_dirs(run_dir):
        wanted = {"model", "error", "test_r2", "test_roc_auc"}
        frame = pd.read_csv(d / "metrics.csv", usecols=lambda c: c in wanted)
        if "error" in frame.columns:
            frame = frame.loc[frame["error"].isna() | (frame["error"].astype(str).str.strip() == "")]
        record = {"dataset": d.name, "best_test_r2": np.nan, "best_test_roc_auc": np.nan}
        for column, key in (("test_r2", "best_test_r2"), ("test_roc_auc", "best_test_roc_auc")):
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce")
                if values.notna().any():
                    record[key] = float(values.max())
        rows.append(record)
    return pd.DataFrame(rows)
