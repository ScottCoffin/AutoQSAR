"""Supplementary tables for the meta-analysis (CSV plus a Markdown twin, like the notebook's tables)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CATALOG_STEM = "tableS9_meta_feature_catalog"
EFFECTS_STEM = "tableS10_meta_effect_sizes"
SELECTOR_STEM = "tableS11_meta_selector_regret"

CATALOG_COLUMNS = {
    "dataset": "Dataset",
    "task": "Task",
    "split_strategy": "Split",
    "primary_metric": "Metric",
    "n_train": "n train",
    "n_test": "n test",
    "pos_prevalence": "Positive fraction",
    "imbalance_ratio": "Imbalance ratio",
    "target_std": "Target SD",
    "target_skew": "Target skew",
    "n_bemis_murcko_scaffolds": "Scaffolds",
    "scaffolds_per_molecule": "Scaffolds / molecule",
    "singleton_scaffold_frac": "Singleton scaffold fraction",
    "internal_diversity": "Internal diversity",
    "mean_snn": "Mean SNN",
    "ood_fraction": "OOD fraction (SNN < 0.40)",
}


def _md_cell(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.{digits}f}"
    return str(value).replace("|", "/")


def write_table(frame: pd.DataFrame, table_dir: Path, stem: str, digits: int = 3) -> pd.DataFrame:
    frame = frame.reset_index(drop=True)
    frame.to_csv(table_dir / f"{stem}.csv", index=False, float_format="%.10g")
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(_md_cell(v, digits) for v in row) + " |" for row in frame.itertuples(index=False)]
    (table_dir / f"{stem}.md").write_text("\n".join([header, rule, *body]) + "\n", encoding="utf-8")
    return frame


def catalog_table(catalog: pd.DataFrame) -> pd.DataFrame:
    table = catalog[list(CATALOG_COLUMNS)].rename(columns=CATALOG_COLUMNS)
    return table.sort_values(["Task", "Dataset"])


def effects_table(grid: pd.DataFrame) -> pd.DataFrame:
    table = grid.copy()
    table["95% CI"] = [f"{lo:.2f} to {hi:.2f}" for lo, hi in zip(table["ci_low"], table["ci_high"])]
    table = table.rename(
        columns={
            "family": "Model family",
            "meta_feature": "Meta-feature",
            "n_datasets": "Datasets",
            "spearman_rho": "Spearman rho",
            "permutation_p": "Permutation p",
            "bh_q": "BH q",
        }
    )
    return table[["Model family", "Meta-feature", "Datasets", "Spearman rho", "95% CI", "Permutation p", "BH q"]]


def write_all(results: dict, table_dir: Path) -> None:
    table_dir = Path(table_dir)
    results["catalog"].to_csv(table_dir / "meta_feature_catalog.csv", index=False, float_format="%.10g")
    results["catalog_alt"].to_csv(
        table_dir / "meta_feature_catalog_sensitivity_ecfp6_4096.csv", index=False, float_format="%.10g"
    )
    write_table(catalog_table(results["catalog"]), table_dir, CATALOG_STEM)
    write_table(effects_table(results["grid"]), table_dir, EFFECTS_STEM)
    v2 = results.get("selector_v2")
    if v2 is not None:
        table = v2["table"].rename(
            columns={
                "variant": "Selector (model | feature blocks)",
                "mean_regret_pct": "Mean regret (%)",
                "median_regret_pct": "Median regret (%)",
                "within5_pct": "Pick within 5% of best (%)",
                "gap_closed": "SBS-to-oracle gap closed",
            }
        )
        write_table(table, table_dir, SELECTOR_STEM)
        v2["grid"].assign(nested=v2["nested_regret"], nested_pick=v2["nested_picks"]).to_csv(
            table_dir / "meta_selector_v2_regret_by_dataset.csv", float_format="%.10g"
        )
        v2["features"].to_csv(table_dir / "meta_selector_v2_features.csv", float_format="%.10g")
    nat = results["natural_experiment"]
    if isinstance(nat, pd.DataFrame) and not nat.empty:
        nat.to_csv(table_dir / "meta_natural_experiment.csv", index=False, float_format="%.10g")
