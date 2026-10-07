"""No-compute reanalyses of the committed benchmark results (revision 2026-10, work order Phase 4).

Run from the repository root, after render_manuscript_assets.py has written the notebook exports:

    python portable_colab_qsar_bundle/reanalysis_coverage_posthoc.py

Reads only committed artifacts: each dataset's ``metrics.csv`` in the manuscript run and the notebook's
``figure6_family_best_models.csv`` (for the per-dataset analysis metric and as a reproduction check). Nothing
is retrained. Writes

- ``manuscript_assets/tables/tableS13_common_subset.csv`` (+ ``.md``): family wins, median relative gap to the
  per-dataset best and the share within 5% of best, on all 44 datasets and on two common subsets (every
  family valid; Uni-Mol V2 valid);
- ``manuscript_assets/tables/tableS14_posthoc_descriptor.csv`` (+ ``.md``): family win tally with and without
  the post-hoc descriptor model, XGBoost (ADMETboost features);
- ``manuscript_assets/reanalysis_numbers.json``: every number the manuscript quotes from these analyses.

``verify_manuscript_numbers.py`` checks the manuscript against the JSON.

Two caveats are part of the result, not bugs. (1) Restricting the dataset set does not change any dataset's
winner, because winners are chosen over all models on that dataset; it changes which datasets are counted.
(2) Removing the descriptor model as a candidate leaves the ensemble rows unchanged: it is a member of the
out-of-fold ensembles, and removing it there would need an ensemble rebuild, which this script does not do.
The without-model tally is therefore a bound on its direct effect only.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = REPO_ROOT / "benchmark_results" / "qsarena_benchmark_oof_ensemble"
TABLES = REPO_ROOT / "manuscript_assets" / "tables"
OUT_JSON = REPO_ROOT / "manuscript_assets" / "reanalysis_numbers.json"
POSTHOC_MODEL = "XGBoost (ADMETboost features)"
UNIMOL_V2 = "Uni-Mol V2 (84m)"
FAMILY_ORDER = [
    "Ensemble (stacking / averaging)", "Conventional ML", "Chemprop v2 GNN", "Uni-Mol (3D pretrained)",
    "TabPFN (tabular foundation)", "MapLight + GNN", "CFA combinatorial fusion", "Deep tabular NN (ChemML MLP)",
]


def load_scores() -> pd.DataFrame:
    """One row per valid (dataset, model) with the notebook's analysis metric, family and direction."""
    fig6 = pd.read_csv(TABLES / "figure6_family_best_models.csv")
    inventory = pd.read_csv(TABLES / "table1_model_inventory.csv")
    family_of = dict(zip(inventory["Model"], inventory["Model family"]))
    metric = fig6.groupby("dataset").agg(metric=("analysis_metric", "first"),
                                         direction=("analysis_metric_direction", "first"))
    frames = []
    for dataset, (column, direction) in metric.iterrows():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = pd.read_csv(RUN_DIR / dataset / "metrics.csv", low_memory=False)
        if "error" in m:
            m = m[m["error"].isna()]
        m = m[m[column].notna()].drop_duplicates("model", keep="first")
        frames.append(pd.DataFrame({
            "dataset": dataset, "model": m["model"].astype(str), "score": m[column].astype(float).to_numpy(),
            "direction": direction,
        }))
    scores = pd.concat(frames, ignore_index=True)
    scores["family"] = scores["model"].map(family_of)
    unmapped = sorted(scores.loc[scores["family"].isna(), "model"].unique())
    if unmapped:
        raise SystemExit(f"models without a family in table1_model_inventory.csv: {unmapped}")
    # Reproduction check: each family's best model per dataset must equal the notebook's export.
    best = family_best(scores)
    merged = best.merge(fig6[["dataset", "family", "analysis_metric_value"]], on=["dataset", "family"], how="outer")
    if merged["score"].isna().any() or merged["analysis_metric_value"].isna().any() or not np.allclose(
            merged["score"], merged["analysis_metric_value"]):
        raise SystemExit("family-best scores do not reproduce figure6_family_best_models.csv")
    return scores


def family_best(scores: pd.DataFrame) -> pd.DataFrame:
    signed = np.where(scores["direction"] == "lower", scores["score"], -scores["score"])
    ordered = scores.assign(_key=signed).sort_values(["dataset", "_key"], kind="stable")
    return ordered.groupby(["dataset", "family"], as_index=False).head(1).drop(columns="_key")


def family_summary(scores: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    """Wins, median relative gap to the per-dataset best and share within 5%, per family, on ``datasets``."""
    sub = scores[scores["dataset"].isin(datasets)]
    signed = np.where(sub["direction"] == "lower", sub["score"], -sub["score"])
    sub = sub.assign(_key=signed)
    best = sub.loc[sub.groupby("dataset")["_key"].idxmin()].set_index("dataset")
    fam = family_best(sub)
    fam["gap_pct"] = 100 * (fam["score"] - fam["dataset"].map(best["score"])).abs() / fam["dataset"].map(
        best["score"]).abs()
    rows = []
    for family in FAMILY_ORDER:
        f = fam[fam["family"] == family]
        rows.append({
            "family": family,
            "datasets": int(len(f)),
            "wins": int((best["family"] == family).sum()),
            "median_gap_pct": float(f["gap_pct"].median()) if len(f) else np.nan,
            "within5_pct": float(100 * (f["gap_pct"] <= 5).mean()) if len(f) else np.nan,
        })
    return pd.DataFrame(rows)


def win_tally(scores: pd.DataFrame) -> dict[str, int]:
    signed = np.where(scores["direction"] == "lower", scores["score"], -scores["score"])
    s = scores.assign(_key=signed)
    winners = s.loc[s.groupby("dataset")["_key"].idxmin()]
    return {family: int((winners["family"] == family).sum()) for family in FAMILY_ORDER}


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    scores = load_scores()
    all_ds = sorted(scores["dataset"].unique())
    fam = family_best(scores)
    valid_families = fam.groupby("dataset")["family"].nunique()
    common = sorted(valid_families[valid_families == len(FAMILY_ORDER)].index)
    v2 = sorted(scores.loc[scores["model"] == UNIMOL_V2, "dataset"].unique())

    subsets = {"All datasets": all_ds, "All eight families valid": common, "Uni-Mol V2 valid": v2}
    blocks = []
    numbers: dict = {"run_dir": RUN_DIR.name, "subsets": {}}
    for name, datasets in subsets.items():
        summary = family_summary(scores, datasets)
        numbers["subsets"][name] = {
            "n_datasets": len(datasets),
            "datasets": datasets,
            "families": {r["family"]: {k: r[k] for k in ("datasets", "wins", "median_gap_pct", "within5_pct")}
                         for r in summary.to_dict("records")},
        }
        blocks.append(summary.assign(subset=f"{name} (n = {len(datasets)})"))
    table13 = pd.concat(blocks, ignore_index=True)
    table13 = table13[["subset", "family", "datasets", "wins", "median_gap_pct", "within5_pct"]].rename(columns={
        "subset": "Dataset subset", "family": "Model family", "datasets": "Datasets", "wins": "Wins",
        "median_gap_pct": "Median gap to best (%)", "within5_pct": "Within 5% of best (%)",
    })
    table13["Median gap to best (%)"] = table13["Median gap to best (%)"].round(1)
    table13["Within 5% of best (%)"] = table13["Within 5% of best (%)"].round(0)

    with_model = win_tally(scores)
    without = win_tally(scores[scores["model"] != POSTHOC_MODEL])
    model_wins = sorted(scores.assign(_k=np.where(scores["direction"] == "lower", scores["score"], -scores["score"]))
                        .pipe(lambda s: s.loc[s.groupby("dataset")["_k"].idxmin()])
                        .query("model == @POSTHOC_MODEL")["dataset"])
    signed = np.where(scores["direction"] == "lower", scores["score"], -scores["score"])
    rest = scores.assign(_k=signed)[lambda s: s["model"] != POSTHOC_MODEL]
    new_winners = rest.loc[rest.groupby("dataset")["_k"].idxmin()].set_index("dataset").loc[model_wins]
    table14 = pd.DataFrame({
        "Model family": FAMILY_ORDER,
        "Wins with the descriptor model": [with_model[f] for f in FAMILY_ORDER],
        "Wins without it": [without[f] for f in FAMILY_ORDER],
    })
    table14["Change"] = table14["Wins without it"] - table14["Wins with the descriptor model"]
    numbers["posthoc"] = {
        "model": POSTHOC_MODEL,
        "datasets_won": model_wins,
        "wins_with": with_model,
        "wins_without": without,
        "replacement_winners": {ds: {"model": r["model"], "family": r["family"]} for ds, r in new_winners.iterrows()},
        "single_family_max_with": max(v for k, v in with_model.items() if not k.startswith(("Ensemble", "CFA"))),
        "single_family_max_without": max(v for k, v in without.items() if not k.startswith(("Ensemble", "CFA"))),
    }

    TABLES.mkdir(parents=True, exist_ok=True)
    table13.to_csv(TABLES / "tableS13_common_subset.csv", index=False)
    (TABLES / "tableS13_common_subset.md").write_text(to_markdown(table13), encoding="utf-8")
    table14.to_csv(TABLES / "tableS14_posthoc_descriptor.csv", index=False)
    (TABLES / "tableS14_posthoc_descriptor.md").write_text(to_markdown(table14), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(numbers, indent=2, default=float), encoding="utf-8")
    print(table13.to_string(index=False))
    print(table14.to_string(index=False))
    print("descriptor model won:", model_wins)
    print("replacement winners:", numbers["posthoc"]["replacement_winners"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
