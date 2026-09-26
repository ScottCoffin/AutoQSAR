"""
qsarena.admet_ai_parity — head-to-head table: QSARena's ADMET-AI-equivalent variant vs ADMET-AI.

ADMET-AI's architecture is Chemprop-RDKit (a D-MPNN graph encoder concatenated with RDKit 2D
descriptors). QSARena's ``Chemprop v2 (D-MPNN + RDKit2D, ...)`` variant is the same recipe
re-implemented through Chemprop v2. This module builds one row per TDC ADMET Benchmark Group
dataset with:

  * our variant's test value on the TDC leaderboard metric, from a run directory's metrics.csv;
  * the Chemprop-RDKit value on the TDC leaderboard, where it appears in the captured top-10.

Every dataset appears in the output. A missing value on either side is an explicit NA with a
reason in ``ours_status`` / ``reference_status`` -- never a dropped row -- because the point of the
table is to show how much of the comparison actually exists. ``parity_holds`` is only computed
where both sides are present.

    python -m qsarena.admet_ai_parity --run-dir benchmark_results/<run> --out results/admet_ai_parity.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["METRIC_COLUMNS", "HIGHER_IS_BETTER", "build_parity_table", "main"]

VARIANT_PATTERN = "D-MPNN + RDKit2D"
REFERENCE_MODEL = "Chemprop-RDKit"

#: TDC leaderboard metric name -> metrics.csv column.
METRIC_COLUMNS = {"MAE": "test_mae", "Spearman": "test_spearman", "AUROC": "test_roc_auc", "AUPRC": "test_auprc"}
HIGHER_IS_BETTER = {"MAE": False, "Spearman": True, "AUROC": True, "AUPRC": True}

TDC22 = (
    "ames", "bbb_martins", "bioavailability_ma", "caco2_wang", "clearance_hepatocyte_az",
    "clearance_microsome_az", "cyp2c9_substrate_carbonmangels", "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels", "cyp2d6_veith", "cyp3a4_substrate_carbonmangels",
    "cyp3a4_veith", "dili", "half_life_obach", "herg", "hia_hou", "ld50_zhu",
    "lipophilicity_astrazeneca", "pgp_broccatelli", "ppbr_az", "solubility_aqsoldb", "vdss_lombardo",
)


#: TDC ADMET Benchmark Group official metric per dataset (https://tdcommons.ai/benchmark/admet_group/).
#: Used when the captured leaderboard file carries no metric name for a dataset.
TDC_OFFICIAL_METRIC = {
    "caco2_wang": "MAE", "hia_hou": "AUROC", "pgp_broccatelli": "AUROC", "bioavailability_ma": "AUROC",
    "lipophilicity_astrazeneca": "MAE", "solubility_aqsoldb": "MAE", "bbb_martins": "AUROC",
    "ppbr_az": "MAE", "vdss_lombardo": "Spearman", "cyp2d6_veith": "AUPRC", "cyp3a4_veith": "AUPRC",
    "cyp2c9_veith": "AUPRC", "cyp2d6_substrate_carbonmangels": "AUPRC",
    "cyp3a4_substrate_carbonmangels": "AUROC", "cyp2c9_substrate_carbonmangels": "AUPRC",
    "half_life_obach": "Spearman", "clearance_microsome_az": "Spearman",
    "clearance_hepatocyte_az": "Spearman", "herg": "AUROC", "ames": "AUROC", "dili": "AUROC",
    "ld50_zhu": "MAE",
}


def _leaderboard_metric(leaderboard: pd.DataFrame, dataset_key: str) -> str | None:
    rows = leaderboard[leaderboard["dataset"] == dataset_key]
    names = rows["leaderboard_metric_name"].dropna().unique()
    if len(names):
        return str(names[0])
    return TDC_OFFICIAL_METRIC.get(dataset_key.removeprefix("tdc_"))


def _our_value(metrics_path: Path, metric: str | None) -> tuple[float, str]:
    if not metrics_path.exists():
        return np.nan, "dataset not in run"
    m = pd.read_csv(metrics_path, low_memory=False)
    rows = m[m["model"].astype(str).str.contains(VARIANT_PATTERN, regex=False)]
    if rows.empty:
        return np.nan, "variant not configured"
    if metric is None or METRIC_COLUMNS.get(metric) not in m.columns:
        return np.nan, "leaderboard metric unknown"
    col = METRIC_COLUMNS[metric]
    err = rows["error"] if "error" in rows.columns else pd.Series(np.nan, index=rows.index)
    valid = rows[err.isna() & rows[col].notna()]
    if valid.empty:
        return np.nan, "variant failed"
    # First valid row, whole-row selection (never groupby().first(): see AGENTS.md).
    return float(valid.iloc[0][col]), "ok"


def _reference_value(leaderboard: pd.DataFrame, dataset_key: str) -> tuple[float, float, str]:
    rows = leaderboard[(leaderboard["dataset"] == dataset_key)
                       & (leaderboard["model"].astype(str).str.strip() == REFERENCE_MODEL)]
    if rows.empty:
        return np.nan, np.nan, f"{REFERENCE_MODEL} not in captured TDC top-10"
    row = rows.iloc[0]
    return float(row["metric_value_numeric"]), float(row["rank_numeric"]), "ok"


def build_parity_table(run_dir: str | Path, leaderboard_csv: str | Path, datasets=TDC22) -> pd.DataFrame:
    run_dir = Path(run_dir)
    leaderboard = pd.read_csv(leaderboard_csv)
    rows = []
    for name in datasets:
        key = f"tdc_{name}"
        metric = _leaderboard_metric(leaderboard, key)
        ours, ours_status = _our_value(run_dir / key / "metrics.csv", metric)
        ref, ref_rank, ref_status = _reference_value(leaderboard, key)
        both = ours_status == "ok" and ref_status == "ok"
        delta = ours - ref if both else np.nan
        if both:
            better = HIGHER_IS_BETTER[metric]
            # Parity: within 5% of the reference in the reference's own units, or better.
            tol = 0.05 * abs(ref)
            parity = bool((ours >= ref - tol) if better else (ours <= ref + tol))
        else:
            parity = pd.NA
        rows.append({
            "dataset": name,
            "leaderboard_metric": metric if metric else pd.NA,
            "ours_variant": f"Chemprop v2 ({VARIANT_PATTERN})",
            "ours_value": ours,
            "ours_status": ours_status,
            "reference_model": REFERENCE_MODEL,
            "reference_value": ref,
            "reference_tdc_rank": ref_rank,
            "reference_status": ref_status,
            "delta_ours_minus_reference": delta,
            "parity_holds_within_5pct": parity,
        })
    out = pd.DataFrame(rows)
    out["parity_holds_within_5pct"] = out["parity_holds_within_5pct"].astype("boolean")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the ADMET-AI parity table.")
    p.add_argument("--run-dir", type=Path, default=Path("benchmark_results/autoqsar_benchmark_20260623_153839"))
    p.add_argument("--leaderboard", type=Path,
                   default=Path("data/benchmark_leaderboards/leaderboard_top10_reference_latest.csv"))
    p.add_argument("--out", type=Path, default=Path("results/admet_ai_parity.csv"))
    args = p.parse_args(argv)
    table = build_parity_table(args.run_dir, args.leaderboard)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, na_rep="NA")
    n_ours = int((table.ours_status == "ok").sum())
    n_ref = int((table.reference_status == "ok").sum())
    n_both = int(table.parity_holds_within_5pct.notna().sum())
    print(f"{len(table)} datasets; ours valid on {n_ours}; reference present on {n_ref}; comparable on {n_both}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
