"""Evaluate the feature-expansion arm against the benchmark and the official TDC leaderboard.

Pre-specified primary comparison (docs/FEATURE_EXPANSION_PLAN.md): the honest CV-selected pick from
the benchmark's CV-scored models ("old pool") versus the same pool plus the arm's models ("new pool"),
on the 22 official TDC ADMET splits. It is scored head-to-head against every published method that
reports all 22, and by mean rank. Secondary: each arm model as a fixed rule, and the best-of-library
(test-selected) numbers, which are optimistic by construction.

Reference cleaning follows AGENTS.md: self-entries are dropped, the three wrong-scale MolGPS rows are
dropped, and only rows on the leaderboard's own metric are kept.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qsarena.feature_expansion.train import OUT_ROOT
from qsarena.meta_analysis import io

OFFICIAL_TDC = [
    "caco2_wang", "hia_hou", "pgp_broccatelli", "bioavailability_ma", "lipophilicity_astrazeneca",
    "solubility_aqsoldb", "bbb_martins", "ppbr_az", "vdss_lombardo", "cyp2d6_veith", "cyp3a4_veith",
    "cyp2c9_veith", "cyp2d6_substrate_carbonmangels", "cyp3a4_substrate_carbonmangels",
    "cyp2c9_substrate_carbonmangels", "half_life_obach", "clearance_hepatocyte_az", "clearance_microsome_az",
    "ld50_zhu", "herg", "ames", "dili",
]  # fmt: skip
LOWER_IS_BETTER = {"mae", "rmse", "mse"}
TEST_COLUMN = {"mae": "test_mae", "spearman": "test_spearman", "auroc": "test_roc_auc", "auprc": "test_auprc"}
REFERENCE_FILES = [
    io.DEFAULT_RUN_DIR / "publication_leaderboard_top10_reference.csv",
    io.REPO_ROOT / "data" / "benchmark_leaderboards" / "TDC_ADMET_Benchmark_Performance_by_Model.csv",
]
WRONG_SCALE = {("MolGPS", "ppbr_az"), ("MolGPS", "ld50_zhu"), ("MolGPS", "vdss_lombardo")}


def metric_kind(text) -> str | None:
    t = str(text).lower().replace("-", "").replace("_", "")
    for key in ("spearman", "auprc", "auroc", "rocauc", "mae"):
        if key in t:
            return "auroc" if key == "rocauc" else key
    return None


def _better(a: float, b: float, kind: str) -> bool:
    return a < b if kind in LOWER_IS_BETTER else a > b


def leaderboard_kinds() -> pd.Series:
    t4 = pd.read_csv(io.MANUSCRIPT_ASSETS / "tables" / "table4_leaderboard_comparison.csv")
    t4["ds"] = t4["Dataset"].str.replace("^tdc_", "", regex=True)
    return t4.set_index("ds")["Metric"].map(metric_kind)


def load_references(kinds: pd.Series) -> pd.DataFrame:
    frames = []
    a = pd.read_csv(REFERENCE_FILES[0])
    frames.append(
        pd.DataFrame(
            {
                "ds": a["benchmark_id"].astype(str),
                "model": a["model"],
                "kind": a["leaderboard_metric_name"].map(metric_kind),
                "value": pd.to_numeric(a["metric_value_numeric"], errors="coerce"),
            }
        )
    )
    b = pd.read_csv(REFERENCE_FILES[1])
    frames.append(
        pd.DataFrame(
            {"ds": b["Dataset"], "model": b["Model"], "kind": b["Metric"].map(metric_kind), "value": b["Score_Mean"]}
        )
    )
    refs = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    refs = refs[refs["ds"].isin(OFFICIAL_TDC)]
    refs["model"] = refs["model"].astype(str).str.strip()
    refs = refs[~refs["model"].str.contains("autoqsar|qsarena", case=False)]
    refs = refs[~refs.apply(lambda r: any(m in r["model"] and r["ds"] == d for m, d in WRONG_SCALE), axis=1)]
    refs = refs[refs["kind"] == refs["ds"].map(kinds)]
    refs["key"] = refs["model"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    return refs.groupby(["ds", "key"], as_index=False).agg(model=("model", "first"), value=("value", "mean"))


def load_candidates(run_dir: Path = io.DEFAULT_RUN_DIR, arm_root: Path = OUT_ROOT) -> pd.DataFrame:
    """Every valid benchmark model row plus every arm row, with a ``pool`` column (benchmark / arm)."""
    rows = []
    wanted = {"model", "error", "primary_metric", "cv_primary", *TEST_COLUMN.values(), "test_rmse", "test_r2"}
    for d in io.dataset_dirs(run_dir):
        m = pd.read_csv(d / "metrics.csv", usecols=lambda c: c in wanted)
        m = m[m["error"].isna()].drop_duplicates("model", keep="last")
        rows.append(m.assign(dataset=d.name, pool="benchmark"))
    for path in sorted(Path(arm_root).glob("*/metrics.csv")):
        # "<feature_set>__<variant>" dirs (e.g. admetboost__scaffoldcv) rescore the same models under another
        # CV protocol with no test metrics; mixing them in would duplicate candidates and blank test values.
        if "__" in path.parent.name:
            continue
        a = pd.read_csv(path)
        rows.append(a.assign(pool="arm"))
    frame = pd.concat(rows, ignore_index=True, sort=False)
    frame["ds"] = frame["dataset"].str.replace("^tdc_", "", regex=True)
    return frame


def cv_pick(candidates: pd.DataFrame, pools: set[str]) -> pd.DataFrame:
    """Per dataset, the model with the best cv_primary (honest: training-set CV only)."""
    c = candidates[candidates["pool"].isin(pools) & candidates["cv_primary"].notna()].copy()
    c["cv_sort"] = np.where(c["primary_metric"].isin(LOWER_IS_BETTER), c["cv_primary"], -c["cv_primary"])
    return c.sort_values(["dataset", "cv_sort", "model"]).groupby("dataset", as_index=False).nth(0)


def tdc_table(entries: dict[str, pd.Series], refs: pd.DataFrame, kinds: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Head-to-head of each QSARena entry against full-coverage methods, and mean ranks."""
    coverage = refs.groupby("model")["ds"].nunique()
    full = coverage[coverage >= len(OFFICIAL_TDC)].index.tolist()
    h2h, ranks = [], []
    for name, values in entries.items():
        for method in full:
            ref = refs[refs["model"] == method].set_index("ds")["value"]
            wins = sum(_better(values[d], ref[d], kinds[d]) for d in OFFICIAL_TDC if np.isfinite(values.get(d, np.nan)))
            losses = sum(
                _better(ref[d], values[d], kinds[d]) for d in OFFICIAL_TDC if np.isfinite(values.get(d, np.nan))
            )
            h2h.append({"entry": name, "method": method, "qsarena_wins": wins, "qsarena_losses": losses})
    for d in OFFICIAL_TDC:
        vals = {m: refs[(refs["model"] == m) & (refs["ds"] == d)]["value"].iloc[0] for m in full}
        vals.update({name: values.get(d, np.nan) for name, values in entries.items()})
        s = pd.Series(vals).dropna()
        for name, r in s.rank(ascending=kinds[d] in LOWER_IS_BETTER, method="min").items():
            ranks.append({"ds": d, "entry": name, "rank": r})
    rank = (
        pd.DataFrame(ranks)
        .groupby("entry")["rank"]
        .agg(mean_rank="mean", median_rank="median", n="count", firsts=lambda s: int((s == 1).sum()))
    )
    return pd.DataFrame(h2h), rank.sort_values("mean_rank").reset_index()


def paired_vs_reference(arm_root: Path = OUT_ROOT, reference: str = "admetboost", model_key: str = "xgboost") -> dict:
    """Paired test-metric comparison of every arm feature set against ``reference`` (same model, same datasets).

    Relative change is oriented so positive = better (lower error or higher score); only datasets whose primary
    test metric is present in both sets count. Wilcoxon signed-rank p-value, two-sided.
    """
    from scipy.stats import wilcoxon

    def load(feature_set: str) -> pd.DataFrame:
        frame = pd.read_csv(Path(arm_root) / feature_set / "metrics.csv")
        frame = frame[frame["model_key"] == model_key]
        return frame.drop_duplicates("dataset", keep="last").set_index("dataset")

    base = load(reference)
    out = {}
    for path in sorted(Path(arm_root).glob("*/metrics.csv")):
        feature_set = path.parent.name
        if feature_set == reference or "__" in feature_set:
            continue
        other, gains = load(feature_set), []
        for dataset in other.index.intersection(base.index):
            metric = str(other.loc[dataset, "primary_metric"])
            col = f"test_{metric}"
            if col not in other or pd.isna(other.loc[dataset, col]) or pd.isna(base.loc[dataset, col]):
                continue
            x, y = float(base.loc[dataset, col]), float(other.loc[dataset, col])
            gains.append(100 * ((x - y) / abs(x) if metric in LOWER_IS_BETTER else (y - x) / abs(x)))
        gains = np.asarray(gains)
        if len(gains) == 0:
            continue
        out[feature_set] = {
            "n": int(len(gains)), "better": int((gains > 0).sum()), "worse": int((gains < 0).sum()),
            "median_pct": round(float(np.median(gains)), 3), "mean_pct": round(float(np.mean(gains)), 3),
            "wilcoxon_p": round(float(wilcoxon(gains).pvalue), 4),
        }
    return out


def evaluate(out_dir: Path = OUT_ROOT) -> dict:
    kinds = leaderboard_kinds()
    refs = load_references(kinds)
    cand = load_candidates()
    tdc = cand[cand["ds"].isin(OFFICIAL_TDC)]

    def test_values(frame: pd.DataFrame) -> pd.Series:
        return pd.Series({r.ds: getattr(r, TEST_COLUMN[kinds[r.ds]]) for r in frame.itertuples()}, dtype=float)

    old_pick, new_pick = cv_pick(tdc, {"benchmark"}), cv_pick(tdc, {"benchmark", "arm"})
    entries = {"CV pick, benchmark pool": test_values(old_pick), "CV pick, benchmark + arm": test_values(new_pick)}
    for model, frame in tdc[tdc["pool"] == "arm"].groupby("model"):
        entries[f"fixed: {model}"] = test_values(frame)
    h2h, rank = tdc_table(entries, refs, kinds)
    out_dir.mkdir(parents=True, exist_ok=True)
    h2h.to_csv(out_dir / "tdc22_head_to_head.csv", index=False)
    rank.to_csv(out_dir / "tdc22_mean_rank.csv", index=False)
    picks = new_pick[["dataset", "model", "pool", "cv_primary"]].assign(
        benchmark_pool_pick=new_pick["dataset"].map(old_pick.set_index("dataset")["model"])
    )
    picks.to_csv(out_dir / "tdc22_cv_picks.csv", index=False)
    summary = {
        "n_arm_rows": int((cand["pool"] == "arm").sum()),
        "arm_models": sorted(cand.loc[cand["pool"] == "arm", "model"].unique()),
        "tdc_datasets_with_arm_results": int(tdc.loc[tdc["pool"] == "arm", "ds"].nunique()),
        "cv_pick_is_arm_model": int((new_pick["pool"] == "arm").sum()),
        "mean_rank": rank.set_index("entry")["mean_rank"].round(3).to_dict(),
        "paired_vs_admetboost_xgboost": paired_vs_reference(),
    }
    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"h2h": h2h, "rank": rank, "picks": picks, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(OUT_ROOT))
    args = parser.parse_args(argv)
    result = evaluate(Path(args.out_dir))
    pd.set_option("display.width", 200)
    print(result["rank"].round(2).to_string(index=False))
    print(result["h2h"].to_string(index=False))
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
