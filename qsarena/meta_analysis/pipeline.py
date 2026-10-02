"""End-to-end meta-analysis: deposited artifacts -> catalog, statistics, figures, tables, meta_numbers.json.

Deterministic (fixed seeds, sorted inputs) and CPU-only; no model is trained.

    python -m qsarena.meta_analysis                      # writes into manuscript_assets/
    python -m qsarena.meta_analysis --build-partitions   # one-off, needs the gitignored predictions.csv files
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qsarena.meta_analysis import cv_leak, io
from qsarena.meta_analysis import gap_matrix as gm
from qsarena.meta_analysis import meta_features as mf
from qsarena.meta_analysis import stats as st

SEED = 0
#: Meta-features screened against every family's gap (Spearman grid, BH-controlled).
GRID_FEATURES = [
    "log10_n_train",
    "internal_diversity",
    "mean_snn",
    "ood_fraction",
    "scaffolds_per_molecule",
    "label_asymmetry",
]
FINGERPRINT_DEPENDENT = ["internal_diversity", "mean_snn", "ood_fraction"]
#: Recommender predictors (at most four; see METHODS.md).
RECOMMENDER_FEATURES = ["log10_n_train", "mean_snn", "label_asymmetry", "is_classification"]
#: Pre-registered crossover contrast, plus the fusion-vs-pretrained secondary contrast.
CROSSOVER_PRIMARY = ("Conventional ML", "Uni-Mol (3D pretrained)")
CROSSOVER_SECONDARY = ("Ensemble (stacking / averaging)", "Uni-Mol (3D pretrained)")
MORGAN_TYPE = ["morgan", "ecfp6", "fcfp6"]
PHYSCHEM_TYPE = ["rdkit", "maplight"]


def add_derived_features(catalog: pd.DataFrame) -> pd.DataFrame:
    """``label_asymmetry``: log10(imbalance ratio) for classification, |skew| for regression.

    Both measure how lopsided the training-label distribution is, so one predictor covers both
    tasks, which keeps the recommender within four predictors.
    """
    out = catalog.copy()
    out["is_classification"] = (out["task"] == "classification").astype(float)
    out["label_asymmetry"] = np.where(
        out["task"] == "classification",
        np.log10(out["imbalance_ratio"].astype(float)),
        out["target_skew"].astype(float).abs(),
    )
    return out


def correlation_grid(catalog: pd.DataFrame, gap_pct: pd.DataFrame, families, features, seed: int = SEED):
    rows = []
    cat = catalog.set_index("dataset")
    for family in families:
        for feature in features:
            joined = pd.concat([cat[feature], gap_pct[family]], axis=1, join="inner").dropna()
            ci = st.spearman_with_ci(joined.iloc[:, 0], joined.iloc[:, 1], seed=seed)
            p = st.permutation_test(joined.iloc[:, 0], joined.iloc[:, 1], seed=seed)
            rows.append(
                {
                    "family": family,
                    "meta_feature": feature,
                    "n_datasets": int(len(joined)),
                    "spearman_rho": ci["estimate"],
                    "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                    "permutation_p": p,
                }
            )
    grid = pd.DataFrame(rows)
    grid["bh_q"] = st.bh_qvalues(grid["permutation_p"])
    return grid


def difficulty_analysis(catalog: pd.DataFrame, achievable: pd.DataFrame, seed: int = SEED) -> dict:
    """Spearman of each meta-feature with the achievable best (R^2 or ROC-AUC), within task and pooled.

    Pooled uses the within-task percentile rank of the achievable best, so the two scales combine.
    """
    frame = catalog.merge(achievable, on="dataset")
    frame["achievable_best"] = np.where(
        frame["task"] == "classification", frame["best_test_roc_auc"], frame["best_test_r2"]
    )
    frame["achievable_rank"] = frame.groupby("task")["achievable_best"].rank(pct=True)
    out = {"per_feature": {}, "frame": frame}
    for feature in ["mean_snn", "ood_fraction", "log10_n_train", "internal_diversity", "scaffolds_per_molecule"]:
        record = {}
        for label, sub, target in [
            ("pooled", frame, "achievable_rank"),
            ("classification", frame[frame["task"] == "classification"], "achievable_best"),
            ("regression", frame[frame["task"] == "regression"], "achievable_best"),
        ]:
            ci = st.spearman_with_ci(sub[feature], sub[target], seed=seed)
            record[label] = {
                "rho": ci["estimate"],
                "ci": [ci["ci_low"], ci["ci_high"]],
                "p": st.permutation_test(sub[feature], sub[target], seed=seed),
                "n": ci["n"],
            }
        out["per_feature"][feature] = record
    pooled = {f: abs(v["pooled"]["rho"]) for f, v in out["per_feature"].items()}
    out["strongest_feature"] = max(pooled, key=pooled.get)
    return out


def natural_experiment(current_best: pd.DataFrame, comparison_best: pd.DataFrame, resplit: list[str]) -> pd.DataFrame:
    """Family gap on the same chemistry before (random / target-quartile) and after the scaffold re-split."""
    cur = gm.load_family_gap_matrix(current_best)
    old = gm.load_family_gap_matrix(comparison_best)
    rows = []
    for dataset in resplit:
        if dataset not in cur.index or dataset not in old.index:
            continue
        for family in cur.columns:
            if family in old.columns and pd.notna(cur.at[dataset, family]) and pd.notna(old.at[dataset, family]):
                rows.append(
                    {
                        "dataset": dataset,
                        "family": family,
                        "gap_low_shift_pct": float(old.at[dataset, family]),
                        "gap_scaffold_pct": float(cur.at[dataset, family]),
                    }
                )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["delta_pp"] = frame["gap_scaffold_pct"] - frame["gap_low_shift_pct"]
    return frame


def feature_selection_vs_diversity(catalog: pd.DataFrame, families: pd.DataFrame, seed: int = SEED) -> tuple:
    fam = families.set_index("dataset")
    total = fam.drop(columns=["maplight_classic"], errors="ignore").sum(axis=1)
    shares = pd.DataFrame(
        {
            "morgan_type_share": fam[[c for c in MORGAN_TYPE if c in fam.columns]].sum(axis=1) / total,
            "physchem_share": fam[[c for c in PHYSCHEM_TYPE if c in fam.columns]].sum(axis=1) / total,
        }
    )
    frame = catalog.set_index("dataset")[["internal_diversity", "task"]].join(shares, how="inner")
    result = {}
    for column in shares.columns:
        ci = st.spearman_with_ci(frame["internal_diversity"], frame[column], seed=seed)
        result[column] = {
            "rho": ci["estimate"],
            "ci": [ci["ci_low"], ci["ci_high"]],
            "p": st.permutation_test(frame["internal_diversity"], frame[column], seed=seed),
        }
    return frame.reset_index(), result


PERMUTATION_CACHE = io.MANUSCRIPT_ASSETS / "tables" / "meta_selector_v2_permutation.json"


def _cached_permutation(X, gap_pct, observed, n_perm, seed, candidates) -> dict:
    """The nested permutation test reruns the whole nested pipeline n_perm times (~1 h on a laptop).

    Its result is cached under a hash of every input (features, gaps, candidates, n_perm, seed), so it
    is recomputed only when the data change, and the default render stays within minutes.
    """
    import hashlib

    from qsarena.meta_analysis import selection as sel

    payload = json.dumps(
        {
            "X": np.round(X.to_numpy(dtype=float), 10).tolist(),
            "gap": np.round(gap_pct.to_numpy(dtype=float), 10).tolist(),
            "rows": list(map(str, X.index)),
            "cols": list(map(str, gap_pct.columns)),
            "candidates": list(candidates),
            "n_perm": int(n_perm),
            "seed": int(seed),
        },
        default=lambda v: None,
    )
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if PERMUTATION_CACHE.exists():
        cached = json.loads(PERMUTATION_CACHE.read_text(encoding="utf-8"))
        if cached.get("input_sha256") == key:
            return cached["result"]
    result = sel.permutation_p_nested(X, gap_pct, observed, n_perm=n_perm, seed=seed, candidates=candidates)
    PERMUTATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PERMUTATION_CACHE.write_text(json.dumps({"input_sha256": key, "result": result}, indent=2) + "\n", encoding="utf-8")
    return result


def selector_v2(
    catalog: pd.DataFrame,
    selector_catalog: pd.DataFrame,
    landmarks: pd.DataFrame,
    gap_pct: pd.DataFrame,
    n_perm: int = 100,
    seed: int = SEED,
) -> dict:
    """Regret-based family selector (docs/meta_analysis/SELECTOR_V2_PLAN.md): grid, nested headline, permutation."""
    from qsarena.meta_analysis import selection as sel

    X = (
        catalog.set_index("dataset")[sel.F0]
        .join(selector_catalog.set_index("dataset"))
        .join(landmarks.set_index("dataset"))
        .loc[gap_pct.index]
    )
    grid = sel.evaluate_grid(X, gap_pct)
    sbs = grid["sbs"].to_numpy()
    candidates = [n for n in grid.columns if not n.startswith("rf|")]
    nested, picks = sel.nested_regret(X, gap_pct, candidates)
    table = pd.DataFrame(
        [{"variant": n, **sel.summarize_regret(grid[n].to_numpy(), sbs)} for n in grid.columns]
        + [{"variant": "nested (headline)", **sel.summarize_regret(nested, sbs)}]
    )
    diff, lo, hi = sel.paired_bootstrap(sbs, nested, seed=seed)
    perm = _cached_permutation(X, gap_pct, float(nested.mean()), n_perm, seed, candidates)
    raw = gap_pct.to_numpy(dtype=float)
    return {
        "features": X,
        "grid": grid,
        "table": table,
        "nested_regret": pd.Series(nested, index=gap_pct.index),
        "nested_picks": pd.Series(picks, index=gap_pct.index),
        "summary": {
            "nested": sel.summarize_regret(nested, sbs),
            "sbs": sel.summarize_regret(sbs, sbs),
            "random_valid_family_mean_regret_pct": float(np.nanmean(raw)),
            "sbs_minus_nested_pct": diff,
            "sbs_minus_nested_ci": [lo, hi],
            "permutation": perm,
            "nested_pick_counts": pd.Series(picks).value_counts().to_dict(),
            "best_grid_variant": str(table.iloc[:-1].sort_values("mean_regret_pct").iloc[0]["variant"]),
            "n_candidates": len(candidates),
        },
    }


def within5_summary(gap_pct: pd.DataFrame) -> dict:
    w = gm.within5_matrix(gap_pct)
    return {
        f: float(np.mean([bool(v) for v in w[f].dropna()]) * 100) if w[f].notna().any() else np.nan
        for f in gap_pct.columns
    }


def sensitivity_summary(grid: pd.DataFrame, grid_alt: pd.DataFrame, catalog, catalog_alt) -> dict:
    merged = grid.merge(grid_alt, on=["family", "meta_feature"], suffixes=("", "_alt"))
    fp = merged[merged["meta_feature"].isin(FINGERPRINT_DEPENDENT)]
    same_sign = np.sign(fp["spearman_rho"]) == np.sign(fp["spearman_rho_alt"])
    sig = fp["bh_q"] < 0.10
    sig_alt = fp["bh_q_alt"] < 0.10
    agree = {}
    a, b = catalog.set_index("dataset"), catalog_alt.set_index("dataset")
    for feature in FINGERPRINT_DEPENDENT:
        agree[feature] = float(st._spearman(a[feature].to_numpy(), b.loc[a.index, feature].to_numpy()))
    return {
        "fingerprint_default": dict(mf.DEFAULT_FP),
        "fingerprint_alternative": dict(mf.SENSITIVITY_FP),
        "cells_compared": int(len(fp)),
        "same_sign_fraction": float(same_sign.mean()) if len(fp) else np.nan,
        "significant_cells_default": int(sig.sum()),
        "significant_cells_alternative": int(sig_alt.sum()),
        "significance_agreement_fraction": float((sig == sig_alt).mean()) if len(fp) else np.nan,
        "feature_value_rank_agreement": agree,
    }


def _fmt_n(value: float) -> str:
    """Two significant figures: a single-seed, 44-dataset crossover does not merit more."""
    if not np.isfinite(value):
        return "n/a"
    digits = max(0, 2 - int(np.floor(np.log10(abs(value)))) - 1)
    return f"{round(value, -int(np.floor(np.log10(abs(value)))) + 1):,.{digits}f}"


FEATURE_LABELS = {
    "log10_n_train": "log10 training-set size",
    "internal_diversity": "internal diversity",
    "mean_snn": "mean SNN",
    "ood_fraction": "the out-of-domain fraction (test molecules with SNN < 0.40)",
    "scaffolds_per_molecule": "scaffolds per molecule",
    "label_asymmetry": "label asymmetry",
}
MODEL_LABELS = {"decision_tree": "depth-3 decision tree", "l1_logistic": "L1-penalised multinomial logistic model"}
NUMBER_WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
MIN_NATURAL_PAIRS = 3


def _fmt_ci(lo: float, hi: float, digits: int = 2) -> str:
    return f"{lo:.{digits}f} to {hi:.{digits}f}" if np.isfinite(lo) and np.isfinite(hi) else "not estimable"


def _size_sentence(row: pd.Series) -> str:
    rho, lo, hi, q = row["spearman_rho"], row["ci_low"], row["ci_high"], row["bh_q"]
    stats_text = f"Spearman ρ = {rho:.2f}, 95% CI {lo:.2f} to {hi:.2f}, BH q = {q:.2f}"
    if hi < 0:
        trend = "narrowed as training sets grew"
    elif lo > 0:
        trend = "widened as training sets grew"
    else:
        trend = "showed no clear monotone trend with training-set size"
    return f"The gap of the best conventional-ML model to the per-dataset best {trend} ({stats_text})."


def _snn_sentence(snn: dict) -> str:
    lo, hi = snn["ci"]
    stats_text = (
        f"within-task rank; Spearman ρ = {snn['rho']:.2f}, 95% CI {lo:.2f} to {hi:.2f}, "
        f"permutation p = {snn['p']:.3f}; Fig. 8a"
    )
    if lo > 0:
        return (
            "Datasets whose test molecules had closer training neighbours were easier: mean SNN correlated "
            f"positively with the achievable best held-out metric ({stats_text})."
        )
    if hi < 0:
        return f"Unexpectedly, mean SNN correlated negatively with the achievable best held-out metric ({stats_text})."
    return f"Mean SNN was not clearly associated with the achievable best held-out metric ({stats_text})."


def _strongest_sentence(feature: str, record: dict) -> str:
    lo, hi = record["ci"]
    label = FEATURE_LABELS.get(feature, feature.replace("_", " "))
    text = (
        f"Of the screened properties, {label} had the strongest association with achievable performance "
        f"(ρ = {record['rho']:.2f}, 95% CI {lo:.2f} to {hi:.2f})"
    )
    return text + (", although its interval also includes zero." if lo <= 0 <= hi else ".")


def _crossover_comparison(cross: dict) -> str:
    if not np.isfinite(cross["crossover_n"]):
        return (
            "no such crossover is resolved: neither family's fitted gap overtakes the other within the observed sizes"
        )
    lo, hi = cross["ci_low_n"], cross["ci_high_n"]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "a point crossover exists but is too unstable across bootstrap replicates to locate"
    wide = np.log10(hi) - np.log10(lo) > 1.0
    caveat = ", although its interval spans more than an order of magnitude" if wide else ""
    if hi >= 500 and lo <= 2000:
        return f"the estimated crossover is consistent with that range{caveat}"
    side = "below" if hi < 500 else "above"
    return f"the estimated crossover lies {side} that range{caveat}"


def _natural_sentence(nat: pd.DataFrame, n_resplit: int) -> str:
    if not isinstance(nat, pd.DataFrame) or nat.empty:
        return "The comparison run was not available, so the natural experiment was not evaluated."
    pairs = nat.groupby("family")["delta_pp"].agg(["median", "size"])
    parts = []
    for family, label in (
        ("Conventional ML", "conventional ML"),
        ("Ensemble (stacking / averaging)", "ensembles"),
        ("Uni-Mol (3D pretrained)", "Uni-Mol"),
        ("Chemprop v2 GNN", "Chemprop"),
    ):
        if family in pairs.index and pairs.at[family, "size"] >= MIN_NATURAL_PAIRS:
            parts.append(f"{pairs.at[family, 'median']:+.1f} for {label}")
    sparse = [
        f"{f.split(' (')[0].replace(' v2 GNN', '')} ({int(r['size'])} of {n_resplit})"
        for f, r in pairs.iterrows()
        if r["size"] < MIN_NATURAL_PAIRS
    ]
    if not parts:
        return "No family was valid on enough of these datasets in both runs to summarise."
    first_value, first_label = parts[0].split(" for ", 1)
    text = f"Moving them to scaffold splits changed the median family gap by {first_value} percentage points for "
    text += first_label + "".join(f", {p}" for p in parts[1:-1]) + (f" and {parts[-1]}." if len(parts) > 1 else ".")
    if sparse:
        text += f" Families valid in both runs on fewer than {MIN_NATURAL_PAIRS} of them are not summarised: "
        text += ", ".join(sparse) + "."
    return text + (
        f" With {n_resplit} datasets, and with the two runs also differing in hardware and model settings, "
        "this is descriptive only."
    )


def _grid_sentence(grid: pd.DataFrame, q_min, n_sig: int) -> str:
    if q_min is None:
        return "no correlation could be estimated"
    cell = (
        f"{q_min['family']} versus {FEATURE_LABELS.get(q_min['meta_feature'], q_min['meta_feature'])} "
        f"(ρ = {q_min['spearman_rho']:.2f}, 95% CI {q_min['ci_low']:.2f} to {q_min['ci_high']:.2f}, "
        f"q = {q_min['bh_q']:.2f})"
    )
    if n_sig == 0:
        return f"none survived at q < 0.10; the smallest q was for {cell}"
    count = NUMBER_WORDS.get(n_sig, str(n_sig))
    return f"{count} survived at q < 0.10, the strongest being {cell}"


VARIANT_LABELS = {"knn": "nearest-datasets", "ridge": "per-family ridge", "rf": "per-family random forest"}


def _variant_label(name: str) -> str:
    if name == "sbs":
        return "the single best family"
    if name.startswith("tree_cls"):
        return "the v1 winner tree"
    kind, block = name.split("|")
    return f"the {VARIANT_LABELS[kind]} model on {block}"


def _selector_v2_macros(s: dict) -> dict:
    nested, sbs = s["nested"], s["sbs"]
    diff, (lo, hi) = s["sbs_minus_nested_pct"], s["sbs_minus_nested_ci"]
    p = s["permutation"]["p_value"]
    common = (
        f"mean regret {nested['mean_regret_pct']:.1f}% versus {sbs['mean_regret_pct']:.1f}% for always choosing the "
        f"family with the best average record (difference {diff:.1f} percentage points, 95% CI {lo:.1f} to {hi:.1f}; "
        f"permutation p = {p:.3f})"
    )
    if lo > 0 and p < 0.05:
        verdict = (
            f"Nested selection beat the single best family ({common}), closing {100 * nested['gap_closed']:.0f}% of "
            "the gap to an oracle that always picks the winner."
        )
    elif hi < 0:
        verdict = f"Nested selection did worse than the single best family ({common})."
    else:
        verdict = (
            f"Nested selection did not reliably beat the single best family ({common}). Both lose little: a "
            f"randomly chosen valid family would cost {s['random_valid_family_mean_regret_pct']:.1f}% on average."
        )
    picked = max(s["nested_pick_counts"], key=s["nested_pick_counts"].get)
    return {
        "v2_nested_regret": f"{nested['mean_regret_pct']:.1f}%",
        "v2_sbs_regret": f"{sbs['mean_regret_pct']:.1f}%",
        "v2_nested_within5": f"{nested['within5_pct']:.0f}%",
        "v2_sbs_within5": f"{sbs['within5_pct']:.0f}%",
        "v2_random_regret": f"{s['random_valid_family_mean_regret_pct']:.1f}%",
        "v2_perm_p": f"{p:.3f}",
        "v2_n_candidates": str(s["n_candidates"]),
        "v2_verdict": verdict,
        "v2_most_picked": _variant_label(picked),
        "v2_most_picked_share": f"{100 * s['nested_pick_counts'][picked] / sum(s['nested_pick_counts'].values()):.0f}%",
    }


def _cv_leak_macros(c: dict) -> dict:
    top = c["top_models"]
    lo, hi = c["per_model_median_range_pct"]
    arm = c["arm_median_overstatement_pct"]
    return {
        "cvleak_median": f"{c['median_overstatement_all_benchmark_pct']:.1f}%",
        "cvleak_range": f"{lo:.1f}% to {hi:.1f}%",
        "cvleak_n_models": str(c["n_benchmark_models"]),
        "cvleak_top": ", ".join(top[:-1]) + f" and {top[-1]}" if len(top) > 1 else (top[0] if top else "n/a"),
        "cvleak_arm": f"{arm:.1f}%" if arm is not None else "an unmeasured level",
    }


def build_numbers(results: dict) -> dict:
    """meta_numbers.json: raw values plus the preformatted ``macros`` used in the manuscript text."""
    cross = results["crossover_primary"]
    cross2 = results["crossover_secondary"]
    diff = results["difficulty"]
    rec = results["recommender"]
    best_model = max(rec["models"], key=lambda k: rec["models"][k]["lodo_balanced_accuracy"])
    brec = rec["models"][best_model]
    grid = results["grid"]
    strongest = diff["strongest_feature"]
    snn = diff["per_feature"]["mean_snn"]["pooled"]
    size_rows = grid[grid["meta_feature"] == "log10_n_train"].set_index("family")
    q_min = grid.loc[grid["bh_q"].idxmin()] if grid["bh_q"].notna().any() else None
    n_sig = int((grid["bh_q"] < 0.10).sum())
    conv_size = size_rows.loc[CROSSOVER_PRIMARY[0]]
    has_cross = np.isfinite(cross["crossover_n"])
    nat = results["natural_experiment"]
    nat_family = (
        nat.groupby("family")["delta_pp"].median().to_dict() if isinstance(nat, pd.DataFrame) and len(nat) else {}
    )
    macros = {
        "n_datasets": str(results["n_datasets"]),
        "n_grid_cells": str(len(grid)),
        "n_grid_significant": str(n_sig),
        "grid_min_q": f"{q_min['bh_q']:.2f}" if q_min is not None else "n/a",
        "grid_min_q_cell": f"{q_min['family']} vs {q_min['meta_feature']}" if q_min is not None else "n/a",
        "rho_size_conventional": f"{conv_size['spearman_rho']:.2f}",
        "rho_size_conventional_ci": _fmt_ci(conv_size["ci_low"], conv_size["ci_high"]),
        "q_size_conventional": f"{conv_size['bh_q']:.2f}",
        "crossover_n": _fmt_n(cross["crossover_n"]),
        "crossover_ci": (
            f"{_fmt_n(cross['ci_low_n'])} to {_fmt_n(cross['ci_high_n'])}"
            if np.isfinite(cross["ci_low_n"])
            else "not estimable"
        ),
        "crossover_p_in_range": f"{100 * cross['p_cross_in_range']:.0f}%",
        "crossover_secondary_n": _fmt_n(cross2["crossover_n"]),
        "crossover_secondary_p_in_range": f"{100 * cross2['p_cross_in_range']:.0f}%",
        "crossover_statement": (
            f"the fitted gap curves cross at about {_fmt_n(cross['crossover_n'])} training molecules "
            f"(95% CI {_fmt_n(cross['ci_low_n'])} to {_fmt_n(cross['ci_high_n'])}; a crossing inside the observed "
            f"size range occurred in {100 * cross['p_cross_in_range']:.0f}% of dataset-bootstrap replicates)"
            if has_cross and np.isfinite(cross["ci_low_n"])
            else "the fitted gap curves do not cross inside the observed size range "
            f"(a crossing occurred in only {100 * cross['p_cross_in_range']:.0f}% of dataset-bootstrap replicates)"
        ),
        "rho_snn": f"{snn['rho']:.2f}",
        "rho_snn_ci": _fmt_ci(*snn["ci"]),
        "p_snn": f"{snn['p']:.3f}",
        "strongest_difficulty_feature": strongest.replace("_", " "),
        "rho_strongest": f"{diff['per_feature'][strongest]['pooled']['rho']:.2f}",
        "rho_strongest_ci": _fmt_ci(*diff["per_feature"][strongest]["pooled"]["ci"]),
        "k": str(len(rec["predictors"])),
        "k_words": NUMBER_WORDS.get(len(rec["predictors"]), str(len(rec["predictors"]))),
        "lodo_model": MODEL_LABELS[best_model],
        "lodo_bacc": f"{brec['lodo_balanced_accuracy']:.2f}",
        "lodo_bacc_ci": _fmt_ci(*brec["lodo_balanced_accuracy_ci"]),
        "baseline_bacc": f"{rec['majority_balanced_accuracy']:.2f}",
        "lodo_perm_p": f"{brec['permutation_p']:.3f}",
        "lodo_perm_null": f"{brec['permutation_null_balanced_accuracy_mean']:.2f}",
        "n_resplit": str(results["n_resplit"]),
        "size_sentence": _size_sentence(conv_size),
        "snn_sentence": _snn_sentence(snn),
        "strongest_sentence": _strongest_sentence(strongest, diff["per_feature"][strongest]["pooled"]),
        "crossover_comparison": _crossover_comparison(cross),
        "natural_sentence": _natural_sentence(nat, results["n_resplit"]),
        "grid_sentence": _grid_sentence(grid, q_min, n_sig),
        "recommender_sentence": (
            "The recommender therefore beats both baselines, but its interval is wide and it should be read as a "
            "hypothesis about which properties matter, not as a deployable selector."
            if brec["lodo_balanced_accuracy_ci"][0] > rec["majority_balanced_accuracy"] and brec["permutation_p"] < 0.05
            else "Its interval overlaps the baselines, so dataset meta-features alone do not reliably identify the "
            "winning family at this sample size; the per-dataset winner remains an empirical question, which is the "
            "case for benchmarking many families on every dataset."
        ),
        **_selector_v2_macros(results["selector_v2"]["summary"]),
        **_cv_leak_macros(results["cv_leak"]),
        "sens_same_sign": f"{100 * results['sensitivity']['same_sign_fraction']:.0f}%",
        "sens_sig_agree": f"{100 * results['sensitivity']['significance_agreement_fraction']:.0f}%",
    }
    return {
        "run_dir": results["run_dir"],
        "n_datasets": results["n_datasets"],
        "seed": SEED,
        "n_bootstrap": st.N_BOOT,
        "n_permutation": st.N_PERM,
        "crossover_primary": {k: v for k, v in cross.items()},
        "crossover_secondary": {k: v for k, v in cross2.items()},
        "difficulty": {k: v for k, v in diff["per_feature"].items()},
        "difficulty_strongest_feature": strongest,
        "recommender": {k: v for k, v in rec.items() if k != "models"}
        | {"models": {name: {k: v for k, v in m.items() if k != "predictions"} for name, m in rec["models"].items()}},
        "within5_pct_by_family": results["within5"],
        "trend_families": results["trend_families"],
        "chemprop_valid_datasets": results["chemprop_valid"],
        "winner_groups": results["winner_groups"],
        "natural_experiment_median_delta_pp": nat_family,
        "feature_selection_vs_diversity": results["feature_selection"],
        "sensitivity": results["sensitivity"],
        "cv_leak": results["cv_leak"],
        "selector_v2": results["selector_v2"]["summary"]
        | {"variants": results["selector_v2"]["table"].set_index("variant").to_dict(orient="index")},
        "macros": macros,
    }


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else round(float(value), 10)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(type(value))


def _clean(obj):
    """Round floats and replace NaN/inf with None so the JSON is byte-stable and valid."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), 10) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def run_meta_analysis(
    out_dir: Path | str = io.MANUSCRIPT_ASSETS,
    run_dir: Path | str = io.DEFAULT_RUN_DIR,
    family_best_path: Path | str | None = None,
    comparison_best_path: Path | str | None = None,
    partitions_path: Path | str = io.PARTITIONS_PATH,
    make_figures: bool = True,
    progress: bool = False,
    include_phase7: bool = False,
) -> dict:
    """Run phases 1-5 and write catalog, tables, figures and ``meta_numbers.json`` under ``out_dir``."""
    out_dir = Path(out_dir)
    table_dir, fig_dir = out_dir / "tables", out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    family_best_path = Path(family_best_path or io.FAMILY_BEST_PATH)
    comparison_best_path = Path(
        comparison_best_path or family_best_path.with_name("figure6_family_best_models_comparison_run.csv")
    )

    summary = io.load_dataset_summary(run_dir)
    partitions = io.load_partitions(partitions_path)
    mismatches = io.verify_partition_hashes(partitions, summary)
    if mismatches:
        raise ValueError(f"partitions do not match the run's split hashes: {mismatches}")

    family_best = gm.load_family_best(family_best_path)
    gap_pct = gm.load_family_gap_matrix(family_best)
    winners = gm.winner_family(family_best)

    catalog = mf.compute_meta_feature_catalog(partitions, summary, seed=SEED, progress=progress)
    # The analysis metric (what Fig 6 ranks on) replaces the catalog's primary_metric, which is wrong
    # for some datasets (e.g. binary tasks catalogued as rmse).
    catalog["primary_metric"] = catalog["dataset"].map(winners.set_index("dataset")["analysis_metric"])
    catalog_alt = mf.compute_meta_feature_catalog(partitions, summary, seed=SEED, **mf.SENSITIVITY_FP)
    catalog = add_derived_features(catalog)
    catalog_alt = add_derived_features(catalog_alt)

    families = gm.trend_families(gap_pct)
    grid = correlation_grid(catalog, gap_pct, families, GRID_FEATURES)
    grid_alt = correlation_grid(catalog_alt, gap_pct, families, FINGERPRINT_DEPENDENT)

    cat = catalog.set_index("dataset").loc[gap_pct.index]
    crossover_primary = st.crossover_estimate(
        cat["n_train"].astype(float), gap_pct[CROSSOVER_PRIMARY[0]], gap_pct[CROSSOVER_PRIMARY[1]], seed=SEED
    )
    crossover_secondary = st.crossover_estimate(
        cat["n_train"].astype(float), gap_pct[CROSSOVER_SECONDARY[0]], gap_pct[CROSSOVER_SECONDARY[1]], seed=SEED
    )

    difficulty = difficulty_analysis(catalog, gm.load_achievable_best(run_dir))

    resplit: list[str] = []
    nat = pd.DataFrame()
    if comparison_best_path.exists():
        numbers_path = family_best_path.parent.parent / "manuscript_numbers.json"
        if numbers_path.exists():
            resplit = (
                io.load_manuscript_numbers(numbers_path)
                .get("run_comparison_vs_rtx", {})
                .get("datasets_respilt_to_scaffold", [])
            )
        nat = natural_experiment(family_best, pd.read_csv(comparison_best_path), resplit)

    target = winners.set_index("dataset")["winner_group"]
    recommender = st.lodo_recommender(catalog.set_index("dataset")[RECOMMENDER_FEATURES], target, seed=SEED)

    fs_frame, fs_result = feature_selection_vs_diversity(catalog, io.load_selected_feature_families(run_dir))

    from qsarena.meta_analysis import selector_features as sf

    v2 = selector_v2(catalog, sf.compute_selector_catalog(partitions), sf.compute_landmarks(run_dir), gap_pct)

    results = {
        "run_dir": Path(run_dir).name,
        "n_datasets": int(len(catalog)),
        "catalog": catalog,
        "catalog_alt": catalog_alt,
        "gap_pct": gap_pct,
        "winners": winners,
        "grid": grid,
        "grid_alt": grid_alt,
        "trend_families": families,
        "crossover_primary": crossover_primary,
        "crossover_secondary": crossover_secondary,
        "difficulty": difficulty,
        "natural_experiment": nat,
        "n_resplit": len(resplit),
        "recommender": recommender,
        "selector_v2": v2,
        "cv_leak": cv_leak.summarize(run_dir),
        "feature_selection": fs_result,
        "feature_selection_frame": fs_frame,
        "within5": within5_summary(gap_pct),
        "chemprop_valid": int(gm.chemprop_valid(gap_pct).sum()),
        "winner_groups": winners["winner_group"].value_counts().sort_index().to_dict(),
        "sensitivity": sensitivity_summary(grid, grid_alt, catalog, catalog_alt),
    }
    numbers = _clean(build_numbers(results))

    from qsarena.meta_analysis import tables

    tables.write_all(results, table_dir)
    if make_figures:
        from qsarena.meta_analysis import figures

        numbers["figures"] = _clean(figures.write_all(results, numbers, fig_dir))
    if include_phase7:
        # Optional learning curves (GPU-trained elsewhere); off by default and never trains here.
        from qsarena.meta_analysis import phase7

        curves = phase7.curve_table(phase7.load_curves(), run_dir)
        curves.to_csv(table_dir / "meta_phase7_learning_curves.csv", index=False, float_format="%.10g")
        numbers["phase7"] = _clean(phase7.summarize(curves))
        if make_figures:
            from qsarena.meta_analysis import figures

            numbers.setdefault("figures", {})["S2"] = _clean(figures.figure_s2(curves, fig_dir))
    (out_dir / "meta_numbers.json").write_text(
        json.dumps(numbers, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8"
    )
    results["numbers"] = numbers
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(io.MANUSCRIPT_ASSETS))
    parser.add_argument("--run-dir", default=str(io.DEFAULT_RUN_DIR))
    parser.add_argument("--family-best", default=None, help="Fig 6 source CSV (default: manuscript_assets/tables)")
    parser.add_argument("--build-partitions", action="store_true", help="Extract partitions from predictions.csv")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--include-phase7", action="store_true", help="Add learning curves from meta_phase7_gpu_patch/ (off by default)"
    )
    parser.add_argument("--sync-manuscript", action="store_true", help="Render section text into the manuscript")
    args = parser.parse_args(argv)
    if args.build_partitions:
        frame = io.build_partitions(args.run_dir)
        print(f"wrote {io.PARTITIONS_PATH} ({len(frame)} rows, {frame['dataset'].nunique()} datasets)")
        return 0
    results = run_meta_analysis(
        args.out_dir,
        args.run_dir,
        args.family_best,
        make_figures=not args.no_figures,
        progress=True,
        include_phase7=args.include_phase7,
    )
    print(f"meta-analysis: {results['n_datasets']} datasets -> {Path(args.out_dir) / 'meta_numbers.json'}")
    if args.sync_manuscript:
        from qsarena.meta_analysis import text

        for line in text.sync_manuscript(results["numbers"]):
            print(line)
    return 0
