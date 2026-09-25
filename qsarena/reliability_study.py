"""
qsarena.reliability_study — applicability domain, conformal calibration and feature attribution
on the 22 official TDC ADMET Benchmark Group splits.

Why a reference model. The canonical benchmark run did not keep per-molecule predictions in the
repository (``predictions.csv`` is gitignored), and re-running 28 models x 22 datasets to recover
them costs GPU-days. Applicability domain is a property of the TRAINING chemical space, not of a
particular learner, so this study fits one fixed, cheap, deterministic reference model per
dataset -- a random forest on Morgan fingerprints + RDKit 2D descriptors, i.e. the
conventional-ML family -- and measures, on the untouched official test set:

  * structural AD coverage by two methods (Roy standardization on descriptors; kNN Tanimoto),
  * test error inside vs outside each domain,
  * split-conformal interval coverage (regression) and LAC set coverage, ECE and Brier score
    (classification), with the calibration set carved from train_val only,
  * the reference model's top features, and a QMRF-style report for one dataset.

Results describe the reference model, not the per-dataset benchmark winner, and the manuscript
says so. Nothing here reads the test labels before the final scoring step.

    python -m qsarena.reliability_study --out results/reliability_tdc22
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qsarena.applicability_domain import knn_similarity_ad, standardization_ad
from qsarena.interpretability import feature_importance, top_features
from qsarena.provenance import split_hash, write_environment_manifest
from qsarena.qmrf import write_qmrf_report
from qsarena.uncertainty import (
    SplitConformalRegressor,
    brier_score,
    conformal_prediction_sets,
    expected_calibration_error,
    probability_confidence,
    reliability_curve,
)

__all__ = ["run_dataset", "run_study", "featurize", "main"]

TDC22 = (
    "ames", "bbb_martins", "bioavailability_ma", "caco2_wang", "clearance_hepatocyte_az",
    "clearance_microsome_az", "cyp2c9_substrate_carbonmangels", "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels", "cyp2d6_veith", "cyp3a4_substrate_carbonmangels",
    "cyp3a4_veith", "dili", "half_life_obach", "herg", "hia_hou", "ld50_zhu",
    "lipophilicity_astrazeneca", "pgp_broccatelli", "ppbr_az", "solubility_aqsoldb", "vdss_lombardo",
)

#: Endpoint descriptions and units for the QMRF report (TDC documentation).
ENDPOINTS = {
    "caco2_wang": ("Caco-2 cell effective permeability", "log10(cm/s)"),
    "lipophilicity_astrazeneca": ("Octanol/water distribution coefficient at pH 7.4", "logD"),
    "solubility_aqsoldb": ("Aqueous solubility", "log10(mol/L)"),
    "ld50_zhu": ("Acute rat oral toxicity (LD50)", "-log10(mol/kg)"),
    "ppbr_az": ("Human plasma protein binding rate", "% bound"),
    "vdss_lombardo": ("Volume of distribution at steady state", "L/kg"),
    "half_life_obach": ("Human half-life", "h"),
    "clearance_hepatocyte_az": ("Hepatocyte intrinsic clearance", "uL/min/1e6 cells"),
    "clearance_microsome_az": ("Microsomal intrinsic clearance", "mL/min/g"),
}

MORGAN_BITS = 2048
#: An in/out error ratio is only reported when both groups hold at least this many compounds; a
#: ratio over one or two out-of-domain molecules is noise, not evidence.
MIN_GROUP = 5


# ---------------------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------------------

def _descriptor_names() -> list[str]:
    from rdkit.Chem import Descriptors

    return [name for name, _ in Descriptors._descList]


def featurize(smiles: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Return (morgan_bits uint8, rdkit_descriptors float, descriptor_names, valid_mask)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = GetMorganGenerator(radius=2, fpSize=MORGAN_BITS)
    names = _descriptor_names()
    fps = np.zeros((len(smiles), MORGAN_BITS), dtype=np.uint8)
    desc = np.full((len(smiles), len(names)), np.nan, dtype=float)
    valid = np.zeros(len(smiles), dtype=bool)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        valid[i] = True
        fps[i] = gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
        values = Descriptors.CalcMolDescriptors(mol, missingVal=np.nan)
        desc[i] = [values.get(n, np.nan) for n in names]
    desc = np.where(np.isfinite(desc) & (np.abs(desc) < 1e10), desc, np.nan)
    return fps, desc, names, valid


def _cached_features(smiles: list[str], cache_path: Path | None):
    if cache_path is not None and cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        if data["n"] == len(smiles) and str(data["hash"]) == split_hash(smiles):
            return data["fps"], data["desc"], list(data["names"]), data["valid"]
    fps, desc, names, valid = featurize(smiles)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path, fps=fps, desc=desc, names=np.array(names), valid=valid,
            n=len(smiles), hash=split_hash(smiles),
        )
    return fps, desc, names, valid


# ---------------------------------------------------------------------------------------------
# One dataset
# ---------------------------------------------------------------------------------------------

def _is_binary(y: np.ndarray) -> bool:
    return set(np.unique(y[np.isfinite(y)])).issubset({0.0, 1.0})


def _error_ratio(err: np.ndarray, in_mask: np.ndarray) -> dict[str, float]:
    n_in, n_out = int(in_mask.sum()), int((~in_mask).sum())
    e_in = float(np.mean(err[in_mask])) if n_in else float("nan")
    e_out = float(np.mean(err[~in_mask])) if n_out else float("nan")
    ratio = e_out / e_in if n_in and n_out and e_in > 0 else float("nan")
    return {"n_in": n_in, "n_out": n_out, "error_in": e_in, "error_out": e_out, "error_ratio_out_in": ratio}


def run_dataset(
    name: str,
    train_val: pd.DataFrame,
    test: pd.DataFrame,
    seed: int = 0,
    n_estimators: int = 300,
    alpha: float = 0.1,
    cal_fraction: float = 0.2,
    knn_k: int = 5,
    knn_quantile: float = 0.95,
    n_jobs: int = -1,
    cache_dir: Path | None = None,
) -> dict:
    """Fit the reference model on one split and return rows for every output table."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, roc_auc_score
    from sklearn.model_selection import train_test_split

    tv_smiles = train_val["Drug"].astype(str).str.strip().tolist()
    te_smiles = test["Drug"].astype(str).str.strip().tolist()
    fp_tv, d_tv, dnames, ok_tv = _cached_features(tv_smiles, cache_dir / f"{name}_train_val.npz" if cache_dir else None)
    fp_te, d_te, _, ok_te = _cached_features(te_smiles, cache_dir / f"{name}_test.npz" if cache_dir else None)
    y_tv = train_val["Y"].to_numpy(dtype=float)
    y_te = test["Y"].to_numpy(dtype=float)
    keep_tv = ok_tv & np.isfinite(y_tv)
    keep_te = ok_te & np.isfinite(y_te)
    fp_tv, d_tv, y_tv = fp_tv[keep_tv], d_tv[keep_tv], y_tv[keep_tv]
    fp_te, d_te, y_te = fp_te[keep_te], d_te[keep_te], y_te[keep_te]
    te_smiles_kept = [s for s, k in zip(te_smiles, keep_te) if k]

    # Impute descriptors with TRAINING medians only.
    med = np.nanmedian(d_tv, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    d_tv = np.where(np.isfinite(d_tv), d_tv, med)
    d_te = np.where(np.isfinite(d_te), d_te, med)
    X_tv = np.hstack([fp_tv.astype(np.float32), d_tv.astype(np.float32)])
    X_te = np.hstack([fp_te.astype(np.float32), d_te.astype(np.float32)])
    feat_names = [f"morgan_r2_bit{i}" for i in range(MORGAN_BITS)] + [f"rdkit_{n}" for n in dnames]

    classification = _is_binary(y_tv)
    idx = np.arange(len(y_tv))
    idx_fit, idx_cal = train_test_split(
        idx, test_size=cal_fraction, random_state=seed, stratify=y_tv if classification else None
    )

    if classification:
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_features="sqrt", min_samples_leaf=1,
            class_weight=None, n_jobs=n_jobs, random_state=seed,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_features=0.3, min_samples_leaf=1, n_jobs=n_jobs, random_state=seed,
        )
    model.fit(X_tv[idx_fit], y_tv[idx_fit])

    # ---- structural AD (training = the model's fitting partition) ------------------------------
    std_ad = standardization_ad(d_tv[idx_fit], d_te)
    knn_ad = knn_similarity_ad(fp_tv[idx_fit], fp_te, k=knn_k, quantile=knn_quantile)
    consensus_in = std_ad.in_domain & knn_ad.in_domain

    per_mol = pd.DataFrame({
        "dataset": name,
        "smiles": te_smiles_kept,
        "y_true": y_te,
        "ad_standardization_s_max": std_ad.s_max,
        "ad_standardization_s_new": std_ad.s_new,
        "ad_standardization_in_domain": std_ad.in_domain,
        "ad_knn_mean_tanimoto_distance": knn_ad.mean_knn_distance,
        "ad_knn_in_domain": knn_ad.in_domain,
        "ad_consensus_in_domain": consensus_in,
    })

    base = {
        "dataset": name,
        "task": "classification" if classification else "regression",
        "n_train_fit": int(len(idx_fit)),
        "n_calibration": int(len(idx_cal)),
        "n_test": int(len(y_te)),
        "knn_threshold_tanimoto_distance": knn_ad.threshold,
    }
    ad_rows, cal_row = [], dict(base)

    if classification:
        p_cal = model.predict_proba(X_tv[idx_cal])[:, 1]
        p_te = model.predict_proba(X_te)[:, 1]
        err = ((p_te >= 0.5).astype(float) != y_te).astype(float)  # 0/1 error at 0.5
        sq = (p_te - y_te) ** 2
        conf, reliable = probability_confidence(p_te, threshold=0.5)
        sets, q = conformal_prediction_sets(
            np.column_stack([1 - p_cal, p_cal]), y_tv[idx_cal].astype(int), np.column_stack([1 - p_te, p_te]), alpha
        )
        covered = sets[np.arange(len(y_te)), y_te.astype(int)]
        set_size = sets.sum(axis=1)
        per_mol = per_mol.assign(
            y_pred=(p_te >= 0.5).astype(int), proba_positive=p_te, abs_error=err,
            confidence=conf, reliable_confidence=reliable,
            conformal_set_size=set_size, conformal_covered=covered,
        )
        auc = roc_auc_score(y_te, p_te) if len(np.unique(y_te)) == 2 else float("nan")
        cal_row.update({
            "test_auroc": auc,
            "test_error_rate": float(err.mean()),
            "brier": brier_score(y_te, p_te),
            "ece_10bin": expected_calibration_error(y_te, p_te, n_bins=10),
            "alpha": alpha,
            "conformal_coverage": float(covered.mean()),
            "conformal_singleton_fraction": float(np.mean(set_size == 1)),
            "conformal_mean_set_size": float(set_size.mean()),
            "confidence_reliable_fraction": float(reliable.mean()),
        })
        for method, mask in [
            ("standardization", std_ad.in_domain),
            ("knn_tanimoto", knn_ad.in_domain),
            ("consensus", consensus_in),
            ("probability_confidence", reliable),
        ]:
            for metric, vec in [("error_rate", err), ("brier", sq)]:
                ad_rows.append({**base, "ad_method": method, "error_metric": metric,
                                "coverage": float(mask.mean()), **_error_ratio(vec, mask)})
        rel = reliability_curve(y_te, p_te, n_bins=10)
    else:
        tree_pred_cal = np.stack([t.predict(X_tv[idx_cal]) for t in model.estimators_])
        tree_pred_te = np.stack([t.predict(X_te) for t in model.estimators_])
        pred_cal, sig_cal = tree_pred_cal.mean(0), tree_pred_cal.std(0)
        pred_te, sig_te = tree_pred_te.mean(0), tree_pred_te.std(0)
        err = np.abs(pred_te - y_te)
        cr = SplitConformalRegressor(alpha=alpha).fit(y_tv[idx_cal], pred_cal, sigma_cal=sig_cal)
        lo, hi = cr.predict_interval(pred_te, sigma=sig_te)
        covered = (y_te >= lo) & (y_te <= hi)
        width = hi - lo
        # Reliability flag: interval no wider than the calibration set's median interval.
        width_cal = 2 * cr.half_width(sigma=sig_cal)
        reliable = width <= np.median(width_cal)
        per_mol = per_mol.assign(
            y_pred=pred_te, abs_error=err, tree_std=sig_te, interval_lower=lo, interval_upper=hi,
            conformal_covered=covered, reliable_interval=reliable,
        )
        y_range = float(np.ptp(y_tv)) or 1.0
        cal_row.update({
            "test_mae": float(mean_absolute_error(y_te, pred_te)),
            "alpha": alpha,
            "conformal_coverage": float(covered.mean()),
            "conformal_mean_width": float(width.mean()),
            "conformal_mean_width_over_train_range": float(width.mean() / y_range),
            "interval_reliable_fraction": float(reliable.mean()),
        })
        for method, mask in [
            ("standardization", std_ad.in_domain),
            ("knn_tanimoto", knn_ad.in_domain),
            ("consensus", consensus_in),
            ("conformal_interval_width", reliable),
        ]:
            ad_rows.append({**base, "ad_method": method, "error_metric": "mae",
                            "coverage": float(mask.mean()), **_error_ratio(err, mask)})
            # Covered-in vs covered-out: are the intervals honest outside the domain?
            ad_rows[-1]["conformal_coverage_in"] = float(covered[mask].mean()) if mask.any() else float("nan")
            ad_rows[-1]["conformal_coverage_out"] = float(covered[~mask].mean()) if (~mask).any() else float("nan")
        rel = None

    imp = feature_importance(model, feat_names, method="native")
    top = top_features(imp, n=10).assign(dataset=name)
    return {
        "ad_rows": ad_rows,
        "calibration_row": cal_row,
        "per_molecule": per_mol,
        "top_features": top,
        "reliability_curve": rel,
        "model": model,
    }


# ---------------------------------------------------------------------------------------------
# All datasets
# ---------------------------------------------------------------------------------------------

def _canonical_split_matches(canonical_run: Path | None, name: str, test_smiles: list[str], train_smiles: list[str]):
    """Compare this study's split against the canonical run's recorded counts and hashes."""
    if canonical_run is None:
        return {}
    mp = canonical_run / f"tdc_{name}" / "metrics.csv"
    if not mp.exists():
        return {"canonical_split_recorded": False}
    cols = ["n_train", "n_test", "split_train_hash", "split_test_hash"]
    m = pd.read_csv(mp, usecols=cols).dropna().iloc[0]
    from rdkit import Chem

    def canon(values):
        out = []
        for s in values:
            mol = Chem.MolFromSmiles(s)
            out.append(Chem.MolToSmiles(mol) if mol is not None else s)
        return out

    return {
        "canonical_split_recorded": True,
        "canonical_n_train": int(m.n_train),
        "canonical_n_test": int(m.n_test),
        "canonical_test_hash_matches": split_hash(canon(test_smiles)) == m.split_test_hash
        or split_hash(test_smiles) == m.split_test_hash,
        "canonical_train_hash_matches": split_hash(canon(train_smiles)) == m.split_train_hash
        or split_hash(train_smiles) == m.split_train_hash,
    }


def _plot_reliability(curves: dict[str, tuple], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # plotting is optional
        return
    names = sorted(curves)
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.4 * nrow), sharex=True, sharey=True)
    for ax, name in zip(np.ravel(axes), names):
        conf, freq, count = curves[name]
        mask = count > 0
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
        ax.plot(conf[mask], freq[mask], marker="o", ms=3, lw=1.2, color="#0072B2")
        ax.set_title(name.replace("_", " "), fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in np.ravel(axes)[len(names):]:
        ax.axis("off")
    fig.supxlabel("Mean predicted probability", fontsize=8)
    fig.supylabel("Observed frequency", fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def summarize(ad: pd.DataFrame, cal: pd.DataFrame) -> dict:
    """Headline numbers for the manuscript, computed from the output tables only."""
    out: dict = {"n_datasets": int(cal["dataset"].nunique()),
                 "n_regression": int((cal["task"] == "regression").sum()),
                 "n_classification": int((cal["task"] == "classification").sum())}
    primary = ad[((ad["task"] == "regression") & (ad["error_metric"] == "mae"))
                 | ((ad["task"] == "classification") & (ad["error_metric"] == "error_rate"))]
    reliability = primary[((primary.task == "regression") & (primary.ad_method == "conformal_interval_width"))
                          | ((primary.task == "classification") & (primary.ad_method == "probability_confidence"))]
    for method, sub_all in [("standardization", None), ("knn_tanimoto", None), ("consensus", None),
                            ("reliability", reliability)]:
        sub = sub_all if sub_all is not None else primary[primary["ad_method"] == method]
        both = sub[(sub["n_in"] >= MIN_GROUP) & (sub["n_out"] >= MIN_GROUP)].dropna(subset=["error_ratio_out_in"])
        out[method] = {
            "median_coverage": float(sub["coverage"].median()),
            "min_coverage": float(sub["coverage"].min()),
            "max_coverage": float(sub["coverage"].max()),
            "min_group_size": MIN_GROUP,
            "n_datasets_with_both_groups": int(len(both)),
            "n_datasets_out_error_higher": int((both["error_ratio_out_in"] > 1).sum()),
            "median_error_ratio_out_in": float(both["error_ratio_out_in"].median()) if len(both) else None,
            "median_pct_higher_error_out": float((both["error_ratio_out_in"].median() - 1) * 100) if len(both) else None,
            "median_pct_higher_error_out_regression": float(
                (both.loc[both.task == "regression", "error_ratio_out_in"].median() - 1) * 100
            ) if (both.task == "regression").any() else None,
            "median_pct_higher_error_out_classification": float(
                (both.loc[both.task == "classification", "error_ratio_out_in"].median() - 1) * 100
            ) if (both.task == "classification").any() else None,
        }
    reg = cal[cal["task"] == "regression"]
    cls = cal[cal["task"] == "classification"]
    out["conformal"] = {
        "alpha": float(cal["alpha"].iloc[0]),
        "nominal_coverage": float(1 - cal["alpha"].iloc[0]),
        "regression_median_coverage": float(reg["conformal_coverage"].median()) if len(reg) else None,
        "regression_min_coverage": float(reg["conformal_coverage"].min()) if len(reg) else None,
        "regression_max_coverage": float(reg["conformal_coverage"].max()) if len(reg) else None,
        "classification_median_coverage": float(cls["conformal_coverage"].median()) if len(cls) else None,
        "classification_min_coverage": float(cls["conformal_coverage"].min()) if len(cls) else None,
        "classification_max_coverage": float(cls["conformal_coverage"].max()) if len(cls) else None,
        "classification_median_ece": float(cls["ece_10bin"].median()) if len(cls) else None,
        "classification_median_brier": float(cls["brier"].median()) if len(cls) else None,
        "classification_median_singleton_fraction": float(cls["conformal_singleton_fraction"].median()) if len(cls) else None,
    }
    return out


def run_study(
    data_dir: Path,
    out_dir: Path,
    datasets: list[str] | None = None,
    seed: int = 0,
    n_estimators: int = 300,
    alpha: float = 0.1,
    canonical_run: Path | None = None,
    cache_dir: Path | None = None,
    qmrf_dataset: str | None = "caco2_wang",
    n_jobs: int = -1,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = list(datasets or TDC22)
    ad_rows, cal_rows, per_mol, tops, curves, split_rows = [], [], [], [], {}, []
    started = time.time()
    qmrf_inputs = None
    for name in datasets:
        tv_path, te_path = data_dir / name / "train_val.csv", data_dir / name / "test.csv"
        if not tv_path.exists() or not te_path.exists():
            print(f"[skip] {name}: split files not found under {data_dir}", flush=True)
            continue
        t0 = time.time()
        train_val, test = pd.read_csv(tv_path), pd.read_csv(te_path)
        res = run_dataset(name, train_val, test, seed=seed, n_estimators=n_estimators, alpha=alpha,
                          cache_dir=cache_dir, n_jobs=n_jobs)
        ad_rows += res["ad_rows"]
        cal_rows.append(res["calibration_row"])
        per_mol.append(res["per_molecule"])
        tops.append(res["top_features"])
        if res["reliability_curve"] is not None:
            curves[name] = res["reliability_curve"]
        split_rows.append({
            "dataset": name,
            "n_train_val": len(train_val),
            "n_test": len(test),
            "train_val_hash": split_hash(train_val["Drug"]),
            "test_hash": split_hash(test["Drug"]),
            **_canonical_split_matches(canonical_run, name, test["Drug"].astype(str).str.strip().tolist(),
                                       train_val["Drug"].astype(str).str.strip().tolist()),
        })
        if name == qmrf_dataset:
            qmrf_inputs = (name, res)
        print(f"[done] {name}: {res['calibration_row']['task']}, {time.time() - t0:.1f}s", flush=True)

    ad = pd.DataFrame(ad_rows)
    cal = pd.DataFrame(cal_rows)
    ad.to_csv(out_dir / "applicability_domain.csv", index=False)
    cal.to_csv(out_dir / "calibration.csv", index=False)
    pd.concat(per_mol, ignore_index=True).to_csv(out_dir / "per_molecule_predictions.csv", index=False)
    pd.concat(tops, ignore_index=True)[["dataset", "rank", "feature", "importance", "method"]].to_csv(
        out_dir / "feature_importance_top10.csv", index=False
    )
    pd.DataFrame(split_rows).to_csv(out_dir / "split_hashes.csv", index=False)
    if curves:
        _plot_reliability(curves, out_dir / "reliability_diagrams")

    summary = summarize(ad, cal)
    summary["reference_model"] = f"RandomForest ({n_estimators} trees) on Morgan r=2 {MORGAN_BITS}-bit + RDKit 2D descriptors"
    summary["seed"] = seed
    summary["calibration_fraction_of_train_val"] = 0.2
    summary["wall_clock_seconds"] = round(time.time() - started, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if qmrf_inputs is not None:
        _write_sample_qmrf(qmrf_inputs, ad, cal, split_rows, out_dir, seed, n_estimators, alpha)

    write_environment_manifest(out_dir, extra={"study": "qsarena.reliability_study", "seed": seed,
                                               "n_estimators": n_estimators, "alpha": alpha,
                                               "datasets": datasets})
    return summary


def _write_sample_qmrf(qmrf_inputs, ad, cal, split_rows, out_dir, seed, n_estimators, alpha):
    import qsarena

    name, res = qmrf_inputs
    c = cal[cal.dataset == name].iloc[0]
    split = next(r for r in split_rows if r["dataset"] == name)
    a = ad[(ad.dataset == name) & (ad.ad_method == "consensus")].iloc[0]
    endpoint, units = ENDPOINTS.get(name, (name, "see TDC documentation"))
    model = res["model"]
    run = {
        "dataset": f"TDC ADMET Benchmark Group: {name}",
        "endpoint": endpoint,
        "task_type": c.task,
        "units": units,
        "data_source": "Therapeutics Data Commons, official train_val/test split (scaffold)",
        "model_name": f"Random forest reference model ({n_estimators} trees)",
        "features": f"Morgan fingerprint (radius 2, {MORGAN_BITS} bits) + RDKit 2D descriptors",
        "software_version": f"qsarena {qsarena.__version__}",
        "random_seed": seed,
        "split": {"strategy": "TDC official scaffold split", "train_hash": split["train_val_hash"],
                  "test_hash": split["test_hash"]},
        "reproduction": "python -m qsarena.reliability_study --out results/reliability_tdc22",
        "applicability_domain": {
            "method": "consensus of Roy standardization (RDKit descriptors, 3 SD) and 5-NN Tanimoto "
                      "distance (95th percentile of training leave-one-out distances)",
            "test_coverage": float(a.coverage),
            "error_in_domain": float(a.error_in),
            "error_out_of_domain": float(a.error_out),
            "reliability": f"split-conformal {int(round((1 - alpha) * 100))}% intervals, "
                           f"test coverage {c.conformal_coverage:.3f}",
        },
        "metrics": {
            "train": {"n_fit": int(c.n_train_fit),
                      "out_of_bag_note": "fit on 80% of train_val; 20% held for calibration"},
            "internal_validation": {"n_calibration": int(c.n_calibration),
                                    "conformal_quantile_source": "calibration partition of train_val"},
            "test": {"n_test": int(c.n_test), "mae": float(c.get("test_mae", float("nan")))},
            "uncertainty": {"alpha": alpha, "coverage": float(c.conformal_coverage)},
        },
        "interpretation": {
            "status": "feature",
            "top_features": res["top_features"][["rank", "feature", "importance"]].to_dict(orient="records"),
            "note": "Impurity-based importances of the reference model; statistical association, not mechanism.",
        },
        "caveats": [
            "Reference model for applicability-domain and calibration analysis; not the benchmark's selected model.",
            "Single seed and single official split.",
        ],
    }
    del model
    write_qmrf_report(run, out_dir, stem=f"qmrf_{name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/admet_group"))
    parser.add_argument("--out", type=Path, default=Path("results/reliability_tdc22"))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--canonical-run", type=Path,
                        default=Path("benchmark_results/autoqsar_benchmark_20260623_153839"))
    parser.add_argument("--cache-dir", type=Path, default=Path("model_cache/reliability_features"))
    parser.add_argument("--qmrf-dataset", default="caco2_wang")
    parser.add_argument("--resummarize", action="store_true",
                        help="rebuild summary.json from the CSVs already in --out, without refitting")
    args = parser.parse_args(argv)
    if args.resummarize:
        old = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
        summary = summarize(pd.read_csv(args.out / "applicability_domain.csv"), pd.read_csv(args.out / "calibration.csv"))
        summary.update({k: old[k] for k in ("reference_model", "seed", "calibration_fraction_of_train_val",
                                             "wall_clock_seconds") if k in old})
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0
    canonical = args.canonical_run if args.canonical_run and args.canonical_run.exists() else None
    summary = run_study(args.data_dir, args.out, args.datasets, seed=args.seed, n_estimators=args.n_estimators,
                        alpha=args.alpha, canonical_run=canonical, cache_dir=args.cache_dir,
                        qmrf_dataset=args.qmrf_dataset, n_jobs=args.n_jobs)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
