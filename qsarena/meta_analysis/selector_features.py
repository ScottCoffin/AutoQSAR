"""Extra meta-feature blocks for the v2 family selector (docs/meta_analysis/SELECTOR_V2_PLAN.md).

F1 chemistry, F2 label landscape and F3 landmarkers are all computed from training-partition data
only: training molecules and targets, and the benchmark's own training-set cross-validation scores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from qsarena.meta_analysis import io
from qsarena.meta_analysis.meta_features import CHUNK, _pairwise_similarity, fingerprints

F1 = ["chem_mw_mean", "chem_mw_sd", "chem_logp_mean", "chem_tpsa_mean", "chem_fsp3_mean", "chem_arom_rings_mean"]
F2 = ["landscape_smoothness", "landscape_cliff_fraction"]
F3 = ["lm_linear_gap", "lm_knn_gap", "lm_mlp_gap", "lm_tabpfn_gap", "lm_cv_spread"]
KNN_K = 5
CLIFF_SIM = 0.7

LINEAR = {"ElasticNetCV", "LogisticRegression"}
TREES = {"Random forest", "Extra trees", "HistGradientBoosting", "XGBoost", "LightGBM", "CatBoost"}
KNN_VOTING = {"Voting Regressor (KNN, SVM)", "Voting Classifier (KNN, SVM)"}
MLP = {"ChemML MLP (PyTorch)"}
TABPFN = {"TabPFNRegressor", "TabPFNClassifier"}
HIGHER_IS_BETTER = {"roc_auc", "auprc", "r2", "spearman", "balanced_accuracy", "mcc", "accuracy"}


def chemistry_features(smiles) -> dict:
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    mw = np.array([Descriptors.MolWt(m) for m in mols])
    return {
        "chem_mw_mean": float(mw.mean()),
        "chem_mw_sd": float(mw.std(ddof=1)) if len(mw) > 1 else float("nan"),
        "chem_logp_mean": float(np.mean([Crippen.MolLogP(m) for m in mols])),
        "chem_tpsa_mean": float(np.mean([rdMolDescriptors.CalcTPSA(m) for m in mols])),
        "chem_fsp3_mean": float(np.mean([rdMolDescriptors.CalcFractionCSP3(m) for m in mols])),
        "chem_arom_rings_mean": float(np.mean([Lipinski.NumAromaticRings(m) for m in mols])),
    }


def _train_neighbours(fp: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and Tanimoto similarities of each training molecule's k nearest *other* training molecules."""
    n = len(fp)
    k = min(k, n - 1)
    idx = np.empty((n, k), dtype=np.int64)
    sim = np.empty((n, k), dtype=float)
    for start in range(0, n, CHUNK):
        block = _pairwise_similarity(fp[start : start + CHUNK], fp)
        rows = np.arange(start, min(start + CHUNK, n))
        block[np.arange(len(rows)), rows] = -1.0  # exclude self
        part = np.argpartition(-block, k - 1, axis=1)[:, :k]
        part_sim = np.take_along_axis(block, part, axis=1)
        order = np.argsort(-part_sim, axis=1)
        idx[rows] = np.take_along_axis(part, order, axis=1)
        sim[rows] = np.take_along_axis(part_sim, order, axis=1)
    return idx, sim


def landscape_features(fp: np.ndarray, y, task: str, k: int = KNN_K) -> dict:
    """Modelability of the training labels in fingerprint space.

    Regression: leave-one-out, similarity-weighted kNN R^2. Classification: MODI, the mean over
    classes of the share of molecules whose nearest neighbour shares their class. Cliff fraction:
    nearest-neighbour pairs with Tanimoto >= 0.7 whose labels differ (another class, or
    |delta y| > 1 training SD), as a share of all such close pairs.
    """
    y = np.asarray(y, dtype=float)
    idx, sim = _train_neighbours(fp, k)
    nn_y = y[idx[:, 0]]
    close = sim[:, 0] >= CLIFF_SIM
    if task == "classification":
        same = nn_y == y
        smooth = float(np.mean([same[y == c].mean() for c in np.unique(y)]))
        differs = nn_y != y
    else:
        weights = np.clip(sim, 1e-6, None)
        pred = (y[idx] * weights).sum(axis=1) / weights.sum(axis=1)
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        smooth = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        differs = np.abs(nn_y - y) > y.std(ddof=1)
    cliff = float(differs[close].mean()) if close.any() else 0.0
    return {"landscape_smoothness": smooth, "landscape_cliff_fraction": cliff}


def compute_selector_catalog(partitions: pd.DataFrame, progress: bool = False) -> pd.DataFrame:
    rows = []
    for dataset, group in partitions.groupby("dataset", sort=True):
        train = group.loc[group["split"] == "train"].sort_values("row_index")
        task = io.infer_task(train["observed"])
        fp = fingerprints(train["smiles"])
        rows.append(
            {
                "dataset": dataset,
                **chemistry_features(train["smiles"]),
                **landscape_features(fp, train["observed"], task),
            }
        )
        if progress:
            print(f"[selector-features] {dataset}", flush=True)
    return pd.DataFrame(rows)[["dataset", *F1, *F2]]


def _oriented_gap(value: float, reference: float, higher_better: bool) -> float:
    if not (np.isfinite(value) and np.isfinite(reference)) or reference == 0:
        return float("nan")
    return (reference - value) / abs(reference) if higher_better else (value - reference) / abs(reference)


def compute_landmarks(run_dir: Path | str = io.DEFAULT_RUN_DIR) -> pd.DataFrame:
    """F3 from each dataset's training-set CV scores (``cv_primary``); positive gap = worse than the best tree."""
    rows = []
    for d in io.dataset_dirs(run_dir):
        wanted = {"model", "error", "cv_primary", "primary_metric"}
        frame = pd.read_csv(d / "metrics.csv", usecols=lambda c: c in wanted)
        frame = frame[frame["error"].isna() & frame["cv_primary"].notna()].drop_duplicates("model", keep="last")
        metric = str(frame["primary_metric"].dropna().iloc[0]) if frame["primary_metric"].notna().any() else ""
        higher = metric in HIGHER_IS_BETTER
        scores = frame.set_index("model")["cv_primary"].astype(float)

        def best(names, scores=scores, higher=higher):
            s = scores[scores.index.isin(names)]
            return float(s.max() if higher else s.min()) if len(s) else float("nan")

        tree = best(TREES)
        oriented = scores if higher else -scores
        rows.append(
            {
                "dataset": d.name,
                "lm_linear_gap": _oriented_gap(best(LINEAR), tree, higher),
                "lm_knn_gap": _oriented_gap(best(KNN_VOTING), tree, higher),
                "lm_mlp_gap": _oriented_gap(best(MLP), tree, higher),
                "lm_tabpfn_gap": _oriented_gap(best(TABPFN), tree, higher),
                "lm_cv_spread": float(oriented.std(ddof=1) / abs(oriented.mean())) if len(oriented) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)[["dataset", *F3]]
