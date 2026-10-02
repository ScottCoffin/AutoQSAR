"""Per-dataset meta-features computed on the benchmark's own train/test partition.

Fingerprints (``make_morgan_matrix``) and Bemis-Murcko scaffold keys (``murcko_scaffold_key``) come
from the workflow core, and Tanimoto distances come from ``qsarena.applicability_domain``, so the
meta-features use the same chemistry code the benchmark used. Tanimoto similarity on binary
fingerprints is 1 - Jaccard distance: they are the same metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy import stats

from qsarena.applicability_domain import tanimoto_distance_matrix

try:
    from portable_colab_qsar_bundle.qsar_workflow_core import murcko_scaffold_key
except ImportError:  # pragma: no cover - script-style checkout
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "portable_colab_qsar_bundle"))
    from qsar_workflow_core import murcko_scaffold_key

from qsarena.meta_analysis.io import infer_task

RDLogger.DisableLog("rdApp.warning")

#: Default fingerprint (ECFP4 = Morgan radius 2) and the sensitivity setting (ECFP6, 4096 bits).
DEFAULT_FP = {"radius": 2, "n_bits": 2048}
SENSITIVITY_FP = {"radius": 3, "n_bits": 4096}
OOD_SNN_THRESHOLD = 0.40
EXACT_DIVERSITY_MAX_N = 2000
DIVERSITY_PAIR_SAMPLE = 20_000
CHUNK = 1024


def fingerprints(smiles, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Boolean Morgan bit matrix, bit-identical to the core's ``make_morgan_matrix``.

    It uses the same RDKit generator but skips the core's per-column DataFrame build, which
    dominated runtime on 10k-molecule datasets. ``tests/meta/test_meta_features.py`` asserts the
    equivalence.
    """
    generator = AllChem.GetMorganGenerator(radius=int(radius), fpSize=int(n_bits))
    return np.stack([generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) > 0 for s in smiles])


def _pairwise_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - tanimoto_distance_matrix(a, b)


def nearest_neighbour_similarity(fp_query: np.ndarray, fp_train: np.ndarray) -> np.ndarray:
    """For each query molecule, the max Tanimoto similarity to any training molecule (SNN)."""
    out = np.empty(len(fp_query), dtype=float)
    for start in range(0, len(fp_query), CHUNK):
        out[start : start + CHUNK] = _pairwise_similarity(fp_query[start : start + CHUNK], fp_train).max(axis=1)
    return out


def internal_diversity(fp: np.ndarray, seed: int = 0) -> tuple[float, str]:
    """1 - mean pairwise Tanimoto over distinct pairs.

    Exact for n <= 2,000. Above that, the mean is estimated from 20,000 random distinct pairs drawn
    with a fixed seed, so the result is deterministic.
    """
    n = len(fp)
    if n < 2:
        return float("nan"), "undefined"
    if n <= EXACT_DIVERSITY_MAX_N:
        total, count = 0.0, 0
        for start in range(0, n, CHUNK):
            block = _pairwise_similarity(fp[start : start + CHUNK], fp)
            rows = np.arange(start, min(start + CHUNK, n))[:, None]
            upper = np.arange(n)[None, :] > rows
            total += float(block[upper].sum())
            count += int(upper.sum())
        return 1.0 - total / count, "exact"
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=DIVERSITY_PAIR_SAMPLE)
    j = rng.integers(0, n - 1, size=DIVERSITY_PAIR_SAMPLE)
    j = j + (j >= i)  # distinct partner, uniform over the other n - 1 molecules
    inter = np.logical_and(fp[i], fp[j]).sum(axis=1)
    union = np.logical_or(fp[i], fp[j]).sum(axis=1)
    sim = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    return 1.0 - float(sim.mean()), f"sampled_{DIVERSITY_PAIR_SAMPLE}_pairs_seed{seed}"


def scaffold_features(train_smiles) -> dict:
    keys = pd.Series([murcko_scaffold_key(s) for s in train_smiles])
    counts = keys.value_counts()
    return {
        "n_bemis_murcko_scaffolds": int(len(counts)),
        "scaffolds_per_molecule": float(len(counts) / max(len(keys), 1)),
        # Share of distinct scaffolds represented by exactly one training molecule.
        "singleton_scaffold_frac": float((counts == 1).mean()) if len(counts) else float("nan"),
    }


def label_features(train_y, task: str) -> dict:
    y = np.asarray(train_y, dtype=float)
    y = y[np.isfinite(y)]
    out = {
        "pos_prevalence": np.nan,
        "imbalance_ratio": np.nan,
        "target_range": np.nan,
        "target_std": np.nan,
        "target_skew": np.nan,
        "target_kurtosis": np.nan,
    }
    if task == "classification":
        p = float(y.mean())
        out["pos_prevalence"] = p
        out["imbalance_ratio"] = float(max(p, 1 - p) / min(p, 1 - p)) if 0 < p < 1 else float("inf")
    else:
        out["target_range"] = float(y.max() - y.min())
        out["target_std"] = float(y.std(ddof=1))
        out["target_skew"] = float(stats.skew(y))
        out["target_kurtosis"] = float(stats.kurtosis(y))  # excess (Fisher) kurtosis
    return out


def compute_meta_features(
    train_smiles,
    train_y,
    test_smiles,
    task: str | None = None,
    radius: int = DEFAULT_FP["radius"],
    n_bits: int = DEFAULT_FP["n_bits"],
    seed: int = 0,
) -> dict:
    """Meta-features for one dataset from its train/test partition (targets as the models saw them)."""
    train_smiles, test_smiles = list(train_smiles), list(test_smiles)
    task = task or infer_task(pd.Series(train_y))
    fp_train = fingerprints(train_smiles, radius, n_bits)
    fp_test = fingerprints(test_smiles, radius, n_bits)
    snn = nearest_neighbour_similarity(fp_test, fp_train)
    diversity, diversity_method = internal_diversity(fp_train, seed=seed)
    n_train, n_test = len(train_smiles), len(test_smiles)
    return {
        "n_train": n_train,
        "n_test": n_test,
        "n_total": n_train + n_test,
        "log10_n_train": float(np.log10(n_train)),
        "task": task,
        **label_features(train_y, task),
        **scaffold_features(train_smiles),
        "internal_diversity": diversity,
        "internal_diversity_method": diversity_method,
        "mean_snn": float(snn.mean()),
        "median_snn": float(np.median(snn)),
        "ood_fraction": float((snn < OOD_SNN_THRESHOLD).mean()),
    }


def compute_meta_feature_catalog(
    partitions: pd.DataFrame,
    summary: pd.DataFrame,
    radius: int = DEFAULT_FP["radius"],
    n_bits: int = DEFAULT_FP["n_bits"],
    seed: int = 0,
    progress: bool = False,
) -> pd.DataFrame:
    """One row per dataset: size/task/split metadata plus every meta-feature."""
    meta = summary.set_index("dataset")
    rows = []
    for dataset, group in partitions.groupby("dataset", sort=True):
        train = group.loc[group["split"] == "train"].sort_values("row_index")
        test = group.loc[group["split"] == "test"].sort_values("row_index")
        features = compute_meta_features(
            train["smiles"], train["observed"], test["smiles"], radius=radius, n_bits=n_bits, seed=seed
        )
        row = {
            "dataset": dataset,
            "split_strategy": meta.at[dataset, "split_strategy"] if dataset in meta.index else np.nan,
            "primary_metric": meta.at[dataset, "primary_metric"] if dataset in meta.index else np.nan,
            "target_transform": meta.at[dataset, "target_transform"] if dataset in meta.index else np.nan,
            **features,
        }
        rows.append(row)
        if progress:
            print(f"[meta-features] {dataset}: n_train={row['n_train']} mean_snn={row['mean_snn']:.3f}", flush=True)
    columns = [
        "dataset",
        "task",
        "split_strategy",
        "primary_metric",
        "target_transform",
        "n_train",
        "n_test",
        "n_total",
        "log10_n_train",
        "pos_prevalence",
        "imbalance_ratio",
        "target_range",
        "target_std",
        "target_skew",
        "target_kurtosis",
        "n_bemis_murcko_scaffolds",
        "scaffolds_per_molecule",
        "singleton_scaffold_frac",
        "internal_diversity",
        "internal_diversity_method",
        "mean_snn",
        "median_snn",
        "ood_fraction",
    ]
    return pd.DataFrame(rows)[columns]
