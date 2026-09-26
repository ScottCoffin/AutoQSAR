"""
qsarena.applicability_domain — structural applicability-domain (AD) methods.

OECD (Q)SAR validation principle 3 asks for "a defined domain of applicability". This module
implements the two structural AD approaches used by the reliability study
(``qsarena.reliability_study``); the prediction-reliability half of the AD story (conformal
intervals and classification confidence) lives in ``qsarena.uncertainty``.

  standardization_ad  Roy, Kar & Ambure (2015), "On a simple approach for determining
                      applicability domain of QSAR models", Chemometr. Intell. Lab. Syst.
                      145:22-29. Descriptor-range rule on training-standardized |z| scores.
                      The same method MetaQSAR ships as its "standardization approach".

  knn_similarity_ad   Mean Tanimoto distance from a query to its k nearest training
                      fingerprints, thresholded at a quantile of the training set's own
                      leave-one-out kNN distances. Leakage-free: the threshold is computed from
                      training compounds only.

Both functions are pure numpy and have no dependency on the benchmark runner, so they are unit
tested directly (``tests/unit/test_applicability_domain.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "StandardizationADResult",
    "KNNADResult",
    "standardization_ad",
    "knn_similarity_ad",
    "tanimoto_distance_matrix",
]

#: Roy et al. (2015) use 3 standard deviations: a normal variate exceeds it with p < 0.003.
ROY_THRESHOLD = 3.0
#: z-value of the 90th percentile, used in Roy's S_new = mean(S) + 1.28 * sd(S).
_Z90 = 1.28


@dataclass(frozen=True)
class StandardizationADResult:
    """Per-query output of :func:`standardization_ad`."""

    s_max: np.ndarray
    s_min: np.ndarray
    s_new: np.ndarray
    in_domain: np.ndarray


@dataclass(frozen=True)
class KNNADResult:
    """Per-query output of :func:`knn_similarity_ad`."""

    mean_knn_distance: np.ndarray
    threshold: float
    in_domain: np.ndarray


def standardization_ad(
    X_train: np.ndarray,
    X_query: np.ndarray,
    threshold: float = ROY_THRESHOLD,
) -> StandardizationADResult:
    """Roy-Kar-Ambure standardization AD.

    For each descriptor i, a query compound k gets ``S_ki = |x_ki - mean_i| / sd_i`` using the
    TRAINING mean and standard deviation. Then, per compound:

      * ``max_i S_ki <= threshold``  -> in domain;
      * ``min_i S_ki >  threshold``  -> out of domain;
      * otherwise ``S_new = mean_i(S_ki) + 1.28 * sd_i(S_ki)``; in domain iff ``S_new <= threshold``.

    Descriptors with zero training variance carry no range information and are dropped, as are
    non-finite values (treated as missing, not as extreme).
    """
    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    if X_train.ndim != 2 or X_query.ndim != 2:
        raise ValueError("X_train and X_query must be 2-D arrays")
    if X_train.shape[1] != X_query.shape[1]:
        raise ValueError(
            f"descriptor count mismatch: train has {X_train.shape[1]}, query has {X_query.shape[1]}"
        )
    mean = np.nanmean(X_train, axis=0)
    sd = np.nanstd(X_train, axis=0, ddof=1) if X_train.shape[0] > 1 else np.zeros(X_train.shape[1])
    keep = np.isfinite(mean) & np.isfinite(sd) & (sd > 0)
    if not keep.any():
        raise ValueError("no descriptor has non-zero training variance")
    S = np.abs(X_query[:, keep] - mean[keep]) / sd[keep]
    S = np.where(np.isfinite(S), S, np.nan)

    s_max = np.nanmax(S, axis=1)
    s_min = np.nanmin(S, axis=1)
    s_mean = np.nanmean(S, axis=1)
    n_valid = np.sum(np.isfinite(S), axis=1)
    s_sd = np.zeros(S.shape[0])
    multi = n_valid > 1
    if multi.any():
        s_sd[multi] = np.nanstd(S[multi], axis=1, ddof=1)
    s_new = s_mean + _Z90 * s_sd

    in_domain = np.where(
        s_max <= threshold,
        True,
        np.where(s_min > threshold, False, s_new <= threshold),
    ).astype(bool)
    return StandardizationADResult(s_max=s_max, s_min=s_min, s_new=s_new, in_domain=in_domain)


def tanimoto_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto (Jaccard) distance between binary fingerprint matrices A (n,d), B (m,d)."""
    A = (np.asarray(A) > 0).astype(np.float32)
    B = (np.asarray(B) > 0).astype(np.float32)
    inter = A @ B.T
    a = A.sum(axis=1)[:, None]
    b = B.sum(axis=1)[None, :]
    union = a + b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / union, 1.0)  # two empty fingerprints are identical
    return 1.0 - sim


def _mean_knn(dist: np.ndarray, k: int) -> np.ndarray:
    k = min(k, dist.shape[1])
    part = np.partition(dist, k - 1, axis=1)[:, :k]
    return part.mean(axis=1)


def knn_similarity_ad(
    fp_train: np.ndarray,
    fp_query: np.ndarray,
    k: int = 5,
    quantile: float = 0.95,
    chunk_size: int = 2048,
) -> KNNADResult:
    """kNN Tanimoto-distance AD with a training-derived threshold.

    The threshold is the ``quantile`` of each training compound's mean distance to its k nearest
    OTHER training compounds (leave-one-out). A query is in domain iff its mean distance to its
    k nearest training compounds does not exceed that threshold. Distances are computed in
    chunks so a 10k x 10k comparison does not allocate one dense matrix.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    if k < 1:
        raise ValueError("k must be >= 1")
    fp_train = np.asarray(fp_train)
    fp_query = np.asarray(fp_query)
    n_train = fp_train.shape[0]
    if n_train < 2:
        raise ValueError("need at least two training compounds")
    k_eff = min(k, n_train - 1)

    train_loo = np.empty(n_train, dtype=float)
    for start in range(0, n_train, chunk_size):
        stop = min(start + chunk_size, n_train)
        d = tanimoto_distance_matrix(fp_train[start:stop], fp_train)
        d[np.arange(stop - start), np.arange(start, stop)] = np.inf  # exclude self
        train_loo[start:stop] = _mean_knn(d, k_eff)
    threshold = float(np.quantile(train_loo, quantile))

    query = np.empty(fp_query.shape[0], dtype=float)
    for start in range(0, fp_query.shape[0], chunk_size):
        stop = min(start + chunk_size, fp_query.shape[0])
        query[start:stop] = _mean_knn(tanimoto_distance_matrix(fp_query[start:stop], fp_train), k_eff)
    return KNNADResult(mean_knn_distance=query, threshold=threshold, in_domain=query <= threshold)
