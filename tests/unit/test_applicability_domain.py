"""Unit tests for qsarena.applicability_domain: known in/out points get the right flags."""

from __future__ import annotations

import numpy as np
import pytest

from qsarena.applicability_domain import (
    knn_similarity_ad,
    standardization_ad,
    tanimoto_distance_matrix,
)


@pytest.fixture
def gaussian_train():
    rng = np.random.default_rng(0)
    return rng.normal(0.0, 1.0, size=(500, 6))


def test_standardization_center_is_in_domain(gaussian_train):
    res = standardization_ad(gaussian_train, np.zeros((1, 6)))
    assert res.in_domain.tolist() == [True]
    assert res.s_max[0] < 0.5


def test_standardization_all_descriptors_extreme_is_out(gaussian_train):
    # Every descriptor 10 SD away: S_min > 3 -> out of domain by the first rule.
    res = standardization_ad(gaussian_train, np.full((1, 6), 10.0))
    assert res.in_domain.tolist() == [False]
    assert res.s_min[0] > 3


def test_standardization_single_mild_excursion_rescued_by_s_new(gaussian_train):
    # One descriptor at ~3.5 SD, the rest at the mean: S_max > 3 but S_new = mean + 1.28 sd < 3.
    q = np.zeros((1, 6))
    q[0, 0] = 3.5 * gaussian_train[:, 0].std(ddof=1) + gaussian_train[:, 0].mean()
    res = standardization_ad(gaussian_train, q)
    assert res.s_max[0] > 3
    assert res.s_new[0] <= 3
    assert res.in_domain.tolist() == [True]


def test_standardization_s_new_formula(gaussian_train):
    q = np.array([[4.0, 3.5, 0.0, 0.0, 5.0, 0.2]])
    res = standardization_ad(gaussian_train, q)
    mean, sd = gaussian_train.mean(0), gaussian_train.std(0, ddof=1)
    S = np.abs(q - mean) / sd
    assert res.s_new[0] == pytest.approx(S.mean() + 1.28 * S.std(ddof=1))
    assert res.in_domain[0] == (S.max() <= 3 or (S.min() <= 3 and S.mean() + 1.28 * S.std(ddof=1) <= 3))


def test_standardization_drops_constant_descriptors():
    train = np.column_stack([np.arange(10.0), np.ones(10)])
    res = standardization_ad(train, np.array([[4.5, 999.0]]))  # constant column ignored
    assert res.in_domain.tolist() == [True]


def test_standardization_shape_mismatch_raises(gaussian_train):
    with pytest.raises(ValueError, match="mismatch"):
        standardization_ad(gaussian_train, np.zeros((1, 5)))


def test_tanimoto_distance_known_values():
    a = np.array([[1, 1, 0, 0]])
    b = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]])
    d = tanimoto_distance_matrix(a, b)[0]
    assert d == pytest.approx([0.0, 1 - 1 / 3, 1.0])


def test_knn_flags_duplicate_in_and_disjoint_out():
    rng = np.random.default_rng(1)
    # Training fingerprints use bits 0..63 only; the "alien" query uses bits 64..127 only.
    train = np.zeros((80, 128), dtype=np.uint8)
    train[:, :64] = rng.random((80, 64)) < 0.3
    alien = np.zeros((1, 128), dtype=np.uint8)
    alien[0, 64:] = 1
    query = np.vstack([train[:1], alien])
    res = knn_similarity_ad(train, query, k=3, quantile=0.95)
    assert res.in_domain.tolist() == [True, False]
    assert res.mean_knn_distance[1] == pytest.approx(1.0)


def test_knn_threshold_uses_training_only_and_chunking_is_exact():
    rng = np.random.default_rng(2)
    train = (rng.random((50, 64)) < 0.4).astype(np.uint8)
    query = (rng.random((30, 64)) < 0.4).astype(np.uint8)
    a = knn_similarity_ad(train, query, k=5, chunk_size=7)
    b = knn_similarity_ad(train, query, k=5, chunk_size=4096)
    assert a.threshold == pytest.approx(b.threshold)
    np.testing.assert_allclose(a.mean_knn_distance, b.mean_knn_distance)
    # Threshold does not move when the query set changes.
    c = knn_similarity_ad(train, query[:3], k=5)
    assert c.threshold == pytest.approx(a.threshold)
