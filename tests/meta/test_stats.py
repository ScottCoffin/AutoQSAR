"""Phase 3: statistical helpers on synthetic data with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qsarena.meta_analysis import stats as st


def test_bootstrap_is_deterministic_and_covers_true_slope():
    rng = np.random.default_rng(1)
    covered, excludes_zero = 0, 0
    sims = 60
    for _ in range(sims):
        x = rng.uniform(0, 1, 44)
        data = np.column_stack([x, 2 * x + rng.normal(0, 0.5, 44)])
        ci = st.bootstrap_ci(lambda a: np.polyfit(a[:, 0], a[:, 1], 1)[0], data, n=400, seed=0)
        covered += ci["ci_low"] <= 2 <= ci["ci_high"]
        excludes_zero += ci["ci_low"] > 0
    assert covered / sims >= 0.85
    assert excludes_zero == sims
    data = np.column_stack([np.arange(10.0), np.arange(10.0) ** 1.5])
    a = st.bootstrap_ci(lambda d: np.polyfit(d[:, 0], d[:, 1], 1)[0], data, n=200, seed=3)
    b = st.bootstrap_ci(lambda d: np.polyfit(d[:, 0], d[:, 1], 1)[0], data, n=200, seed=3)
    assert a == b


def test_vectorized_spearman_ci_matches_generic_bootstrap():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=30), rng.normal(size=30)
    y[:5] = 0.0  # ties
    fast = st.spearman_with_ci(x, y, n=500, seed=4)
    slow = st.bootstrap_ci(lambda a: st._spearman(a[:, 0], a[:, 1]), np.column_stack([x, y]), n=500, seed=4)
    assert fast["estimate"] == pytest.approx(slow["estimate"])
    assert fast["ci_low"] == pytest.approx(slow["ci_low"], abs=1e-9)
    assert fast["ci_high"] == pytest.approx(slow["ci_high"], abs=1e-9)


def test_permutation_p_decreases_with_signal_and_is_calm_under_noise():
    rng = np.random.default_rng(3)
    draws = [(rng.normal(size=44), rng.normal(size=44)) for _ in range(20)]
    mean_p = [
        np.mean([st.permutation_test(x, beta * x + noise, n=1000, seed=0) for x, noise in draws])
        for beta in (0.0, 0.25, 0.6, 1.5)
    ]
    assert mean_p == sorted(mean_p, reverse=True)
    assert mean_p[-1] < 0.002
    null_ps = [st.permutation_test(rng.normal(size=44), rng.normal(size=44), n=500, seed=k) for k in range(100)]
    assert np.mean(np.array(null_ps) > 0.05) >= 0.90


def test_bh_qvalues_known_example():
    p = np.array([0.01, 0.04, 0.03, 0.20, np.nan])
    q = st.bh_qvalues(p)
    # raw p*m/rank for sorted p (0.01, 0.03, 0.04, 0.20) = 0.04, 0.06, 0.0533, 0.20; then running min from the top
    assert np.allclose(q[:4], [0.04, 0.16 / 3, 0.16 / 3, 0.20])
    assert np.isnan(q[4])


def test_crossover_recovers_known_size():
    rng = np.random.default_rng(4)
    n = np.round(10 ** rng.uniform(2.4, 4.1, 44))
    logn = np.log10(n)
    a = 10 - 6 * (logn - 3) + rng.normal(0, 0.5, 44)
    b = 10 + 6 * (logn - 3) + rng.normal(0, 0.5, 44)
    result = st.crossover_estimate(n, a, b, n=1000, seed=0)
    assert result["crossover_n"] == pytest.approx(1000, rel=0.1)
    assert result["ci_low_n"] <= 1000 <= result["ci_high_n"]
    assert result["p_cross_in_range"] > 0.95


def test_crossover_reports_none_for_parallel_curves():
    n = np.logspace(2.5, 4, 40)
    result = st.crossover_estimate(n, np.log10(n), np.log10(n) + 5, n=200, seed=0)
    assert np.isnan(result["crossover_n"]) and result["p_cross_in_range"] < 0.05


def _features(n_train):
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {"log10_n_train": np.log10(n_train), "noise": rng.normal(size=len(n_train))},
        index=[f"d{i}" for i in range(len(n_train))],
    )


def test_recommender_recovers_size_rule_and_stays_at_baseline_on_random_labels():
    n_train = np.logspace(2.4, 4.1, 44)
    features = _features(n_train)
    rule = pd.Series(np.where(features["log10_n_train"] < 3.2, "small", "large"), index=features.index)
    good = st.lodo_recommender(features, rule, n_perm=50, n_boot=500)
    assert good["models"]["decision_tree"]["lodo_accuracy"] >= 0.95
    random_labels = pd.Series(np.random.default_rng(6).choice(["a", "b", "c"], 44), index=features.index)
    rand = st.lodo_recommender(features, random_labels, n_perm=100, n_boot=500)
    for model in rand["models"].values():
        assert model["lodo_balanced_accuracy"] < 0.55
        assert model["permutation_p"] > 0.05


def test_recommender_refuses_sparse_classes():
    features = _features(np.logspace(2.4, 4.1, 20))
    labels = pd.Series(["a"] * 17 + ["b"] * 3, index=features.index)
    with pytest.raises(ValueError, match=">= 4 datasets"):
        st.lodo_recommender(features, labels, n_perm=5, n_boot=50)


def test_recommender_limits_predictors():
    features = pd.DataFrame(np.zeros((20, 5)), index=[f"d{i}" for i in range(20)])
    with pytest.raises(ValueError, match="at most four"):
        st.lodo_recommender(features, pd.Series(["a", "b"] * 10, index=features.index))
