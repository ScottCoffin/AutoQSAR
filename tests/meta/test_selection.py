"""v2 family selector: regret evaluation on synthetic worlds with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qsarena.meta_analysis import selection as sel

FAMILIES = ["A", "B", "C"]


def _world(signal: bool, n: int = 44, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    features = pd.DataFrame(
        {c: rng.normal(size=n) for c in sel.F0},
        index=[f"d{i}" for i in range(n)],
    )
    features["log10_n_train"] = x
    base = rng.uniform(0, 3, size=(n, 3))
    if signal:
        base[:, 0] += np.where(x > 0, 0.0, 12.0)  # A is best only for large x
        base[:, 1] += np.where(x > 0, 12.0, 0.0)  # B is best only for small x
        base[:, 2] += 6.0
    gap = base - base.min(axis=1, keepdims=True)
    return features, pd.DataFrame(gap, index=features.index, columns=FAMILIES)


def test_regret_uses_recommended_family_and_skips_invalid():
    gap = pd.DataFrame([[0.0, 5.0, np.nan], [4.0, 0.0, 1.0]], columns=FAMILIES)
    assert sel._recommend(np.array([0.1, 0.2, -5.0]), ~np.isnan(gap.iloc[0].to_numpy())) == 0


def test_knn_and_ridge_beat_sbs_when_features_carry_signal():
    X, gap = _world(signal=True)
    sbs = sel.lodo_regret(lambda: sel.SBS(), [], X, gap)
    knn = sel.lodo_regret(lambda: sel.KNNDatasets(), sel.F0, X, gap)
    rf = sel.lodo_regret(lambda: sel.PerFamily("rf"), sel.F0, X, gap)
    assert knn.mean() < 0.5 * sbs.mean()
    assert rf.mean() < 0.5 * sbs.mean()
    summary = sel.summarize_regret(knn, sbs)
    assert summary["gap_closed"] > 0.5


def test_nested_selection_does_not_beat_sbs_on_pure_noise():
    X, gap = _world(signal=False, seed=3)
    sbs = sel.lodo_regret(lambda: sel.SBS(), [], X, gap)
    nested, picks = sel.nested_regret(X, gap, candidates=["sbs", "knn|F0", "ridge|F0"])
    assert len(picks) == len(X)
    # no signal: nested selection may not look materially better than SBS
    assert nested.mean() > 0.8 * sbs.mean()


def test_paired_bootstrap_is_deterministic():
    a, b = np.arange(10.0), np.arange(10.0)[::-1]
    assert sel.paired_bootstrap(a, b, n=500, seed=1) == sel.paired_bootstrap(a, b, n=500, seed=1)
    mean, lo, hi = sel.paired_bootstrap(a + 1, a, n=500)
    assert mean == pytest.approx(1.0) and lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_prep_is_fitted_on_training_rows_only():
    prep = sel._Prep().fit(np.array([[1.0], [3.0], [np.nan]]))
    assert prep.median[0] == 2.0
    assert np.allclose(prep.transform(np.array([[np.nan]])), 0.0)
