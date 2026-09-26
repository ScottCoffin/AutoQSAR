"""Unit tests for qsarena.interpretability: shapes, sums and graceful skips."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from qsarena.interpretability import NO_ATTRIBUTION, attribution_status, feature_importance, top_features


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = 3 * X[:, 0] + 0.5 * X[:, 2] + rng.normal(scale=0.1, size=300)
    return X, y, ["signal", "noise_a", "weak", "noise_b"]


def test_native_tree_importance_sums_to_one_and_ranks_signal_first(toy):
    X, y, names = toy
    model = RandomForestRegressor(n_estimators=50, random_state=0).fit(X, y)
    imp = feature_importance(model, names)
    assert imp.shape == (4, 3)
    assert imp["importance"].sum() == pytest.approx(1.0)
    assert imp.iloc[0]["feature"] == "signal"
    assert set(imp["method"]) == {"native"}


def test_linear_coef_importance(toy):
    X, y, names = toy
    imp = feature_importance(LinearRegression().fit(X, y), names)
    assert imp.iloc[0]["feature"] == "signal"
    assert imp["importance"].sum() == pytest.approx(1.0)


def test_permutation_fallback_for_models_without_native_importance(toy):
    X, y, names = toy
    model = KNeighborsRegressor().fit(X[:200], y[:200])
    imp = feature_importance(model, names, X=X[200:], y=y[200:], method="auto")
    assert set(imp["method"]) == {"permutation"}
    assert imp.iloc[0]["feature"] == "signal"
    assert (imp["importance"] >= 0).all()


def test_permutation_without_data_raises(toy):
    X, y, names = toy
    with pytest.raises(ValueError, match="held-out"):
        feature_importance(KNeighborsRegressor().fit(X, y), names)


def test_native_requested_but_missing_raises(toy):
    X, y, names = toy
    with pytest.raises(ValueError, match="no native"):
        feature_importance(KNeighborsRegressor().fit(X, y), names, method="native")


def test_name_count_mismatch_raises(toy):
    X, y, names = toy
    model = RandomForestRegressor(n_estimators=5, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="feature names"):
        feature_importance(model, names[:3])


@pytest.mark.parametrize(
    "name,expected",
    [
        ("CatBoost", "feature"),
        ("Random forest", "feature"),
        ("Chemprop v2 (D-MPNN + RDKit2D, ensemble=3)", NO_ATTRIBUTION),
        ("Uni-Mol V2 (84m)", NO_ATTRIBUTION),
        ("MapLight + GNN (CatBoost, Strict Parity)", NO_ATTRIBUTION),
        ("Ensemble (OOF Stacking (RidgeCV, 5-fold))", "meta-model"),
        ("CFA (Combinatorial Fusion)", "meta-model"),
    ],
)
def test_attribution_status(name, expected):
    assert attribution_status(name) == expected


def test_top_features_adds_rank(toy):
    X, y, names = toy
    imp = feature_importance(RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y), names)
    top = top_features(imp, n=2)
    assert top["rank"].tolist() == [1, 2]
