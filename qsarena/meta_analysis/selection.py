"""v2 family selector: algorithm selection judged by regret (docs/meta_analysis/SELECTOR_V2_PLAN.md).

A selector predicts every family's gap for a held-out dataset and recommends the argmin among the
families valid there. Its regret is the true Fig 6 gap (%) of that family. Everything is evaluated
leave-one-dataset-out; the headline chooses the variant inside an inner LODO (nested), so the grid
search itself is part of what is being evaluated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qsarena.meta_analysis.gap_matrix import FAMILY_GROUPS
from qsarena.meta_analysis.selector_features import F1, F2, F3

F0 = ["log10_n_train", "mean_snn", "label_asymmetry", "is_classification"]
BLOCKS = {
    "F0": F0,
    "F0+F1": F0 + F1,
    "F0+F2": F0 + F2,
    "F0+F3": F0 + F3,
    "F0-F3": F0 + F1 + F2 + F3,
}
RIDGE_ALPHA = 3.0
KNN_K = 5


# ---------------------------------------------------------------------------------------------
# Selectors: fit(X, G) on training datasets; predict(x) -> predicted log1p gap per family
# ---------------------------------------------------------------------------------------------
class _Prep:
    """Median imputation plus standardisation, fitted on training rows only."""

    def fit(self, X: np.ndarray):
        self.median = np.nanmedian(X, axis=0)
        filled = np.where(np.isnan(X), self.median, X)
        self.mean = filled.mean(axis=0)
        self.sd = filled.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.where(np.isnan(X), self.median, X) - self.mean) / self.sd


class SBS:
    def fit(self, X, G):
        self.pred = np.nanmean(G, axis=0)
        return self

    def predict(self, X):
        return np.tile(self.pred, (len(X), 1))


class KNNDatasets:
    """Distance-weighted mean of the k nearest training datasets' gap vectors (NaN-aware)."""

    def __init__(self, k: int = KNN_K):
        self.k = k

    def fit(self, X, G):
        self.prep = _Prep().fit(X)
        self.Z = self.prep.transform(X)
        self.G = G
        self.fallback = np.nanmean(G, axis=0)
        return self

    def predict(self, X):
        out = []
        for z in self.prep.transform(X):
            dist = np.sqrt(((self.Z - z) ** 2).sum(axis=1))
            nn = np.argsort(dist)[: self.k]
            w = 1.0 / (dist[nn] + 1e-6)
            g = self.G[nn]
            weights = np.where(np.isnan(g), 0.0, w[:, None])
            pred = np.where(
                weights.sum(axis=0) > 0,
                np.nansum(np.nan_to_num(g) * weights, axis=0) / np.maximum(weights.sum(axis=0), 1e-12),
                self.fallback,
            )
            out.append(pred)
        return np.array(out)


class PerFamily:
    """One regressor per family, trained on the datasets where that family is valid."""

    def __init__(self, kind: str, seed: int = 0):
        self.kind = kind
        self.seed = seed

    def _model(self):
        if self.kind == "ridge":
            from sklearn.linear_model import Ridge

            return Ridge(alpha=RIDGE_ALPHA)
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=100, max_depth=3, min_samples_leaf=3, random_state=self.seed, n_jobs=1
        )

    def fit(self, X, G):
        self.prep = _Prep().fit(X)
        Z = self.prep.transform(X)
        self.models = []
        for j in range(G.shape[1]):
            ok = ~np.isnan(G[:, j])
            self.models.append(self._model().fit(Z[ok], G[ok, j]) if ok.sum() >= 5 else float(np.nanmean(G[:, j])))
        return self

    def predict(self, X):
        Z = self.prep.transform(X)
        cols = [m.predict(Z) if hasattr(m, "predict") else np.full(len(Z), m) for m in self.models]
        return np.column_stack(cols)


class WinnerTree:
    """v1: depth-3 tree on the grouped winner; recommends the group's best family by training mean gap."""

    def __init__(self, families, seed: int = 0):
        self.families = list(families)
        self.seed = seed

    def fit(self, X, G):
        from sklearn.tree import DecisionTreeClassifier

        self.prep = _Prep().fit(X)
        groups = np.array([FAMILY_GROUPS.get(self.families[j], "Descriptor-based ML") for j in np.nanargmin(G, 1)])
        self.tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=self.seed)
        self.tree.fit(self.prep.transform(X), groups)
        self.mean_gap = np.nanmean(G, axis=0)
        return self

    def predict(self, X):
        out = []
        for group in self.tree.predict(self.prep.transform(X)):
            pred = np.full(len(self.families), 10.0)  # families outside the predicted group rank last
            for j, family in enumerate(self.families):
                if FAMILY_GROUPS.get(family, "Descriptor-based ML") == group:
                    pred[j] = self.mean_gap[j] if np.isfinite(self.mean_gap[j]) else 9.0
            out.append(pred)
        return np.array(out)


def variant_grid(families) -> dict:
    grid = {"sbs": (lambda: SBS(), [])}
    for block, cols in BLOCKS.items():
        grid[f"knn|{block}"] = (lambda: KNNDatasets(), cols)
        grid[f"ridge|{block}"] = (lambda: PerFamily("ridge"), cols)
        grid[f"rf|{block}"] = (lambda: PerFamily("rf"), cols)
    grid["tree_cls|F0"] = (lambda fam=tuple(families): WinnerTree(fam), F0)
    return grid


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------
def _recommend(pred: np.ndarray, valid: np.ndarray) -> int:
    return int(np.argmin(np.where(valid, pred, np.inf)))


def lodo_regret(make, cols, X: pd.DataFrame, gap: pd.DataFrame, rows=None) -> np.ndarray:
    """Regret (% gap of the recommended family) for each held-out dataset in ``rows``."""
    G = np.log1p(gap.to_numpy(dtype=float))
    raw = gap.to_numpy(dtype=float)
    Xa = X[cols].to_numpy(dtype=float) if cols else np.zeros((len(X), 1))
    rows = range(len(X)) if rows is None else rows
    out = []
    for i in rows:
        train = np.ones(len(X), dtype=bool)
        train[i] = False
        model = make().fit(Xa[train], G[train])
        j = _recommend(model.predict(Xa[i : i + 1])[0], ~np.isnan(raw[i]))
        out.append(raw[i, j])
    return np.array(out)


def evaluate_grid(X: pd.DataFrame, gap: pd.DataFrame) -> pd.DataFrame:
    """Per-variant LODO regret per dataset (datasets x variants)."""
    grid = variant_grid(gap.columns)
    return pd.DataFrame({name: lodo_regret(make, cols, X, gap) for name, (make, cols) in grid.items()}, index=X.index)


def nested_regret(X: pd.DataFrame, gap: pd.DataFrame, candidates=None) -> tuple[np.ndarray, list[str]]:
    """Outer LODO; for each outer dataset an inner LODO over the rest picks the variant (mean regret)."""
    grid = variant_grid(gap.columns)
    names = [n for n in grid if candidates is None or n in candidates]
    regrets, picks = [], []
    idx = np.arange(len(X))
    for i in idx:
        inner = idx[idx != i]
        Xi, Gi = X.iloc[inner], gap.iloc[inner]
        inner_scores = {n: lodo_regret(grid[n][0], grid[n][1], Xi, Gi).mean() for n in names}
        best = min(names, key=lambda n: (inner_scores[n], n))
        picks.append(best)
        regrets.append(lodo_regret(grid[best][0], grid[best][1], X, gap, rows=[i])[0])
    return np.array(regrets), picks


def summarize_regret(regret: np.ndarray, sbs: np.ndarray, oracle_mean: float = 0.0) -> dict:
    mean, sbs_mean = float(np.mean(regret)), float(np.mean(sbs))
    return {
        "mean_regret_pct": mean,
        "median_regret_pct": float(np.median(regret)),
        "within5_pct": float(np.mean(regret <= 5.0 + 1e-9) * 100),
        "gap_closed": float((sbs_mean - mean) / (sbs_mean - oracle_mean)) if sbs_mean > oracle_mean else np.nan,
    }


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) with a dataset-bootstrap 95% CI."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    reps = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi)


def permutation_p_nested(
    X: pd.DataFrame, gap: pd.DataFrame, observed_mean: float, n_perm: int = 200, seed: int = 0, candidates=None
) -> dict:
    """Shuffle which dataset's gap vector goes with which feature row; rerun the whole nested pipeline.

    One-sided p = share of permutations with mean nested regret <= observed (lower is better).
    """
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        order = rng.permutation(len(gap))
        permuted = pd.DataFrame(gap.to_numpy()[order], index=gap.index, columns=gap.columns)
        null.append(nested_regret(X, permuted, candidates)[0].mean())
    null = np.array(null)
    return {
        "null_mean_regret_pct": float(null.mean()),
        "p_value": float((np.sum(null <= observed_mean + 1e-12) + 1) / (n_perm + 1)),
        "n_perm": n_perm,
    }
