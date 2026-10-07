"""Dataset-level inference for the meta-analysis.

Every interval is a percentile bootstrap over datasets (the unit of replication). None of them
estimates seed variance: the benchmark is single-seed. All results are exploratory.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

N_BOOT = 10_000
N_PERM = 10_000


def _finite_pairs(x, y) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def bootstrap_ci(func, data, n: int = N_BOOT, seed: int = 0, alpha: float = 0.05) -> dict:
    """Resample rows of ``data`` (one row per dataset) with replacement; percentile CI of ``func``.

    ``data`` is a DataFrame or array whose first axis is datasets. Replicates where ``func`` is not
    finite (e.g. a constant resample) are dropped and counted in ``n_valid``.
    """
    rng = np.random.default_rng(seed)
    is_frame = isinstance(data, pd.DataFrame)
    values = data if is_frame else np.asarray(data)
    m = len(values)
    point = float(func(values))
    reps = np.empty(n, dtype=float)
    for b in range(n):
        idx = rng.integers(0, m, size=m)
        sample = values.iloc[idx] if is_frame else values[idx]
        try:
            reps[b] = float(func(sample))
        except (ValueError, FloatingPointError, ZeroDivisionError):
            reps[b] = np.nan
    finite = reps[np.isfinite(reps)]
    lo, hi = np.quantile(finite, [alpha / 2, 1 - alpha / 2]) if len(finite) else (np.nan, np.nan)
    return {"estimate": point, "ci_low": float(lo), "ci_high": float(hi), "n_valid": int(len(finite)), "n": m}


def _spearman(x, y) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.spearmanr(x, y).statistic)


def _rowwise_spearman(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Spearman rho for each row of two (B, m) matrices (average ranks for ties)."""
    rx = stats.rankdata(xs, axis=1)
    ry = stats.rankdata(ys, axis=1)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    denom = np.sqrt((rx**2).sum(axis=1) * (ry**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (rx * ry).sum(axis=1) / denom, np.nan)


def spearman_with_ci(x, y, n: int = N_BOOT, seed: int = 0) -> dict:
    """Spearman rho with a dataset-bootstrap 95% percentile CI (pairs with a missing value dropped).

    Equivalent to ``bootstrap_ci`` over rows of (x, y), vectorized across replicates.
    """
    x, y = _finite_pairs(x, y)
    m = len(x)
    point = _spearman(x, y)
    if m < 3:
        return {"estimate": point, "ci_low": np.nan, "ci_high": np.nan, "n_valid": 0, "n": m}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n, m))
    reps = _rowwise_spearman(x[idx], y[idx])
    finite = reps[np.isfinite(reps)]
    lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
    return {"estimate": point, "ci_low": float(lo), "ci_high": float(hi), "n_valid": int(len(finite)), "n": m}


def permutation_test(x, y, n: int = N_PERM, seed: int = 0) -> float:
    """Two-sided label-shuffle p-value for Spearman rho; (hits + 1) / (n + 1)."""
    x, y = _finite_pairs(x, y)
    observed = _spearman(x, y)
    if not np.isfinite(observed):
        return float("nan")
    rng = np.random.default_rng(seed)
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    perms = np.array([rng.permutation(ry) for _ in range(n)])
    null = perms @ rx / len(rx)
    hits = int(np.sum(np.abs(null) >= abs(observed) - 1e-12))
    return (hits + 1) / (n + 1)


def bh_qvalues(p_values) -> np.ndarray:
    """Benjamini-Hochberg q-values; NaN p-values stay NaN and do not count toward m."""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def robust_line(x, y) -> tuple[float, float]:
    """Theil-Sen slope and intercept (robust to the heavy-tailed gaps of a few datasets)."""
    x, y = _finite_pairs(x, y)
    if len(x) < 3 or np.ptp(x) == 0:
        return np.nan, np.nan
    slope, intercept, _, _ = stats.theilslopes(y, x)
    return float(slope), float(intercept)


def _crossing(x_a, y_a, x_b, y_b, lo: float, hi: float) -> float:
    sa, ia = robust_line(x_a, y_a)
    sb, ib = robust_line(x_b, y_b)
    if not all(np.isfinite([sa, ia, sb, ib])) or abs(sa - sb) < 1e-12:
        return np.nan
    x_cross = (ib - ia) / (sa - sb)
    return float(x_cross) if lo <= x_cross <= hi else np.nan


def crossover_estimate(size, gap_family_a, gap_family_b, n: int = N_BOOT, seed: int = 0) -> dict:
    """Training-set size where robust fits of gap-vs-log10(size) for families A and B cross.

    ``size`` is n_train (not logged). Each family is fitted on its own finite cells (Theil-Sen);
    the crossing is reported only inside the observed log10(size) range. The bootstrap resamples
    datasets and records how often a crossing exists (``p_cross_in_range``) and the percentile CI
    of the crossing size among replicates where it does.
    """
    logn = np.log10(np.asarray(size, dtype=float))
    a = np.asarray(gap_family_a, dtype=float)
    b = np.asarray(gap_family_b, dtype=float)
    lo, hi = float(np.nanmin(logn)), float(np.nanmax(logn))
    ok_a, ok_b = np.isfinite(a) & np.isfinite(logn), np.isfinite(b) & np.isfinite(logn)
    point = _crossing(logn[ok_a], a[ok_a], logn[ok_b], b[ok_b], lo, hi)
    sa, ia = robust_line(logn[ok_a], a[ok_a])
    sb, ib = robust_line(logn[ok_b], b[ok_b])
    rng = np.random.default_rng(seed)
    m = len(logn)
    reps = np.empty(n, dtype=float)
    for k in range(n):
        idx = rng.integers(0, m, size=m)
        la, aa, bb = logn[idx], a[idx], b[idx]
        fa, fb = np.isfinite(aa), np.isfinite(bb)
        reps[k] = _crossing(la[fa], aa[fa], la[fb], bb[fb], lo, hi)
    finite = reps[np.isfinite(reps)]
    ci = np.quantile(finite, [0.025, 0.975]) if len(finite) >= 20 else (np.nan, np.nan)
    return {
        "crossover_log10_n": point,
        "crossover_n": float(10**point) if np.isfinite(point) else np.nan,
        "ci_low_n": float(10 ** ci[0]) if np.isfinite(ci[0]) else np.nan,
        "ci_high_n": float(10 ** ci[1]) if np.isfinite(ci[1]) else np.nan,
        "p_cross_in_range": float(len(finite) / n),
        "slope_a": sa,
        "intercept_a": ia,
        "slope_b": sb,
        "intercept_b": ib,
        "log10_n_range": (lo, hi),
    }


def trend_band(x, y, grid, n: int = 2000, seed: int = 0) -> dict:
    """Theil-Sen fit on ``grid`` with a dataset-bootstrap 95% band (for Fig M1a)."""
    x, y = _finite_pairs(x, y)
    slope, intercept = robust_line(x, y)
    rng = np.random.default_rng(seed)
    grid = np.asarray(grid, dtype=float)
    curves = []
    for _ in range(n):
        idx = rng.integers(0, len(x), size=len(x))
        s, i = robust_line(x[idx], y[idx])
        if np.isfinite(s):
            curves.append(i + s * grid)
    curves = np.asarray(curves)
    lo, hi = np.quantile(curves, [0.025, 0.975], axis=0) if len(curves) else (grid * np.nan, grid * np.nan)
    return {"fit": intercept + slope * grid, "low": lo, "high": hi, "slope": slope, "intercept": intercept}


# ---------------------------------------------------------------------------------------------
# Leave-one-dataset-out family recommender
# ---------------------------------------------------------------------------------------------
MIN_CLASS_SIZE = 4


def _make_models(seed: int = 0):
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    major, minor = (int(part) for part in sklearn.__version__.split(".")[:2])
    # scikit-learn 1.8 deprecated `penalty`; pure L1 is l1_ratio=1 there, penalty="l1" before.
    l1 = {"l1_ratio": 1.0} if (major, minor) >= (1, 8) else {"penalty": "l1"}
    return {
        "decision_tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=seed),
        "l1_logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(solver="saga", C=1.0, max_iter=20_000, random_state=seed, **l1),
        ),
    }


def _lodo_predictions(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.base import clone

    pred = np.empty(len(y), dtype=object)
    for i in range(len(y)):
        train = np.arange(len(y)) != i
        fitted = clone(model).fit(X[train], y[train])
        pred[i] = fitted.predict(X[i : i + 1])[0]
    return pred


def _majority_constant(y: np.ndarray) -> np.ndarray:
    """Predict the overall most frequent class for every dataset (the conventional majority baseline).

    Not refitted per left-out dataset: with near-tied classes, a leave-one-out majority flips
    whenever a member of the leading class is held out and scores ~0, which is a misleading floor.
    Its balanced accuracy is exactly 1 / n_classes.
    """
    counts = pd.Series(y).value_counts()
    top = sorted(counts[counts == counts.max()].index)[0]  # deterministic tie-break
    return np.array([top] * len(y), dtype=object)


def _balanced_accuracy(y_true, y_pred) -> float:
    """Mean per-class recall over classes present in ``y_true`` (sklearn's balanced_accuracy_score)."""
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    recalls = [np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)]
    return float(np.mean(recalls))


def _bootstrap_balanced_accuracy(y_true, y_pred, n: int, seed: int) -> tuple[float, float]:
    """Percentile 95% CI of balanced accuracy over dataset resamples, vectorized across replicates."""
    y_true = np.asarray(y_true, dtype=object)
    correct = (np.asarray(y_pred, dtype=object) == y_true).astype(float)
    classes = np.unique(y_true)
    onehot = np.stack([(y_true == c).astype(float) for c in classes], axis=1)  # (m, k)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n, len(y_true)))
    counts = onehot[idx].sum(axis=1)  # (n, k)
    hits = (onehot[idx] * correct[idx][:, :, None]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(counts > 0, hits / counts, np.nan)
    reps = np.nanmean(recall, axis=1)
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return float(lo), float(hi)


def lodo_recommender(
    meta_features: pd.DataFrame,
    target: pd.Series,
    n_perm: int = 500,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> dict:
    """Leave-one-dataset-out evaluation of a shallow tree and an L1 multinomial logistic model.

    ``meta_features``: one row per dataset, at most four numeric predictor columns.
    ``target``: grouped best-family label per dataset (same index). Refuses to run when any class
    has fewer than four datasets, which forces a grouped target.
    """
    if meta_features.shape[1] > 4:
        raise ValueError("the recommender is limited to at most four predictors")
    target = target.loc[meta_features.index]
    counts = target.value_counts()
    if (counts < MIN_CLASS_SIZE).any():
        raise ValueError(
            f"every class needs >= {MIN_CLASS_SIZE} datasets (got {counts.to_dict()}); use the grouped target"
        )
    X = meta_features.to_numpy(dtype=float)
    y = target.astype(str).to_numpy(dtype=object)
    rng = np.random.default_rng(seed)
    majority = _majority_constant(y)
    out = {
        "n_datasets": int(len(y)),
        "classes": {str(k): int(v) for k, v in counts.sort_index().items()},
        "predictors": list(meta_features.columns),
        "majority_accuracy": float(np.mean(majority == y)),
        "majority_balanced_accuracy": _balanced_accuracy(y, majority),
        "models": {},
    }
    for name, model in _make_models(seed).items():
        pred = _lodo_predictions(model, X, y)
        correct = (pred == y).astype(float)
        bacc = _balanced_accuracy(y, pred)
        bacc_lo, bacc_hi = _bootstrap_balanced_accuracy(y, pred, n=n_boot, seed=seed)
        acc_ci = bootstrap_ci(np.mean, correct, n=n_boot, seed=seed)
        null = []
        for _ in range(n_perm):
            y_perm = rng.permutation(y)
            null.append(_balanced_accuracy(y_perm, _lodo_predictions(model, X, y_perm)))
        null = np.asarray(null)
        fitted = model.fit(X, y)
        record = {
            "lodo_accuracy": acc_ci["estimate"],
            "lodo_accuracy_ci": (acc_ci["ci_low"], acc_ci["ci_high"]),
            "lodo_balanced_accuracy": bacc,
            "lodo_balanced_accuracy_ci": (bacc_lo, bacc_hi),
            "permutation_null_balanced_accuracy_mean": float(null.mean()),
            "permutation_p": float((np.sum(null >= bacc - 1e-12) + 1) / (n_perm + 1)),
            "predictions": dict(zip(meta_features.index, pred)),
        }
        if name == "decision_tree":
            from sklearn.tree import export_text

            record["rules"] = export_text(fitted, feature_names=list(meta_features.columns))
            record["feature_importance"] = dict(zip(meta_features.columns, map(float, fitted.feature_importances_)))
        else:
            logit = fitted[-1]
            record["coefficients"] = {
                str(cls): dict(zip(meta_features.columns, map(float, coef)))
                for cls, coef in zip(logit.classes_, logit.coef_)
            }
            record["feature_importance"] = dict(
                zip(meta_features.columns, map(float, np.abs(logit.coef_).mean(axis=0)))
            )
        out["models"][name] = record
    return out
