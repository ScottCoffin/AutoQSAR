"""
qsarena.uncertainty — conformal prediction and calibration metrics.

Supports OECD (Q)SAR principle 4 (goodness-of-fit, robustness, predictivity) and the "prediction
reliability" check of the OECD (Q)SAR Assessment Framework. Everything here is split
(inductive) conformal prediction: a calibration set carved from the TRAINING partition is used to
set the interval width, so the held-out test set is never consulted.

Regression
  conformal_quantile          finite-sample-corrected (1 - alpha) quantile of scores
  SplitConformalRegressor     absolute-residual or difficulty-normalised intervals

Classification
  conformal_prediction_sets   least-ambiguous set-valued classifier (LAC) score 1 - p(y)
  probability_confidence      Mathea et al. (2016)-style |p1 - p0| confidence flag

Calibration metrics
  expected_calibration_error, brier_score, reliability_curve

The finite-sample guarantee is marginal coverage >= 1 - alpha when calibration and test points
are exchangeable. A scaffold split breaks exchangeability, so empirical test coverage on the TDC
scaffold splits is a measured quantity, not a guaranteed one; the reliability study reports it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "conformal_quantile",
    "SplitConformalRegressor",
    "conformal_prediction_sets",
    "probability_confidence",
    "expected_calibration_error",
    "brier_score",
    "reliability_curve",
]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the ceil((n + 1)(1 - alpha)) / n empirical quantile of ``scores``.

    This is the standard split-conformal threshold. When (n + 1)(1 - alpha) > n the guarantee
    requires an infinite threshold, which is returned as ``np.inf``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = scores.size
    if n == 0:
        raise ValueError("no finite calibration scores")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf")
    return float(np.sort(scores)[rank - 1])


@dataclass
class SplitConformalRegressor:
    """Split-conformal intervals around an already-fitted point predictor.

    ``fit`` takes calibration predictions and targets (and, optionally, a per-sample difficulty
    estimate such as the across-tree standard deviation of a random forest). With a difficulty
    estimate the score is ``|y - yhat| / (sigma + eps)`` and intervals scale with sigma, so they
    widen for compounds the model is unsure about; without one they have constant width.
    """

    alpha: float = 0.1
    eps: float = 1e-6
    q_: float | None = None
    normalized_: bool = False

    def fit(self, y_cal: np.ndarray, pred_cal: np.ndarray, sigma_cal: np.ndarray | None = None):
        y_cal = np.asarray(y_cal, dtype=float)
        pred_cal = np.asarray(pred_cal, dtype=float)
        resid = np.abs(y_cal - pred_cal)
        if sigma_cal is not None:
            resid = resid / (np.asarray(sigma_cal, dtype=float) + self.eps)
            self.normalized_ = True
        self.q_ = conformal_quantile(resid, self.alpha)
        return self

    def half_width(self, sigma: np.ndarray | None = None, n: int | None = None) -> np.ndarray:
        if self.q_ is None:
            raise RuntimeError("call fit() first")
        if self.normalized_:
            if sigma is None:
                raise ValueError("this regressor was fitted with sigma; pass sigma for the queries")
            return self.q_ * (np.asarray(sigma, dtype=float) + self.eps)
        if n is None:
            raise ValueError("pass n (the number of queries) for an unnormalised regressor")
        return np.full(n, self.q_, dtype=float)

    def predict_interval(self, pred: np.ndarray, sigma: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        pred = np.asarray(pred, dtype=float)
        hw = self.half_width(sigma=sigma, n=pred.size)
        return pred - hw, pred + hw


def conformal_prediction_sets(
    proba_cal: np.ndarray,
    y_cal: np.ndarray,
    proba_test: np.ndarray,
    alpha: float = 0.1,
) -> tuple[np.ndarray, float]:
    """LAC conformal prediction sets for a probabilistic classifier.

    ``proba_*`` are (n, n_classes) probability matrices whose column j is the probability of
    class index j; ``y_cal`` holds integer class indices. Returns a boolean (n_test, n_classes)
    membership matrix and the score threshold. A singleton set is a "reliable" prediction; an
    empty or full set flags a compound the model cannot resolve at this confidence level.
    """
    proba_cal = np.asarray(proba_cal, dtype=float)
    proba_test = np.asarray(proba_test, dtype=float)
    y_cal = np.asarray(y_cal).astype(int)
    scores = 1.0 - proba_cal[np.arange(y_cal.size), y_cal]
    q = conformal_quantile(scores, alpha)
    return (1.0 - proba_test) <= q, q


def probability_confidence(proba_positive: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Binary-classification confidence ``|p1 - p0| = |2 p1 - 1|`` and a reliable flag.

    Follows the confidence-estimation idea of Mathea et al. (2016) as implemented in MetaQSAR: a
    larger gap between the class posteriors means a more reliable prediction, and ``threshold``
    (default 0.5, i.e. p1 <= 0.25 or p1 >= 0.75) separates reliable from unreliable calls.
    """
    p = np.asarray(proba_positive, dtype=float)
    conf = np.abs(2.0 * p - 1.0)
    return conf, conf >= threshold


def reliability_curve(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10):
    """Equal-width reliability-diagram bins for binary classification.

    Returns (bin_mean_confidence, bin_observed_frequency, bin_count); empty bins are NaN with
    count 0 so the arrays always have ``n_bins`` entries.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    conf = np.full(n_bins, np.nan)
    freq = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = idx == b
        count[b] = int(mask.sum())
        if count[b]:
            conf[b] = p[mask].mean()
            freq[b] = y[mask].mean()
    return conf, freq, count


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Binary ECE: count-weighted mean |observed frequency - mean predicted probability|."""
    conf, freq, count = reliability_curve(y_true, proba, n_bins=n_bins)
    total = count.sum()
    if total == 0:
        return float("nan")
    mask = count > 0
    return float(np.sum(count[mask] * np.abs(freq[mask] - conf[mask])) / total)


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean squared difference between the positive-class probability and the 0/1 outcome."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba, dtype=float)
    return float(np.mean((p - y) ** 2))
