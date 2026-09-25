"""Unit tests for qsarena.uncertainty: conformal coverage and calibration metrics."""

from __future__ import annotations

import numpy as np
import pytest

from qsarena.uncertainty import (
    SplitConformalRegressor,
    brier_score,
    conformal_prediction_sets,
    conformal_quantile,
    expected_calibration_error,
    probability_confidence,
    reliability_curve,
)


def test_conformal_quantile_finite_sample_rank():
    scores = np.arange(1, 11, dtype=float)  # n = 10
    # ceil(11 * 0.9) = 10 -> the 10th smallest score.
    assert conformal_quantile(scores, 0.1) == 10.0
    # ceil(11 * 0.8) = 9
    assert conformal_quantile(scores, 0.2) == 9.0
    # ceil(11 * 0.95) = 11 > n -> infinite threshold.
    assert conformal_quantile(scores, 0.05) == np.inf


@pytest.mark.parametrize("normalized", [False, True])
def test_split_conformal_regression_coverage_near_nominal(normalized):
    rng = np.random.default_rng(42)
    alpha, n_cal, n_test, reps = 0.1, 400, 2000, 20
    coverages = []
    for _ in range(reps):
        sigma_cal = rng.uniform(0.2, 2.0, n_cal)
        sigma_te = rng.uniform(0.2, 2.0, n_test)
        pred_cal, pred_te = rng.normal(size=n_cal), rng.normal(size=n_test)
        y_cal = pred_cal + rng.normal(scale=sigma_cal)
        y_te = pred_te + rng.normal(scale=sigma_te)
        cr = SplitConformalRegressor(alpha=alpha).fit(y_cal, pred_cal, sigma_cal if normalized else None)
        lo, hi = cr.predict_interval(pred_te, sigma_te if normalized else None)
        coverages.append(np.mean((y_te >= lo) & (y_te <= hi)))
    assert np.mean(coverages) == pytest.approx(1 - alpha, abs=0.02)


def test_normalized_intervals_scale_with_sigma():
    cr = SplitConformalRegressor(alpha=0.1).fit(np.array([0.0, 1, 2, 3]), np.zeros(4), np.ones(4))
    hw = cr.half_width(sigma=np.array([1.0, 2.0]))
    assert hw[1] == pytest.approx(2 * hw[0], rel=1e-5)


def test_unnormalized_requires_n_and_normalized_requires_sigma():
    cr = SplitConformalRegressor().fit(np.zeros(20), np.zeros(20))
    with pytest.raises(ValueError):
        cr.half_width()
    crn = SplitConformalRegressor().fit(np.zeros(20), np.zeros(20), np.ones(20))
    with pytest.raises(ValueError):
        crn.half_width(n=3)


def test_conformal_classification_coverage_near_nominal():
    rng = np.random.default_rng(7)
    alpha, reps, n_cal, n_test = 0.1, 20, 500, 2000
    covs = []
    for _ in range(reps):
        p_cal = rng.uniform(size=n_cal)
        y_cal = (rng.uniform(size=n_cal) < p_cal).astype(int)  # calibrated probabilities
        p_te = rng.uniform(size=n_test)
        y_te = (rng.uniform(size=n_test) < p_te).astype(int)
        sets, _ = conformal_prediction_sets(
            np.column_stack([1 - p_cal, p_cal]), y_cal, np.column_stack([1 - p_te, p_te]), alpha
        )
        covs.append(sets[np.arange(n_test), y_te].mean())
    assert np.mean(covs) == pytest.approx(1 - alpha, abs=0.02)


def test_probability_confidence_threshold():
    conf, reliable = probability_confidence(np.array([0.5, 0.74, 0.76, 0.1]), threshold=0.5)
    np.testing.assert_allclose(conf, [0.0, 0.48, 0.52, 0.8])
    assert reliable.tolist() == [False, False, True, True]


def test_ece_and_brier_on_toy_set():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, p) == 0.0
    assert brier_score(y, p) == 0.0
    # Constant 0.5 prediction on a balanced set: perfectly calibrated, Brier 0.25.
    p_half = np.full(4, 0.5)
    assert expected_calibration_error(y, p_half) == pytest.approx(0.0)
    assert brier_score(y, p_half) == pytest.approx(0.25)
    # Always 0.9 on a set that is 50% positive: ECE = |0.5 - 0.9| = 0.4.
    assert expected_calibration_error(y, np.full(4, 0.9)) == pytest.approx(0.4)


def test_reliability_curve_bins_and_counts():
    y = np.array([0, 1, 1, 1])
    p = np.array([0.05, 0.95, 0.95, 0.55])
    conf, freq, count = reliability_curve(y, p, n_bins=10)
    assert count.sum() == 4 and len(conf) == 10
    assert count[0] == 1 and count[9] == 2 and count[5] == 1
    assert freq[9] == 1.0 and np.isnan(freq[3])
