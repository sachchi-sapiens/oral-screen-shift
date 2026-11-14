import numpy as np

from src.conformal import calibrate, prediction_sets, summarize


def test_coverage_holds_on_exchangeable_data():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=4000)
    y = (rng.uniform(size=4000) < p).astype(int)
    q = calibrate(p[:2000], y[:2000], alpha=0.10)
    cov = summarize(prediction_sets(p[2000:], q), y[2000:])["coverage"]
    assert 0.88 <= cov <= 0.93


def test_tiny_calibration_set_refers_everything():
    q = calibrate(np.array([0.9, 0.2, 0.7]), np.array([1, 0, 1]), alpha=0.10)  # 3 < 1/alpha
    s = prediction_sets(np.array([0.99, 0.01]), q)
    assert np.isinf(q) and s.all() and summarize(s, np.array([1, 0]))["answered"] == 0.0


def test_confident_predictions_are_answered():
    s = prediction_sets(np.array([0.99, 0.01, 0.5]), qhat=0.2)
    assert s.tolist() == [[False, True], [True, False], [False, False]]


def test_missed_refer_counts_confident_wrong_negatives():
    s = np.array([[True, False], [False, True]])
    y = np.array([1, 1])
    assert summarize(s, y)["missed_refer"] == 0.5
