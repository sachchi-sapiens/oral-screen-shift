import numpy as np
import pytest

from src.evaluate import bootstrap_ci, metrics, pick_threshold


def test_metrics_hand_computed():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1])
    m = metrics(y, p, threshold=0.5)
    assert m["auroc"] == 0.75 and m["sensitivity"] == 0.5 and m["specificity"] == 0.5
    assert m["n"] == 4 and m["n_refer"] == 2


def test_ece_zero_when_perfectly_calibrated():
    y = np.array([1] * 8 + [0] * 2)
    p = np.full(10, 0.8)
    assert metrics(y, p, 0.5)["ece"] == pytest.approx(0.0)


def test_ece_large_when_overconfident():
    y = np.array([1] * 5 + [0] * 5)
    p = np.full(10, 0.9)
    assert metrics(y, p, 0.5)["ece"] == pytest.approx(0.4)


def test_single_class_gives_nan_auroc_not_crash():
    m = metrics(np.zeros(5, int), np.linspace(0, 1, 5), 0.5)
    assert np.isnan(m["auroc"]) and np.isnan(m["sensitivity"]) and m["specificity"] == 0.4


def test_pick_threshold_meets_sensitivity():
    y = np.array([1] * 10 + [0] * 10)
    p = np.r_[np.linspace(0.3, 0.95, 10), np.linspace(0.0, 0.6, 10)]
    t = pick_threshold(y, p, 0.90)
    assert ((p >= t) & (y == 1)).sum() / 10 >= 0.90
    assert ((p >= t + 0.08) & (y == 1)).sum() / 10 < 0.90  # not lower than it needs to be


def test_bootstrap_resamples_groups_and_is_seeded():
    rng = np.random.default_rng(0)
    y = np.repeat(rng.integers(0, 2, 40), 5)            # 40 patients, 5 identical images each
    p = np.clip(y * 0.3 + np.repeat(rng.uniform(0.2, 0.6, 40), 5), 0, 1)
    groups = np.repeat(np.arange(40), 5)
    ci = bootstrap_ci(y, p, groups, 0.5, n_boot=200, seed=1)
    assert ci["auroc"][0] <= metrics(y, p, 0.5)["auroc"] <= ci["auroc"][1]
    assert ci == bootstrap_ci(y, p, groups, 0.5, n_boot=200, seed=1)
    # treating the 200 copies as independent would give a falsely narrow interval
    by_row = bootstrap_ci(y, p, np.arange(200), 0.5, n_boot=200, seed=1)
    assert ci["auroc"][1] - ci["auroc"][0] > by_row["auroc"][1] - by_row["auroc"][0]
