"""Metrics for a refer / do-not-refer classifier, with intervals that respect patients."""

import numpy as np
from sklearn.metrics import roc_auc_score


def pick_threshold(y_val, p_val, min_sensitivity: float = 0.90) -> float:
    """Highest cut-off that still catches `min_sensitivity` of the refer cases on validation data."""
    y_val, p_val = np.asarray(y_val), np.asarray(p_val)
    positives = np.sort(p_val[y_val == 1])
    allowed_misses = int(np.floor(len(positives) * (1 - min_sensitivity) + 1e-9))
    return float(positives[allowed_misses])


def metrics(y, p, threshold: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    refer = p >= threshold
    pos, neg = y == 1, y == 0
    both_classes = pos.any() and neg.any()

    # Calibration error: in each probability bin, how far is the mean prediction from the refer rate?
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in np.unique(bins))

    return {
        "auroc": float(roc_auc_score(y, p)) if both_classes else np.nan,
        "sensitivity": float(refer[pos].mean()) if pos.any() else np.nan,
        "specificity": float((~refer[neg]).mean()) if neg.any() else np.nan,
        "ece": float(ece),
        "n": int(len(y)),
        "n_refer": int(pos.sum()),
    }


def bootstrap_ci(y, p, groups, threshold: float, n_boot: int = 1000, seed: int = 0) -> dict:
    """95% intervals from resampling whole groups (patients), since one patient's photos are not independent."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    rows_of = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[i] for i in rng.integers(0, len(rows_of), len(rows_of))])
        draws.append(metrics(y[idx], p[idx], threshold))
    out = {}
    for k in ("auroc", "sensitivity", "specificity", "ece"):
        lo, hi = np.nanpercentile([d[k] for d in draws], [2.5, 97.5])
        out[k] = (float(lo), float(hi))
    return out
