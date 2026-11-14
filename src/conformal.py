"""Split conformal prediction for two classes: answer only when one class is plausible, else refer.

`p` is always the model's probability of "refer". A prediction set is a pair of booleans
(do-not-refer plausible, refer plausible). Exactly one true means the model answers.
"""

import numpy as np


def calibrate(p_cal, y_cal, alpha: float = 0.10) -> float:
    """Score = 1 - probability given to the true class. Return the quantile that covers 1 - alpha."""
    p_cal, y_cal = np.asarray(p_cal, float), np.asarray(y_cal)
    scores = 1 - np.where(y_cal == 1, p_cal, 1 - p_cal)
    n = len(scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    if level > 1:  # too few calibration points to promise anything: every class stays plausible
        return float("inf")
    return float(np.quantile(scores, level, method="higher"))


def prediction_sets(p, qhat: float) -> np.ndarray:
    p = np.asarray(p, float)
    return np.column_stack([p <= qhat, 1 - p <= qhat])


def summarize(sets, y) -> dict:
    sets, y = np.asarray(sets), np.asarray(y)
    answered = sets.sum(axis=1) == 1
    said_refer = sets[:, 1]
    wrong = answered & (said_refer != (y == 1))
    return {
        "coverage": float(sets[np.arange(len(y)), y].mean()),
        "answered": float(answered.mean()),
        "error_when_answered": float(wrong.sum() / answered.sum()) if answered.any() else np.nan,
        "missed_refer": float((wrong & (y == 1)).sum() / (y == 1).sum()) if (y == 1).any() else np.nan,
    }
