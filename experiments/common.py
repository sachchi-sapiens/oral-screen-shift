"""What the experiment scripts share: where predictions live, and how a cell's threshold is chosen."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate import pick_threshold

MAIN = ["smartom", "code", "china"]       # sites large enough to train on
ALL = MAIN + ["karnataka"]                # karnataka is a test site only
SEEDS = [0, 1, 2]
NAMES = {"smartom": "SMART-OM (India)", "code": "CODE (India)", "china": "China", "karnataka": "Karnataka (India)"}

# Fixed colours per method, so a method keeps its colour in every figure.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def pred_path(results: Path, model: str, train: str, seed: int, test: str, split: str) -> Path:
    return Path(results) / "preds" / f"{model}__{train}__seed{seed}__{test}__{split}.npy"


def training_sites(train: str) -> list[str]:
    """'all-but-china' was trained on the other two main sites."""
    return [s for s in MAIN if s != train.removeprefix("all-but-")] if train.startswith("all-but-") else [train]


def load_preds(results: Path, splits_dir: Path, model: str, train: str, seed: int, test: str, split: str):
    """Labels, refer probabilities and patient groups for one saved prediction file, or None if it is missing."""
    path = pred_path(results, model, train, seed, test, split)
    if not path.exists():
        return None
    rows = pd.read_csv(Path(splits_dir) / f"seed{seed}.csv")
    rows = rows[(rows.site == test) & (rows.split == split)]
    p = np.load(path)
    assert len(p) == len(rows), f"{path.name}: {len(p)} predictions for {len(rows)} photos"
    return rows.label.to_numpy(), p, rows.group.to_numpy()


def source_val(results: Path, splits_dir: Path, model: str, train: str, seed: int):
    """Validation labels and scores from the site(s) the model was trained on. Never from the test site."""
    parts = [load_preds(results, splits_dir, model, train, seed, s, "val") for s in training_sites(train)]
    if any(part is None for part in parts):
        return None
    return np.concatenate([part[0] for part in parts]), np.concatenate([part[1] for part in parts])


def threshold_for(results: Path, splits_dir: Path, model: str, train: str, seed: int):
    val = source_val(results, splits_dir, model, train, seed)
    return None if val is None else pick_threshold(*val)
