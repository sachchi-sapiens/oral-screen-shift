"""Experiment 4: can the features tell which dataset a healthy photo came from?

If they can, a model has an easy shortcut. One dataset is mostly lesions and another mostly
healthy mouths, so "looks like dataset A" predicts the label without looking at any lesion,
and that shortcut is worthless at a new site.

Only do-not-refer photos are used, so the classifier cannot lean on lesions to tell sites apart.

    python -m experiments.e4_site
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.splits import load_splits


def main(features: Path = Path("features"), splits_dir: Path = Path("splits"),
         results: Path = Path("results")) -> pd.DataFrame:
    splits = load_splits(0, splits_dir)
    healthy = {}
    for site in sorted(splits.site.unique()):
        rows = splits[splits.site == site]
        keep = (rows.label == 0).to_numpy()
        healthy[site] = (np.load(Path(features) / f"{site}.npy")[keep], rows.group.to_numpy()[keep])

    table = []
    for a, b in combinations(healthy, 2):
        X = np.concatenate([healthy[a][0], healthy[b][0]])
        which = np.r_[np.zeros(len(healthy[a][0])), np.ones(len(healthy[b][0]))]
        groups = np.concatenate([healthy[a][1], healthy[b][1]])
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
        # patients are kept whole within a fold, so the score is not "recognise this patient again"
        score = cross_val_score(model, X, which, groups=groups, cv=GroupKFold(5), scoring="balanced_accuracy")
        table.append({"site_a": a, "site_b": b, "healthy_photos_a": len(healthy[a][0]),
                      "healthy_photos_b": len(healthy[b][0]), "balanced_accuracy": score.mean()})
    table = pd.DataFrame(table)
    table.to_csv(Path(results) / "e4_site.csv", index=False, float_format="%.4f")
    return table


if __name__ == "__main__":
    print(main().to_string(index=False, float_format="%.3f"))
