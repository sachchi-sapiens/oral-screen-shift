"""Experiment 1: train on each site, test on every site.

    python -m experiments.e1_matrix
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import bootstrap_ci, metrics

from .common import ALL, MAIN, NAMES, SEEDS, load_preds, plt, threshold_for

MODELS = {"probe": "Linear probe on frozen DINOv2", "resnet": "Fine-tuned ResNet-50"}


def cell(results, splits_dir, model, train, test):
    """Mean over seeds of each metric for one (model, train site, test site).

    Each seed is a different split and a different trained model. Every seed gets its own
    patient-level bootstrap interval, and the reported interval is the mean of those ends.
    """
    per_seed = []
    for seed in SEEDS:
        threshold = threshold_for(results, splits_dir, model, train, seed)
        preds = load_preds(results, splits_dir, model, train, seed, test, "test")
        if threshold is None or preds is None:
            continue
        y, p, groups = preds
        row = {**metrics(y, p, threshold), "threshold": threshold}
        for name, (lo, hi) in bootstrap_ci(y, p, groups, threshold).items():
            row[f"{name}_lo"], row[f"{name}_hi"] = lo, hi
        per_seed.append(row)
    if not per_seed:
        return None
    row = pd.DataFrame(per_seed).mean().to_dict()
    return {"model": model, "train": train, "test": test, "seeds": len(per_seed), **row}


def figure(table: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 3.6))
    for ax, (model, title) in zip(axes, MODELS.items()):
        t = table[table.model == model]
        if t.empty:  # this model has not been trained yet
            ax.set_axis_off()
            continue
        trains = [s for s in MAIN if s in set(t.train)]
        tests = [s for s in ALL if s in set(t.test)]
        grid = t.pivot(index="train", columns="test", values="auroc").reindex(index=trains, columns=tests)
        ax.imshow(grid, cmap="Blues", vmin=0.5, vmax=1.0, aspect="auto")
        for i in range(len(trains)):
            for j in range(len(tests)):
                value = grid.iloc[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                            color="white" if value > 0.8 else "#0b0b0b", fontweight="bold")
        ax.set_xticks(range(len(tests)), [NAMES[s] for s in tests], rotation=20, ha="right")
        ax.set_yticks(range(len(trains)), [NAMES[s] for s in trains])
        ax.set_xlabel("tested on")
        ax.set_ylabel("trained on")
        ax.set_title(title)
        ax.grid(False)
    fig.suptitle("AUROC for refer / do not refer. Diagonal = same site; off-diagonal = moved to a new site.", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main(results: Path = Path("results"), splits_dir: Path = Path("splits")) -> pd.DataFrame:
    rows = [cell(results, splits_dir, m, train, test) for m in MODELS for train in MAIN for test in ALL]
    table = pd.DataFrame([r for r in rows if r is not None])
    table.to_csv(Path(results) / "e1_matrix.csv", index=False, float_format="%.4f")
    figure(table, Path(results) / "e1_matrix.png")
    return table


if __name__ == "__main__":
    print(main().to_string(index=False, float_format="%.3f"))
