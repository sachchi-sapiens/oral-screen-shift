"""Experiment 2: which fix recovers performance at a site the model never saw?

For each held-out site, compare models trained on one other site (averaged over the choices)
with models trained on both other sites together.

    python -m experiments.e2_fixes
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .common import AQUA, BLUE, MAGENTA, MAIN, NAMES, ORANGE, YELLOW, plt
from .e1_matrix import cell

# method name -> (model, trained on both other sites?, colour)
METHODS = {
    "ResNet, one site": ("resnet", False, BLUE),
    "ResNet + colour augmentation, one site": ("resnet_color", False, ORANGE),
    "DINOv2 probe, one site": ("probe", False, AQUA),
    "ResNet + colour augmentation, two sites": ("resnet_color", True, YELLOW),
    "DINOv2 probe, two sites": ("probe", True, MAGENTA),
}
KEEP = ["auroc", "sensitivity", "specificity", "ece"]


def main(results: Path = Path("results"), splits_dir: Path = Path("splits")) -> pd.DataFrame:
    rows = []
    for test in MAIN:
        for method, (model, pooled, _) in METHODS.items():
            trains = [f"all-but-{test}"] if pooled else [s for s in MAIN if s != test]
            cells = [c for c in (cell(results, splits_dir, model, train, test) for train in trains) if c]
            if cells:
                rows.append({"test": test, "method": method, "sources": len(cells),
                             **pd.DataFrame(cells)[KEEP].mean().to_dict()})
    table = pd.DataFrame(rows)
    table.to_csv(Path(results) / "e2_fixes.csv", index=False, float_format="%.4f")

    fig, ax = plt.subplots(figsize=(9, 4))
    tests = [s for s in MAIN if s in set(table.test)]
    width = 0.8 / len(METHODS)
    for k, (method, (_, _, colour)) in enumerate(METHODS.items()):
        t = table[table.method == method].set_index("test").reindex(tests)
        x = np.arange(len(tests)) + (k - (len(METHODS) - 1) / 2) * width
        ax.bar(x, t.auroc - 0.5, width * 0.9, bottom=0.5, color=colour, label=method)
        for xi, value in zip(x, t.auroc):
            if not np.isnan(value):
                ax.text(xi, value + 0.005, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(tests)), [NAMES[s] for s in tests])
    ax.set_xlabel("site held out for testing")
    ax.set_ylabel("AUROC (0.5 = chance)")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Performance at a site the model never saw, by fix")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(Path(results) / "e2_fixes.png", dpi=160)
    plt.close(fig)
    return table


if __name__ == "__main__":
    print(main().to_string(index=False, float_format="%.3f"))
