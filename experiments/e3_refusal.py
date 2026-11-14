"""Experiment 3: does the "not sure, refer" rule keep its promise at a new site?

The rule promises that the true answer is inside the prediction set 90% of the time.
`recal_n = 0` calibrates on the training site. `recal_n > 0` calibrates on that many
labelled photos from the new site, which are then left out of scoring.

    python -m experiments.e3_refusal
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.conformal import calibrate, prediction_sets, summarize

from .common import ALL, AQUA, BLUE, MAIN, MUTED, NAMES, ORANGE, SEEDS, YELLOW, load_preds, plt, source_val

ALPHA = 0.10
RECAL = [25, 50, 100]
DRAWS = 20
MODELS = ["probe", "resnet_color"]
COLOURS = dict(zip(ALL, [BLUE, ORANGE, AQUA, YELLOW]))


def recalibrated(y, p, groups, n: int, rng) -> dict:
    """Calibrate on about `n` photos from randomly drawn patients; score everyone else."""
    order = rng.permutation(np.unique(groups))
    sizes = np.array([(groups == g).sum() for g in order])
    chosen = order[: int(np.searchsorted(np.cumsum(sizes), n) + 1)]
    held = np.isin(groups, chosen)
    qhat = calibrate(p[held][:n], y[held][:n], ALPHA)
    return {**summarize(prediction_sets(p[~held], qhat), y[~held]), "n": int((~held).sum())}


def rows_for(results, splits_dir, model, train, test) -> list[dict]:
    by_n = {n: [] for n in [0] + RECAL}
    for seed in SEEDS:
        val = source_val(results, splits_dir, model, train, seed)
        preds = load_preds(results, splits_dir, model, train, seed, test, "test")
        if val is None or preds is None:
            continue
        y, p, groups = preds
        qhat = calibrate(val[1], val[0], ALPHA)
        by_n[0].append({**summarize(prediction_sets(p, qhat), y), "n": len(y)})
        rng = np.random.default_rng(seed)
        for n in RECAL:
            if len(y) >= 2 * n:  # keep at least as many photos for scoring as for calibrating
                by_n[n].append(pd.DataFrame([recalibrated(y, p, groups, n, rng) for _ in range(DRAWS)]).mean().to_dict())
    return [{"model": model, "train": train, "test": test, "recal_n": n, **pd.DataFrame(v).mean().to_dict()}
            for n, v in by_n.items() if v]


def risk_coverage(y, p):
    """Answer the most confident photos first: error among answered against fraction answered."""
    order = np.argsort(-np.abs(p - 0.5))
    wrong = ((p >= 0.5) != (y == 1))[order]
    answered = np.arange(1, len(y) + 1)
    return answered / len(y), np.cumsum(wrong) / answered


def figure(table: pd.DataFrame, results, splits_dir, out: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))

    cross = table[(table.model == MODELS[0]) & (table.train != table.test)]
    mean = cross.groupby(["test", "recal_n"]).coverage.mean().unstack("test")
    for test in [s for s in ALL if s in mean.columns]:
        left.plot(range(len(mean)), mean[test], marker="o", markersize=7, linewidth=2, color=COLOURS[test], label=NAMES[test])
    left.axhline(1 - ALPHA, color=MUTED, linewidth=1, linestyle="--")
    left.text(len(mean) - 1, 1 - ALPHA, " promised 90%", va="bottom", ha="right", color=MUTED, fontsize=9)
    left.set_xticks(range(len(mean)), ["none" if n == 0 else str(n) for n in mean.index])
    left.set_xlabel("labelled photos from the new site used to recalibrate")
    left.set_ylabel("coverage at the new site")
    left.set_title("Does the promise hold after moving?")
    left.legend(title="tested on")

    for same_site, colour, label in [(True, BLUE, "same site"), (False, ORANGE, "new site")]:
        drawn = False
        for train in MAIN:
            for test in MAIN:
                preds = load_preds(results, splits_dir, MODELS[0], train, 0, test, "test")
                if preds is None or (train == test) != same_site:
                    continue
                answered, error = risk_coverage(preds[0], preds[1])
                right.plot(answered, error, color=colour, linewidth=2, alpha=0.75, label=None if drawn else label)
                drawn = True
    right.set_xlabel("fraction of photos the model answers (most confident first)")
    right.set_ylabel("error among answered photos")
    right.set_title("What refusing buys")
    right.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main(results: Path = Path("results"), splits_dir: Path = Path("splits")) -> pd.DataFrame:
    rows = [r for m in MODELS for train in MAIN for test in ALL for r in rows_for(results, splits_dir, m, train, test)]
    table = pd.DataFrame(rows)
    table.to_csv(Path(results) / "e3_refusal.csv", index=False, float_format="%.4f")
    figure(table, results, splits_dir, Path(results) / "e3_refusal.png")
    return table


if __name__ == "__main__":
    print(main().to_string(index=False, float_format="%.3f"))
