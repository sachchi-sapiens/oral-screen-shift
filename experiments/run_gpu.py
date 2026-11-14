"""Train every model and save its predictions, so tables and figures can be made without a GPU.

    python -m experiments.run_gpu --data data --stage features
    python -m experiments.run_gpu --data data --stage probe          # runs on a laptop CPU
    python -m experiments.run_gpu --data data --stage resnet         # needs a GPU
    python -m experiments.run_gpu --data data --stage resnet_color   # needs a GPU

A run that is interrupted can be started again: anything already saved is skipped.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import extract
from src.splits import load_splits
from src.train import finetune, fit_probe, predict

from .common import MAIN, POOLED, SEEDS, pred_path, training_sites

RESULTS, FEATURES, CHECKPOINTS = Path("results"), Path("features"), Path("checkpoints")
EVAL_SPLITS = ["val", "test"]


def targets(splits: pd.DataFrame):
    """Every (site, split) a model is scored on."""
    return [(site, split) for site in splits.site.unique() for split in EVAL_SPLITS
            if ((splits.site == site) & (splits.split == split)).any()]


def done(model, train, seed, splits) -> bool:
    return all(pred_path(RESULTS, model, train, seed, site, split).exists() for site, split in targets(splits))


def save(model, train, seed, site, split, p) -> None:
    path = pred_path(RESULTS, model, train, seed, site, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, p.astype(np.float32))


def site_features(data: Path, splits: pd.DataFrame, site: str) -> np.ndarray:
    return extract(splits[splits.site == site], data, FEATURES / f"{site}.npy")


def run_probe(data: Path, train: str, seed: int, splits: pd.DataFrame) -> None:
    feats = {site: site_features(data, splits, site) for site in splits.site.unique()}

    def rows(sites, split):
        X = np.concatenate([feats[s][(splits[splits.site == s].split == split).to_numpy()] for s in sites])
        y = np.concatenate([splits[(splits.site == s) & (splits.split == split)].label.to_numpy() for s in sites])
        return X, y

    sources = training_sites(train)
    model = fit_probe(*rows(sources, "train"), *rows(sources, "val"))
    for site, split in targets(splits):
        save("probe", train, seed, site, split, model.predict_proba(rows([site], split)[0])[:, 1])


def run_resnet(data: Path, model_name: str, train: str, seed: int, splits: pd.DataFrame) -> None:
    sources = splits[splits.site.isin(training_sites(train))]
    checkpoint = finetune(sources[sources.split == "train"], sources[sources.split == "val"], data,
                          CHECKPOINTS / f"{model_name}__{train}__seed{seed}.pt",
                          aug="color" if model_name == "resnet_color" else "basic", seed=seed)
    for site, split in targets(splits):
        save(model_name, train, seed, site, split,
             predict(checkpoint, splits[(splits.site == site) & (splits.split == split)], data))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--stage", required=True, choices=["features", "probe", "resnet", "resnet_color"])
    a = p.parse_args()

    if a.stage == "features":
        splits = load_splits(0)
        for site in splits.site.unique():
            print(site, site_features(a.data, splits, site).shape)
    else:
        # the plain ResNet is the published-paper baseline, so it is only trained on single sites
        trains = MAIN if a.stage == "resnet" else MAIN + [POOLED]
        for seed in SEEDS:
            splits = load_splits(seed)
            for train in trains:
                if done(a.stage, train, seed, splits):
                    print(f"{a.stage} {train} seed {seed}: skipped, already saved")
                    continue
                print(f"{a.stage} {train} seed {seed}")
                if a.stage == "probe":
                    run_probe(a.data, train, seed, splits)
                else:
                    run_resnet(a.data, a.stage, train, seed, splits)
