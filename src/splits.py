"""Keep every patient on one side of a split, and catch the same photo appearing twice."""

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SPLITS_DIR = Path("splits")


def dhash(path: Path) -> int:
    """64-bit difference hash: survives resizing and re-compression, changes for a different picture."""
    small = np.asarray(Image.open(path).convert("L").resize((9, 8)), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _bits_apart(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bitwise_count(a[:, None] ^ b[None, :])


def add_groups(df: pd.DataFrame, root: Path, max_bits: int = 4) -> pd.DataFrame:
    """A group is a patient where the dataset names one; otherwise a cluster of near-identical photos."""
    df = df.reset_index(drop=True).copy()
    df["dhash"] = [dhash(Path(root) / p) for p in df.path]
    df["group"] = None
    for site, rows in df.groupby("site"):
        has_id = rows.patient_id.notna()
        df.loc[rows.index[has_id], "group"] = site + ":" + rows.patient_id[has_id].astype(str)

        loose = rows.index[~has_id].to_numpy()
        if len(loose) == 0:
            continue
        hashes = df.dhash[loose].to_numpy(dtype=np.uint64)
        parent = np.arange(len(loose))  # union-find: near-duplicates end up under one parent

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, j in zip(*np.nonzero(np.triu(_bits_apart(hashes, hashes) <= max_bits, k=1))):
            parent[find(j)] = find(i)
        # name each cluster after its first path, so the name is the same whatever order the files were listed in
        roots = np.array([find(i) for i in range(len(loose))])
        paths = df.path[loose].to_numpy()
        df.loc[loose, "group"] = [f"{site}:{min(paths[roots == r])}" for r in roots]
    return df


def cross_site_duplicates(df: pd.DataFrame, max_bits: int = 4) -> pd.DataFrame:
    """Pairs of near-identical photos that sit in two different datasets."""
    pairs = []
    sites = sorted(df.site.unique())
    for n, a in enumerate(sites):
        for b in sites[n + 1:]:
            da, db = df[df.site == a], df[df.site == b]
            apart = _bits_apart(da.dhash.to_numpy(np.uint64), db.dhash.to_numpy(np.uint64))
            for i, j in zip(*np.nonzero(apart <= max_bits)):
                pairs.append({"site_a": a, "path_a": da.path.iloc[i], "site_b": b, "path_b": db.path.iloc[j],
                              "bits": int(apart[i, j])})
    return pd.DataFrame(pairs, columns=["site_a", "path_a", "site_b", "path_b", "bits"])


def make_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """70/15/15 by group inside each site, balanced on whether the group has any refer image."""
    split_of = {}
    group_label = df.groupby(["site", "group"], sort=True).label.max()
    for (site, label), groups in group_label.groupby([group_label.index.get_level_values("site"), group_label]):
        # seeded per site and class, so adding another dataset never reshuffles this one
        rng = np.random.default_rng([seed, zlib.crc32(site.encode()), int(label)])
        names = rng.permutation(groups.index.get_level_values("group").to_numpy())
        a, b = round(0.70 * len(names)), round(0.85 * len(names))
        split_of.update({g: "train" for g in names[:a]})
        split_of.update({g: "val" for g in names[a:b]})
        split_of.update({g: "test" for g in names[b:]})
    return df.assign(split=df.group.map(split_of))


RESTRICTED = {"srilanka"}  # licence forbids redistribution: its rows go in a file that is never committed


def save_splits(df: pd.DataFrame, seed: int, directory: Path = SPLITS_DIR) -> None:
    Path(directory).mkdir(exist_ok=True)
    rows = df[["path", "site", "group", "label", "split"]]
    private = rows.site.isin(RESTRICTED)
    rows[~private].to_csv(Path(directory) / f"seed{seed}.csv", index=False)
    if private.any():
        rows[private].to_csv(Path(directory) / f"seed{seed}.restricted.csv", index=False)


def load_splits(seed: int, directory: Path = SPLITS_DIR) -> pd.DataFrame:
    files = [Path(directory) / f"seed{seed}.csv", Path(directory) / f"seed{seed}.restricted.csv"]
    return pd.concat([pd.read_csv(f) for f in files if f.exists()], ignore_index=True)
