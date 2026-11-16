import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.splits import add_groups, cross_site_duplicates, make_splits


def fake_index(n_groups=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        label = int(rng.uniform() < 0.3)
        for i in range(rng.integers(1, 5)):
            rows.append({"path": f"{g}_{i}.jpg", "site": "smartom", "label": label, "group": f"smartom:P{g}"})
    return pd.DataFrame(rows)


@pytest.fixture
def index_with_groups():
    return fake_index()


def test_no_group_in_two_splits(index_with_groups):
    s = make_splits(index_with_groups, seed=0)
    assert (s.groupby("group").split.nunique() == 1).all()


def test_fractions_roughly_70_15_15(index_with_groups):
    f = make_splits(index_with_groups, seed=0).drop_duplicates("group").split.value_counts(normalize=True)
    assert abs(f["train"] - 0.70) < 0.05 and abs(f["val"] - 0.15) < 0.05


def test_same_seed_same_split_and_other_seed_differs(index_with_groups):
    a = make_splits(index_with_groups, 0).split
    assert a.equals(make_splits(index_with_groups, 0).split)
    assert not a.equals(make_splits(index_with_groups, 1).split)


def test_mixed_label_patient_stays_together(index_with_groups):
    df = index_with_groups.copy()
    mixed = df.index[df.group == "smartom:P1"]
    df.loc[mixed, "label"] = 0
    df = pd.concat([df, df.loc[mixed[:1]].assign(path="extra.jpg", label=1)], ignore_index=True)
    s = make_splits(df, seed=0)
    assert s[s.group == "smartom:P1"].split.nunique() == 1


def test_each_split_has_both_classes(index_with_groups):
    assert (make_splits(index_with_groups, 0).groupby("split").label.nunique() == 2).all()


def test_sites_are_split_separately():
    df = pd.concat([fake_index(), fake_index(60, seed=1).assign(site="china")], ignore_index=True)
    df["group"] = df.site + ":" + df.group.str.split(":").str[1]
    s = make_splits(df, 0)
    assert set(s[s.site == "china"].split) == {"train", "val", "test"}


def save_images(tmp_path):
    """Two JPEG saves of one picture (different quality) and one unrelated picture."""
    rng = np.random.default_rng(0)
    gradient = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    picture = np.stack([gradient, gradient.T, gradient], axis=2)
    Image.fromarray(picture).save(tmp_path / "a.jpg", quality=95)
    Image.fromarray(picture).save(tmp_path / "b.jpg", quality=60)
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(tmp_path / "c.jpg")


def test_near_duplicates_share_group(tmp_path):
    save_images(tmp_path)
    df = pd.DataFrame({"path": ["a.jpg", "b.jpg", "c.jpg"], "site": "china", "patient_id": None, "label": 0})
    g = add_groups(df, tmp_path).group
    assert g[0] == g[1] != g[2]


def test_patient_id_wins_over_hash(tmp_path):
    save_images(tmp_path)
    df = pd.DataFrame({"path": ["a.jpg", "b.jpg", "c.jpg"], "site": "smartom", "patient_id": ["7", "8", "7"], "label": 0})
    assert add_groups(df, tmp_path).group.tolist() == ["smartom:7", "smartom:8", "smartom:7"]


def test_cross_site_duplicates_found(tmp_path):
    save_images(tmp_path)
    df = pd.DataFrame({"path": ["a.jpg", "b.jpg", "c.jpg"], "site": ["smartom", "code", "code"], "patient_id": None, "label": 0})
    pairs = cross_site_duplicates(add_groups(df, tmp_path))
    assert len(pairs) == 1 and set(pairs.iloc[0][["path_a", "path_b"]]) == {"a.jpg", "b.jpg"}


def test_a_site_split_does_not_depend_on_which_other_sites_are_loaded():
    smartom = fake_index()
    china = fake_index(60, seed=1).assign(site="china")     # sorts before smartom, so it is split first
    china["group"] = "china:" + china.group.str.split(":").str[1]
    alone = make_splits(smartom, 0).split.tolist()
    with_others = make_splits(pd.concat([smartom, china], ignore_index=True), 0)
    assert with_others[with_others.site == "smartom"].split.tolist() == alone


def test_restricted_site_rows_are_saved_to_a_separate_file(tmp_path):
    from src.splits import load_splits, save_splits
    df = pd.concat([fake_index(20), fake_index(20, seed=1).assign(site="srilanka")], ignore_index=True)
    df = make_splits(df, 0)
    save_splits(df, 0, tmp_path)
    public = pd.read_csv(tmp_path / "seed0.csv")
    assert set(public.site) == {"smartom"}                       # safe to commit
    assert (tmp_path / "seed0.restricted.csv").exists()
    back = load_splits(0, tmp_path)
    assert len(back) == len(df) and set(back.site) == {"smartom", "srilanka"}
    assert back.equals(load_splits(0, tmp_path))                 # same row order every time


def test_duplicate_group_names_do_not_depend_on_row_order(tmp_path):
    save_images(tmp_path)
    df = pd.DataFrame({"path": ["a.jpg", "b.jpg", "c.jpg"], "site": "china", "patient_id": None, "label": 0})
    forward = add_groups(df, tmp_path).set_index("path").group
    backward = add_groups(df.iloc[::-1], tmp_path).set_index("path").group
    assert forward.to_dict() == backward.to_dict()
