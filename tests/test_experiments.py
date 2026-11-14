import numpy as np
import pandas as pd
import pytest

from experiments import e1_matrix, e3_refusal
from experiments.common import pred_path


def fake_run(tmp_path, cross="random", n_groups=400):
    """Two sites, three seeds. In-site scores are calibrated; cross-site scores are `cross`."""
    results, splits_dir = tmp_path / "results", tmp_path / "splits"
    (results / "preds").mkdir(parents=True)
    splits_dir.mkdir()
    for seed in range(3):
        rng = np.random.default_rng(seed)
        rows = []
        for site in ("smartom", "china"):
            for g in range(n_groups):
                split = "train" if g % 4 < 2 else ("val" if g % 4 == 2 else "test")
                rows += [{"path": f"{site}/{g}_{i}.jpg", "site": site, "group": f"{site}:{g}", "split": split}
                         for i in range(2)]
        df = pd.DataFrame(rows)
        df["quality"] = rng.uniform(size=len(df))                 # a hidden "true" refer probability
        df["label"] = (rng.uniform(size=len(df)) < df.quality).astype(int)
        df.drop(columns="quality").to_csv(splits_dir / f"seed{seed}.csv", index=False)
        for model in ("probe", "resnet", "resnet_color"):
            for train in ("smartom", "china"):
                for test in ("smartom", "china"):
                    for split in ("val", "test"):
                        rows_ = df[(df.site == test) & (df.split == split)]
                        if train == test:
                            p = rows_.quality.to_numpy()
                        elif cross == "random":
                            p = rng.uniform(size=len(rows_))
                        else:  # overconfident: right direction, but far too sure
                            p = np.where(rows_.quality > 0.5, 0.98, 0.02)
                        np.save(pred_path(results, model, train, seed, test, split), p)
    return results, splits_dir


def test_e1_matrix_rows_and_drop(tmp_path):
    results, splits_dir = fake_run(tmp_path)
    table = e1_matrix.main(results, splits_dir)
    assert len(table) == 2 * 2 * 2 and set(table.model) == {"probe", "resnet"}
    same = table.train == table.test
    assert (table[same].auroc > 0.8).all() and table[~same].auroc.between(0.4, 0.6).all()
    assert (table.auroc_lo <= table.auroc).all() and (table.auroc <= table.auroc_hi).all()
    assert (results / "e1_matrix.csv").exists() and (results / "e1_matrix.png").exists()


def test_threshold_never_uses_test_site_labels(tmp_path):
    results, splits_dir = fake_run(tmp_path)
    before = e1_matrix.main(results, splits_dir)
    for seed in range(3):
        f = splits_dir / f"seed{seed}.csv"
        df = pd.read_csv(f)
        df.loc[df.site == "china", "label"] = 1 - df.loc[df.site == "china", "label"]
        df.to_csv(f, index=False)
    after = e1_matrix.main(results, splits_dir)
    cell = lambda t: t[(t.train == "smartom") & (t.test == "china")].threshold.to_numpy()
    assert np.array_equal(cell(before), cell(after))


def test_e3_promise_breaks_cross_site_and_recalibration_restores_it(tmp_path):
    results, splits_dir = fake_run(tmp_path, cross="overconfident")
    t = e3_refusal.main(results, splits_dir)
    row = lambda train, test, n: t[(t.model == "probe") & (t.train == train) & (t.test == test) & (t.recal_n == n)].iloc[0]
    assert row("smartom", "smartom", 0).coverage >= 0.87
    assert row("smartom", "china", 0).coverage < 0.80
    assert row("smartom", "china", 100).coverage >= 0.85
    assert row("smartom", "china", 100).n <= row("smartom", "china", 0).n - 100   # calibration photos are not scored
    assert (results / "e3_refusal.csv").exists() and (results / "e3_refusal.png").exists()


def test_e2_one_row_per_available_method_and_site(tmp_path):
    from experiments import e2_fixes
    results, splits_dir = fake_run(tmp_path)          # no leave-one-out predictions exist
    t = e2_fixes.main(results, splits_dir)
    assert set(t.test) == {"smartom", "china"} and len(t) == 2 * 3
    assert not t.method.str.contains("two sites").any()
    assert t.auroc.between(0.4, 0.6).all()            # every row is a cross-site number
    assert (results / "e2_fixes.png").exists()
