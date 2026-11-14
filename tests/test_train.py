import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.train import fit_probe


def test_probe_learns_separable_data():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(100), np.ones(100)].astype(int)
    X = rng.normal(size=(200, 8))
    X[:, 0] += 3 * y
    m = fit_probe(X[::2], y[::2], X[1::2], y[1::2])
    assert roc_auc_score(y[1::2], m.predict_proba(X[1::2])[:, 1]) > 0.95


@pytest.fixture
def red_blue(tmp_path, monkeypatch):
    """8 red photos labelled refer, 8 blue labelled do-not-refer, and a tiny network in place of ResNet."""
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")

    def tiny(*args, **kwargs):
        return torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.AdaptiveAvgPool2d(1),
                                   torch.nn.Flatten(), torch.nn.Linear(4, 1))

    monkeypatch.setattr(timm, "create_model", tiny)
    rows = []
    for i in range(8):
        for colour, label in [((220, 20, 20), 1), ((20, 20, 220), 0)]:
            name = f"{label}_{i}.png"
            Image.new("RGB", (40, 40), colour).save(tmp_path / name)
            rows.append({"path": name, "label": label})
    return pd.DataFrame(rows), tmp_path


def run(df, root, name):
    from src.train import finetune, predict
    ckpt = finetune(df, df, root, root / name, epochs=20, img_size=32, lr=0.05, workers=0, device="cpu")
    return predict(ckpt, df, root, workers=0, device="cpu")


def test_finetune_separates_and_predict_keeps_row_order(red_blue):
    df, root = red_blue
    p = run(df, root, "m.pt")
    assert p.shape == (16,) and ((p >= 0) & (p <= 1)).all()
    assert p[df.label == 1].min() > p[df.label == 0].max()


def test_same_seed_same_predictions(red_blue):
    df, root = red_blue
    assert np.allclose(run(df, root, "a.pt"), run(df, root, "b.pt"), atol=1e-5)
