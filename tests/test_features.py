import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.features import extract, load_rgb


@pytest.mark.parametrize("mode", ["L", "CMYK", "RGBA", "P"])
def test_load_rgb_converts_modes(tmp_path, mode):
    p = tmp_path / ("a.jpg" if mode == "CMYK" else "a.png")
    Image.new(mode, (10, 20)).save(p)
    assert load_rgb(p).mode == "RGB"


def test_load_rgb_applies_exif_rotation(tmp_path):
    im = Image.new("RGB", (10, 20))
    exif = im.getexif()
    exif[0x0112] = 6  # "rotate 90 degrees to view"
    im.save(tmp_path / "r.jpg", exif=exif)
    assert load_rgb(tmp_path / "r.jpg").size == (20, 10)


def test_extract_shape_order_and_cache(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")

    class ChannelMean(torch.nn.Module):
        pretrained_cfg = {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}

        def forward(self, x):
            return x.mean(dim=(2, 3))

    monkeypatch.setattr(timm, "create_model", lambda *a, **k: ChannelMean())
    for name, grey in [("black.png", 0), ("grey.png", 128), ("white.png", 255)]:
        Image.new("RGB", (40, 40), (grey,) * 3).save(tmp_path / name)
    df = pd.DataFrame({"path": ["black.png", "grey.png", "white.png"]})

    f = extract(df, tmp_path, tmp_path / "f.npy", img_size=32, batch_size=2, device="cpu")
    assert f.shape == (3, 3) and f.dtype == np.float32
    assert f[0, 0] < f[1, 0] < f[2, 0]

    monkeypatch.setattr(timm, "create_model", lambda *a, **k: 1 / 0)  # a second call must not build a model
    assert np.array_equal(extract(df, tmp_path, tmp_path / "f.npy", img_size=32), f)

    with pytest.raises(ZeroDivisionError):  # a different image list must not be served from the cache
        extract(df.iloc[:2], tmp_path, tmp_path / "f.npy", img_size=32)
