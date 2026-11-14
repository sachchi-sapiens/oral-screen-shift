"""Load photos the same way everywhere, and turn them into frozen-model features once."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


def load_rgb(path: Path) -> Image.Image:
    """Phone photos carry a rotation flag and come in several colour modes; return upright RGB."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def to_tensor(image: Image.Image, img_size: int, mean, std):
    import torch

    pixels = np.asarray(image.resize((img_size, img_size), Image.BICUBIC), dtype=np.float32) / 255
    return torch.from_numpy((pixels - np.float32(mean)) / np.float32(std)).permute(2, 0, 1)


def extract(df: pd.DataFrame, root: Path, out: Path, model_name: str = "vit_base_patch14_dinov2.lvd142m",
            img_size: int = 224, batch_size: int = 64, device: str | None = None) -> np.ndarray:
    """One feature row per row of `df`, in the same order. Reuses `out` if it was made from the same paths."""
    out, paths_file = Path(out), Path(str(out) + ".paths.csv")
    if out.exists() and paths_file.exists() and pd.read_csv(paths_file).path.tolist() == df.path.tolist():
        return np.load(out)

    import timm
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=img_size).eval().to(device)
    mean, std = model.pretrained_cfg["mean"], model.pretrained_cfg["std"]

    rows = []
    paths = df.path.tolist()
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = torch.stack([to_tensor(load_rgb(Path(root) / p), img_size, mean, std)
                                 for p in paths[start:start + batch_size]])
            rows.append(model(batch.to(device)).float().cpu().numpy())
            print(f"features {min(start + batch_size, len(paths))}/{len(paths)}", end="\r")
    features = np.concatenate(rows).astype(np.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, features)
    df[["path"]].to_csv(paths_file, index=False)
    return features
