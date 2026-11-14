"""The two models the study compares: a linear probe on frozen features, and a fine-tuned ResNet-50."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import load_rgb

RESNET = "resnet50.a1_in1k"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet statistics the ResNet was trained with


def fit_probe(X_train, y_train, X_val, y_val):
    """Logistic regression on frozen features; regularisation strength picked on validation AUROC."""
    best, best_auc = None, -1.0
    for C in (0.01, 0.1, 1, 10):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000))
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        if auc > best_auc:
            best, best_auc = model, auc
    return best


def _transform(aug: str | None, img_size: int):
    from torchvision import transforms as T

    steps = [T.Resize((img_size, img_size))]
    if aug in ("basic", "color"):
        steps = [T.RandomResizedCrop(img_size, scale=(0.7, 1.0)), T.RandomHorizontalFlip()]
    if aug == "color":  # imitate a different phone and different lighting
        steps += [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05), T.RandomGrayscale(0.1)]
    return T.Compose(steps + [T.ToTensor(), T.Normalize(MEAN, STD)])


class Photos:
    def __init__(self, df: pd.DataFrame, root: Path, transform):
        self.paths = [Path(root) / p for p in df.path]
        self.labels = df.label.to_numpy(dtype=np.float32) if "label" in df else np.zeros(len(df), np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(load_rgb(self.paths[i])), self.labels[i]


def _scores(model, loader, device) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        return torch.cat([torch.sigmoid(model(x.to(device)).squeeze(1)).cpu() for x, _ in loader]).numpy()


def finetune(train_df, val_df, root: Path, out: Path, aug: str = "basic", seed: int = 0, epochs: int = 15,
             img_size: int = 224, lr: float = 1e-4, workers: int = 2, device: str | None = None) -> Path:
    """Fine-tune every layer; keep the epoch with the best validation AUROC."""
    import timm
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(RESNET, pretrained=True, num_classes=1).to(device)

    train = DataLoader(Photos(train_df, root, _transform(aug, img_size)), batch_size=32, shuffle=True,
                       num_workers=workers, generator=torch.Generator().manual_seed(seed))
    val = DataLoader(Photos(val_df, root, _transform(None, img_size)), batch_size=64, num_workers=workers)

    # refer cases are the minority; weight them up so the loss does not ignore them
    n_pos = float(train_df.label.sum())
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train_df) - n_pos) / n_pos, device=device))
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_auc = -1.0
    for epoch in range(epochs):
        model.train()
        for x, y in train:
            optimiser.zero_grad()
            loss_fn(model(x.to(device)).squeeze(1), y.to(device)).backward()
            optimiser.step()
        auc = roc_auc_score(val_df.label, _scores(model, val, device))
        print(f"epoch {epoch + 1}/{epochs} val AUROC {auc:.3f}")
        if auc > best_auc:
            best_auc = auc
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": model.state_dict(), "img_size": img_size}, out)
    return Path(out)


def predict(checkpoint: Path, df, root: Path, workers: int = 2, device: str | None = None) -> np.ndarray:
    """Refer probability for each row of `df`, in row order."""
    import timm
    import torch
    from torch.utils.data import DataLoader

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint, map_location=device)
    model = timm.create_model(RESNET, pretrained=False, num_classes=1).to(device)
    model.load_state_dict(saved["state"])
    loader = DataLoader(Photos(df, root, _transform(None, saved["img_size"])), batch_size=64, num_workers=workers)
    return _scores(model, loader, device)
