"""Research demo: upload a mouth photo, get "refer", "do not refer" or "not sure".

    streamlit run app/demo.py

Needs the feature files made by `python -m experiments.run_gpu --stage features`.
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.common import MAIN
from src.conformal import calibrate
from src.features import load_rgb, to_tensor
from src.splits import load_splits
from src.train import fit_probe

MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"

st.set_page_config(page_title="Oral screen shift demo")
st.title("Oral lesion screening: research demo")
st.warning(
    "This is a research demo, not a medical device. It was trained on about 2,700 photos from two "
    "datasets (SMART-OM, India, and one from China) and its accuracy drops on photos from elsewhere. "
    "It cannot rule out cancer. See a dentist or doctor about any mouth sore, patch or lump that "
    "lasts more than two weeks."
)


@st.cache_resource
def load():
    """Fit the probe on both training sites, and set its threshold and refusal rule on their validation data."""
    import timm

    splits = load_splits(0, ROOT / "splits")
    X, y, split = [], [], []
    for site in MAIN:
        rows = splits[splits.site == site]
        X.append(np.load(ROOT / "features" / f"{site}.npy"))
        y.append(rows.label.to_numpy())
        split.append(rows.split.to_numpy())
    X, y, split = np.concatenate(X), np.concatenate(y), np.concatenate(split)
    train, val = split == "train", split == "val"
    probe = fit_probe(X[train], y[train], X[val], y[val])
    qhat = calibrate(probe.predict_proba(X[val])[:, 1], y[val])
    backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=224).eval()
    return probe, qhat, backbone


uploaded = st.file_uploader("Photo of the inside of the mouth", type=["jpg", "jpeg", "png"])
if uploaded:
    import torch

    probe, qhat, backbone = load()
    image = load_rgb(uploaded)
    image.thumbnail((1024, 1024))
    st.image(image, width=360)

    cfg = backbone.pretrained_cfg
    with torch.no_grad():
        features = backbone(to_tensor(image, 224, cfg["mean"], cfg["std"])[None]).numpy()
    p = float(probe.predict_proba(features)[0, 1])

    # the same rule as experiment 3: answer only when exactly one class is plausible
    refer_plausible, clear_plausible = 1 - p <= qhat, p <= qhat
    if refer_plausible and not clear_plausible:
        st.error(f"Refer: this photo looks like the pre-cancer or cancer photos the model saw (score {p:.2f}).")
    elif clear_plausible and not refer_plausible:
        st.success(f"Do not refer: this photo looks like the healthy photos the model saw (score {p:.2f}).")
    else:
        st.info(f"Not sure, so refer: the model cannot tell for this photo (score {p:.2f}).")
    st.caption("The model has no way to know whether a photo shows a mouth at all. A photo of anything else still gets a score.")
