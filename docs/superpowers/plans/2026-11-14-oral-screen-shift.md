# Oral Screen Shift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how much an oral-lesion "refer / do not refer" classifier loses when moved between four public photo datasets, test three fixes, and test whether a conformal refusal rule keeps its promise at a new site.

**Architecture:** One index table (a pandas DataFrame, one row per image) flows through everything. GPU steps (feature extraction, fine-tuning) run in Colab and write predictions and features to disk as `.npy`; every metric, table and figure is computed from those files on CPU. Only `features.py` and `train.py` import torch, so the rest of the code and its tests run without it.

**Tech Stack:** Python 3.11+ (local machine has 3.14.8, no GPU, no pip outside a venv), PyTorch, timm, scikit-learn, pandas, numpy, Pillow, matplotlib, pytest, Streamlit.

**Spec:** `docs/superpowers/specs/2026-11-14-oral-cancer-shift-design.md`

## Global Constraints

- No patient (or near-duplicate group) may appear in more than one of train/val/test.
- Thresholds, hyperparameters and conformal quantiles are chosen on the training site's validation split only. Test-site labels are used only in E3 recalibration, and only the stated 25/50/100 images.
- Splits are 70/15/15 by group, seeds 0, 1, 2, saved to `splits/`.
- 95% intervals by bootstrap over groups, 1000 resamples.
- Label: `1` = refer (OPMD or cancer), `0` = do not refer (normal, variation from normal, benign).
- No images, features or checkpoints in git. `data/`, `features/`, `checkpoints/` are git-ignored except `data/README.md`.
- Every task that creates a code file adds that file's entry to `WHY.md` in the same commit: why it exists, what would break without it.
- No LLM calls. No experiment-tracking service. No dependency for something that is under about 20 lines.
- Report every experiment that was run, including ones that went against expectation.
- Commit messages end with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.
- Site names used everywhere: `smartom`, `code`, `china`, `karnataka`.

## Review Focus

1. A file in a dataset folder that is not a readable image (thumbnail db, truncated jpeg): the index build skips it and prints how many were skipped; it does not crash. (Task 2)
2. A test set with only one class present: AUROC is undefined; `metrics` returns `nan` for it and still returns the other numbers. (Task 4)
3. A patient with both refer and do-not-refer images: the split stratifies on the patient's highest label and all of that patient's images stay together. (Task 3)
4. Images that are grayscale, CMYK, have an alpha channel or an EXIF rotation: loading returns upright 3-channel RGB. (Task 6)
5. A conformal calibration set smaller than `1/alpha`: the quantile is infinite, every prediction set contains both classes (always refer), and nothing crashes. (Task 5)

---

### Task 1: Project skeleton and label mapping

**Files:**
- Create: `.gitignore`, `requirements.txt`, `WHY.md`, `src/__init__.py`, `src/datasets.py`, `tests/test_datasets.py`

**Interfaces:**
- Produces: `src.datasets.REFER: dict[str, dict[str, int]]` (site → lower-cased original class name → 0/1) and `refer_label(site: str, orig_class: str) -> int` (raises `KeyError` naming the site and class when unmapped).

- [ ] **Step 1: Create the environment**

```bash
python -m venv .venv
```
```bash
.venv/Scripts/python -m pip install numpy pandas scikit-learn pillow matplotlib pytest
```
```bash
.venv/Scripts/python -m pip install torch torchvision timm --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```
If the torch install fails on Python 3.14, record that in `README.md` under "Local setup", continue, and run the torch-dependent tests of Tasks 6 and 7 in Colab only. Write `requirements.txt` with the packages above plus `streamlit`, unpinned except any version the install forced.

- [ ] **Step 2: Write the failing tests** in `tests/test_datasets.py`

```python
import pytest
from src.datasets import refer_label

@pytest.mark.parametrize("site,cls,expected", [
    ("smartom", "normal", 0), ("smartom", "variations from normal", 0),
    ("smartom", "opmd", 1), ("smartom", "oc", 1),
    ("code", "normal mucosa", 0), ("code", "variations from normal", 0),
    ("code", "opmd", 1), ("code", "oral cancer", 1),
    ("china", "normal", 0), ("china", "olk", 1), ("china", "olp", 1),
    ("china", "osf", 1), ("china", "oca", 1),
    ("karnataka", "benign", 0), ("karnataka", "malignant", 1),
])
def test_refer_label(site, cls, expected):
    assert refer_label(site, cls) == expected

def test_case_and_spaces_ignored():
    assert refer_label("china", " OLK ") == 1

def test_unknown_class_names_site_and_class():
    with pytest.raises(KeyError, match="china.*ulcer"):
        refer_label("china", "ulcer")
```

- [ ] **Step 3: Run** `.venv/Scripts/python -m pytest tests/test_datasets.py -q` — expected: import error.
- [ ] **Step 4: Implement `REFER` and `refer_label`** in `src/datasets.py`. `.gitignore`: `.venv/`, `data/*`, `!data/README.md`, `features/`, `checkpoints/`, `__pycache__/`, `.pytest_cache/`.
- [ ] **Step 5: Run the tests** — expected: 17 passed.
- [ ] **Step 6: Start `WHY.md`** with a one-paragraph purpose and the entry for `src/datasets.py`. Commit.

### Task 2: Download and index the four datasets

**Files:**
- Create: `scripts/download.py`, `data/README.md`
- Modify: `src/datasets.py`, `tests/test_datasets.py`, `WHY.md`

**Interfaces:**
- Consumes: `refer_label`.
- Produces: `build_index(site: str, root: Path) -> pd.DataFrame` with columns `path` (str, relative to `root`), `site`, `patient_id` (str or `None`), `orig_class` (str), `label` (int). `load_index(data_root: Path, sites: list[str]) -> pd.DataFrame` concatenates them and caches to `data_root/index.csv`.
- Produces: `python scripts/download.py --site <name> --out <dir>` leaving `<dir>/<name>/` unpacked.

Sources: `smartom` figshare article 31341790; `code` figshare article 30550889, file `..._Version_5.zip` only; `china` `https://huggingface.co/datasets/HRruiH/Dataset-of-oral-mucosal-diseases/resolve/main/mouth_data.zip`; `karnataka` Mendeley dataset `mhjyrn35p4`, `original_data` folder only (never `augmented_data`: augmented copies of test images in training would leak).

- [ ] **Step 1: Write `scripts/download.py`** using `urllib` and `zipfile` only. figshare file URLs come from `https://api.figshare.com/v2/articles/<id>`. Skip a site whose folder already exists.
- [ ] **Step 2: Download `karnataka`, `china`, `smartom` locally** to `data/` (about 3 GB; the machine has 29 GB free). `code` (6.6 GB zipped) is downloaded in Colab only, in Task 8.
- [ ] **Step 3: Inspect each unpacked layout** and write `data/README.md`: source, licence, citation, the folder layout actually found, where class and patient id come from (folder name, file name, or metadata sheet), image count per original class. For `code`, read the zip's file list without downloading the body (HTTP range request for the central directory) and record the same. If a class name differs from the keys in `REFER`, fix `REFER` and its test to the real names.
- [ ] **Step 4: Write the failing tests** using a fake tree built in `tmp_path` that copies each real layout with 1×1 images:

```python
def test_build_index_columns_and_labels(fake_china):
    df = build_index("china", fake_china)
    assert list(df.columns) == ["path", "site", "patient_id", "orig_class", "label"]
    assert set(df.label) == {0, 1} and (df.site == "china").all()

def test_karnataka_ignores_augmented(fake_karnataka):
    assert not build_index("karnataka", fake_karnataka).path.str.contains("augmented").any()

def test_smartom_patient_id_from_filename(fake_smartom):
    # A_B_C.jpeg: A is the patient id
    assert build_index("smartom", fake_smartom).patient_id.notna().all()

def test_unreadable_file_skipped(fake_china, capsys):
    (fake_china / "1" / "Thumbs.db").write_bytes(b"junk")
    n = len(build_index("china", fake_china))
    assert "skipped 1" in capsys.readouterr().out and n == 4
```

- [ ] **Step 5: Run** — expected: fail, `build_index` not defined.
- [ ] **Step 6: Implement `build_index` and `load_index`.** One small private function per site. Readability check is `PIL.Image.open(p).verify()`.
- [ ] **Step 7: Run the tests**, then run `load_index` on the three real datasets and paste the class counts per site into `data/README.md`. Commit.

### Task 3: Groups, overlap check and splits

**Files:**
- Create: `src/splits.py`, `tests/test_splits.py`
- Modify: `WHY.md`

**Interfaces:**
- Consumes: the index DataFrame.
- Produces: `dhash(path: Path) -> int` (64-bit difference hash: grayscale, resize to 9×8, compare horizontal neighbours).
- Produces: `add_groups(df, root: Path, max_bits: int = 4) -> df` adding `dhash` and `group` columns. `group` = `site:patient_id` when the id exists; otherwise images of the same site whose hashes differ by ≤ `max_bits` share a group (union-find).
- Produces: `cross_site_duplicates(df, max_bits: int = 4) -> pd.DataFrame` of image pairs from different sites within `max_bits`.
- Produces: `make_splits(df, seed: int) -> df` adding `split` in `{"train","val","test"}`, 70/15/15 by group within each site, stratified on each group's highest label. `save_splits` / `load_splits` write and read `splits/seed<seed>.csv` (columns `path, site, split`).

- [ ] **Step 1: Write the failing tests**

```python
def test_no_group_in_two_splits(index_with_groups):
    s = make_splits(index_with_groups, seed=0)
    assert (s.groupby("group").split.nunique() == 1).all()

def test_fractions_roughly_70_15_15(index_with_groups):   # 200 groups
    f = make_splits(index_with_groups, seed=0).drop_duplicates("group").split.value_counts(normalize=True)
    assert abs(f["train"] - 0.70) < 0.05 and abs(f["val"] - 0.15) < 0.05

def test_same_seed_same_split(index_with_groups):
    assert make_splits(index_with_groups, 0).split.equals(make_splits(index_with_groups, 0).split)

def test_mixed_label_patient_stays_together(mixed_patient_index):
    s = make_splits(mixed_patient_index, seed=0)
    assert s[s.group == "smartom:P1"].split.nunique() == 1

def test_each_split_has_both_classes(index_with_groups):
    assert (make_splits(index_with_groups, 0).groupby("split").label.nunique() == 2).all()

def test_near_duplicates_share_group(tmp_path):
    # same gradient image saved twice at different jpeg quality, plus one different image
    df = add_groups(three_image_index(tmp_path), tmp_path)
    assert df.group[0] == df.group[1] != df.group[2]

def test_cross_site_duplicates_found(tmp_path):
    assert len(cross_site_duplicates(two_sites_same_image(tmp_path))) == 1
```

- [ ] **Step 2: Run** — expected: fail on import.
- [ ] **Step 3: Implement.** Stratified group split: `sklearn.model_selection.StratifiedGroupKFold` is not a 70/15/15 splitter, so shuffle groups within each stratum with `numpy.random.default_rng(seed)` and cut at 70% and 85%.
- [ ] **Step 4: Run the tests** — expected: pass.
- [ ] **Step 5: Run on the real local data.** Save splits for seeds 0, 1, 2. Record in `data/README.md`: groups per site, how many images were merged as near-duplicates, and the `smartom`/`china`/`karnataka` cross-site duplicate count. (The `smartom`-vs-`code` check runs in Task 8 once `code` is downloaded; any pair found is dropped from `code`.) Commit.

### Task 4: Metrics and bootstrap intervals

**Files:**
- Create: `src/evaluate.py`, `tests/test_evaluate.py`
- Modify: `WHY.md`

**Interfaces:**
- Produces: `pick_threshold(y_val, p_val, min_sensitivity: float = 0.90) -> float`: the highest threshold whose validation sensitivity is ≥ `min_sensitivity` (a screening tool is tuned to miss few).
- Produces: `metrics(y, p, threshold: float) -> dict` with keys `auroc`, `sensitivity`, `specificity`, `ece` (10 equal-width bins), `n`, `n_refer`.
- Produces: `bootstrap_ci(y, p, groups, threshold, n_boot: int = 1000, seed: int = 0) -> dict[str, tuple[float, float]]` resampling groups with replacement, 2.5 and 97.5 percentiles ignoring `nan`.

- [ ] **Step 1: Write the failing tests**

```python
def test_metrics_hand_computed():
    y = np.array([1, 1, 0, 0]); p = np.array([0.9, 0.4, 0.6, 0.1])
    m = metrics(y, p, threshold=0.5)
    assert m["auroc"] == 0.75 and m["sensitivity"] == 0.5 and m["specificity"] == 0.5
    assert m["n"] == 4 and m["n_refer"] == 2

def test_ece_zero_when_perfectly_calibrated():
    y = np.array([1] * 8 + [0] * 2); p = np.full(10, 0.8)
    assert metrics(y, p, 0.5)["ece"] == pytest.approx(0.0)

def test_single_class_gives_nan_auroc_not_crash():
    m = metrics(np.zeros(5, int), np.linspace(0, 1, 5), 0.5)
    assert np.isnan(m["auroc"]) and np.isnan(m["sensitivity"]) and m["specificity"] == 0.6

def test_pick_threshold_meets_sensitivity():
    y = np.array([1] * 10 + [0] * 10); p = np.r_[np.linspace(0.3, 0.95, 10), np.linspace(0.0, 0.6, 10)]
    t = pick_threshold(y, p, 0.90)
    assert ((p >= t) & (y == 1)).sum() / 10 >= 0.90
    assert ((p >= t + 0.08) & (y == 1)).sum() / 10 < 0.90   # not lower than it needs to be

def test_bootstrap_resamples_groups_not_rows():
    # every group is internally identical, so with groups the interval is wide,
    # and a seed gives the same interval twice
    ci = bootstrap_ci(y, p, groups, 0.5, n_boot=200, seed=1)
    assert ci["auroc"][0] <= metrics(y, p, 0.5)["auroc"] <= ci["auroc"][1]
    assert ci == bootstrap_ci(y, p, groups, 0.5, n_boot=200, seed=1)
```

- [ ] **Step 2: Run** — fail. **Step 3: Implement** with numpy and `sklearn.metrics.roc_auc_score`. **Step 4: Run** — pass. **Step 5: Commit.**

### Task 5: Conformal refusal

**Files:**
- Create: `src/conformal.py`, `tests/test_conformal.py`
- Modify: `WHY.md`

**Interfaces:**
- Produces: `calibrate(p_cal, y_cal, alpha: float = 0.10) -> float`: split-conformal quantile of the scores `1 - (probability given to the true class)`, at level `ceil((n+1)(1-alpha))/n`; returns `inf` when that level exceeds 1.
- Produces: `prediction_sets(p, qhat: float) -> np.ndarray` of shape `(n, 2)`, bool; column `k` is true when `1 - p_k <= qhat`.
- Produces: `summarize(sets, y) -> dict` with `coverage` (true class is in the set), `answered` (fraction of sets with exactly one class), `error_when_answered`, `missed_refer` (fraction of true refer cases answered "do not refer").

An image is *answered* when its set has exactly one class; otherwise the tool says "not sure, refer".

- [ ] **Step 1: Write the failing tests**

```python
def test_coverage_holds_on_exchangeable_data():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=4000); y = (rng.uniform(size=4000) < p).astype(int)
    q = calibrate(p[:2000], y[:2000], alpha=0.10)
    cov = summarize(prediction_sets(p[2000:], q), y[2000:])["coverage"]
    assert 0.88 <= cov <= 0.93

def test_tiny_calibration_set_refers_everything():
    q = calibrate(np.array([0.9, 0.2, 0.7]), np.array([1, 0, 1]), alpha=0.10)   # 3 < 1/alpha
    s = prediction_sets(np.array([0.99, 0.01]), q)
    assert np.isinf(q) and s.all() and summarize(s, np.array([1, 0]))["answered"] == 0.0

def test_confident_predictions_are_answered():
    s = prediction_sets(np.array([0.99, 0.01, 0.5]), qhat=0.2)
    assert s.tolist() == [[False, True], [True, False], [False, False]]

def test_missed_refer_counts_confident_wrong_negatives():
    s = np.array([[True, False], [False, True]]); y = np.array([1, 1])
    assert summarize(s, y)["missed_refer"] == 0.5
```

- [ ] **Step 2: Run** — fail. **Step 3: Implement** (about 20 lines, numpy only). **Step 4: Run** — pass. **Step 5: Commit.**

### Task 6: Image loading and frozen features

**Files:**
- Create: `src/features.py`, `tests/test_features.py`
- Modify: `WHY.md`

**Interfaces:**
- Produces: `load_rgb(path: Path) -> PIL.Image` — EXIF-transposed, converted to RGB.
- Produces: `extract(df, root: Path, out: Path, model_name: str = "vit_base_patch14_dinov2.lvd142m", img_size: int = 224, batch_size: int = 64, device: str | None = None) -> np.ndarray` — float32 array `(len(df), dim)` in `df` row order, saved to `out` with `df.path` saved beside it as `<out>.paths.csv`. Uses `timm.create_model(model_name, pretrained=True, num_classes=0, img_size=img_size)` and timm's own data config for normalisation. If `out` exists and its paths file matches `df.path`, it is loaded instead of recomputed.

- [ ] **Step 1: Write the failing tests** (torch-free tests always run; the torch one is skipped when torch is missing)

```python
@pytest.mark.parametrize("mode", ["L", "CMYK", "RGBA", "P"])
def test_load_rgb_converts_modes(tmp_path, mode):
    p = tmp_path / "a.png" if mode != "CMYK" else tmp_path / "a.jpg"
    Image.new(mode, (10, 20)).save(p)
    assert load_rgb(p).mode == "RGB"

def test_load_rgb_applies_exif_rotation(tmp_path):
    im = Image.new("RGB", (10, 20)); exif = im.getexif(); exif[0x0112] = 6
    im.save(tmp_path / "r.jpg", exif=exif)
    assert load_rgb(tmp_path / "r.jpg").size == (20, 10)

def test_extract_shape_order_and_cache(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    # monkeypatch timm.create_model to a tiny module returning the image mean per channel (dim 3)
    f = extract(df3, tmp_path, tmp_path / "f.npy", img_size=32, batch_size=2, device="cpu")
    assert f.shape == (3, 3) and f.dtype == np.float32
    assert f[0, 0] < f[1, 0] < f[2, 0]            # df3 images are black, grey, white, in that order
    monkeypatch.setattr("timm.create_model", lambda *a, **k: 1 / 0)
    assert np.array_equal(extract(df3, tmp_path, tmp_path / "f.npy", img_size=32), f)   # served from cache
```

- [ ] **Step 2: Run** — fail. **Step 3: Implement.** Import torch and timm inside `extract`, not at module top, so `load_rgb` works without them. **Step 4: Run** — pass. **Step 5: Commit.**

### Task 7: Training the two models

**Files:**
- Create: `src/train.py`, `tests/test_train.py`
- Modify: `WHY.md`

**Interfaces:**
- Consumes: `load_rgb`, features arrays, split DataFrames.
- Produces: `fit_probe(X_train, y_train, X_val, y_val) -> sklearn Pipeline` — `StandardScaler` + `LogisticRegression(class_weight="balanced", max_iter=2000)`, `C` chosen from `[0.01, 0.1, 1, 10]` by validation AUROC. Predict with `.predict_proba(X)[:, 1]`.
- Produces: `finetune(train_df, val_df, root: Path, out: Path, aug: str = "basic", seed: int = 0, epochs: int = 15, device: str | None = None) -> Path` — `timm` `resnet50.a1_in1k`, one output logit, 224 px, AdamW lr 1e-4, weight decay 1e-4, batch 32, `BCEWithLogitsLoss` with `pos_weight = n_neg / n_pos`, keeps the epoch with the best validation AUROC. `aug="basic"`: random resized crop (scale 0.7–1.0) + horizontal flip. `aug="color"`: basic + `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05)` + random grayscale 0.1.
- Produces: `predict(checkpoint: Path, df, root: Path, device: str | None = None) -> np.ndarray` — refer probabilities in `df` row order (centre resize, no augmentation).

- [ ] **Step 1: Write the failing tests**

```python
def test_probe_learns_separable_data():
    rng = np.random.default_rng(0); y = np.r_[np.zeros(100), np.ones(100)].astype(int)
    X = rng.normal(size=(200, 8)); X[:, 0] += 3 * y
    m = fit_probe(X[::2], y[::2], X[1::2], y[1::2])
    assert roc_auc_score(y[1::2], m.predict_proba(X[1::2])[:, 1]) > 0.95

def test_finetune_overfits_tiny_set_and_predict_is_ordered(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    # 8 red images labelled 1, 8 blue labelled 0, 32 px; monkeypatch timm.create_model to a 2-layer convnet
    ckpt = finetune(df16, df16, tmp_path, tmp_path / "m.pt", epochs=20, device="cpu")
    p = predict(ckpt, df16, tmp_path, device="cpu")
    assert p.shape == (16,) and p[df16.label == 1].min() > p[df16.label == 0].max()

def test_same_seed_same_predictions(...):   # two finetune runs, seed=0, np.allclose(p1, p2, atol=1e-5)
```

- [ ] **Step 2: Run** — fail. **Step 3: Implement.** Seed `torch`, `numpy` and the DataLoader generator from `seed`. **Step 4: Run** — pass (or skipped locally if torch is unavailable; then they must pass in Colab in Task 8). **Step 5: Commit.**

### Task 8: Colab run — features, fine-tuning, cached predictions

**Files:**
- Create: `experiments/run_gpu.py`, `notebooks/colab_gpu.ipynb`
- Modify: `WHY.md`, `data/README.md`

**Interfaces:**
- Produces prediction files that Tasks 9–11 read: `results/preds/<model>__<train>__seed<s>__<test>__<split>.npy` (refer probabilities, row order = that site's split rows in `splits/seed<s>.csv`). `<model>` ∈ `probe`, `resnet`, `resnet_color`. `<train>` ∈ `smartom`, `code`, `china`, or leave-one-out names `all-but-smartom`, `all-but-code`, `all-but-china`. `<test>` ∈ the four sites. `<split>` ∈ `val`, `test`.
- Produces: `features/<site>.npy`.

`run_gpu.py --data <dir> --stage features|probe|resnet|resnet_color|loo` does, for seeds 0–2: extract features for all four sites; fit the probe per training site; fine-tune ResNet per training site with `aug="basic"` and `aug="color"`; for leave-one-out, train probe and `resnet_color` on the pooled train splits of the other two main sites. After each model, write predictions for every site's `val` and `test` rows. `karnataka` is never a training site; all its images are scored as `test`. A stage skips any output file that already exists, so a disconnected Colab session resumes.

- [ ] **Step 1: Write `run_gpu.py`.** It only loops and calls Tasks 2, 3, 6, 7 functions.
- [ ] **Step 2: Write the notebook**: mount Drive, clone the repository, `pip install -r requirements.txt`, run `pytest` (the torch tests must pass here), `scripts/download.py` for all four sites into Drive, build the index with `code` included, run `cross_site_duplicates` for `smartom` vs `code` and drop pairs from `code`, regenerate and commit `splits/`, then one cell per stage. Each cell is a single shell command.
- [ ] **Step 3: Dry run locally on CPU** with `--stage probe --limit 40` (add `--limit N`: first N images per site) on `karnataka` and `china` to prove the file naming and resume logic. Expected: files appear under `results/preds/`; a second run prints "skipped" for each.
- [ ] **Step 4: Hand the notebook to Sachchidananda to run in Colab** (his Google account and Drive). Expected time on a free T4: features about 15 minutes; fine-tuning 18 models about 3–4 hours, resumable. He copies `results/preds/` and `features/` back from Drive (a few hundred MB).
- [ ] **Step 5: Record in `data/README.md`** the `code` class counts and the `smartom`/`code` overlap result. Commit `splits/`, the script and the notebook. `results/preds/` is committed (small `.npy` files, no images) so tables can be regenerated by anyone.

### Task 9: E1 the train/test matrix and E2 fixes

**Files:**
- Create: `experiments/e1_matrix.py`, `experiments/e2_fixes.py`, `tests/test_experiments.py`
- Modify: `WHY.md`

**Interfaces:**
- Consumes: prediction files (Task 8 naming), `splits/`, `pick_threshold`, `metrics`, `bootstrap_ci`.
- Produces: `results/e1_matrix.csv` — one row per (`model`, `train`, `test`) with each metric's mean over seeds and the bootstrap interval of seed 0; `results/e1_matrix.png` — two heatmaps of AUROC (probe, resnet), rows train site, columns test site. Only `probe` and `resnet` appear in E1.
- Produces: `results/e2_fixes.csv` and `.png` — for each held-out test site, cross-site AUROC and sensitivity for: `resnet` (single source, averaged over sources), `resnet_color` (same), `probe` (same), `resnet_color` leave-one-out, `probe` leave-one-out.
- Produces: `load_preds(model, train, seed, test, split) -> (y, p, groups)` in `experiments/common.py`, shared by Tasks 9–11.

The threshold for a cell is `pick_threshold` on the **training** site's `val` predictions (for leave-one-out: pooled `val` of the training sites).

- [ ] **Step 1: Write the failing test** — build fake prediction files for two sites in `tmp_path` where the in-site scores are informative and cross-site scores are random; run `e1_matrix.main(results_dir)`; assert the CSV has 2 models × 2 train × 2 test rows, diagonal AUROC > 0.9, off-diagonal between 0.3 and 0.7, and that changing the **test** site's labels does not change the threshold column.
- [ ] **Step 2: Run** — fail. **Step 3: Implement** both scripts and `common.py`. **Step 4: Run** — pass.
- [ ] **Step 5: Run on the real predictions.** Commit scripts, CSVs and figures.

### Task 10: E3 refusal under shift

**Files:**
- Create: `experiments/e3_refusal.py`
- Modify: `tests/test_experiments.py`, `WHY.md`

**Interfaces:**
- Consumes: `load_preds`, `calibrate`, `prediction_sets`, `summarize`.
- Produces: `results/e3_refusal.csv` — rows per (`model`, `train`, `test`, `recal_n`) with `coverage`, `answered`, `error_when_answered`, `missed_refer`, mean over seeds. `recal_n = 0` means the quantile comes from the training site's `val` split; `recal_n` ∈ {25, 50, 100} means it comes from that many test-site images drawn at random by group (20 draws, averaged), which are then excluded from scoring. `alpha = 0.10`. Models: `probe` and `resnet_color`.
- Produces: `results/e3_refusal.png` — coverage against `recal_n` per test site with the 0.90 target line; and risk–coverage curves (error among answered images against fraction answered, sweeping the confidence cut-off) in-site against cross-site.

- [ ] **Step 1: Write the failing test** — fake predictions that are calibrated in-site and overconfident cross-site; assert in-site coverage ≥ 0.87, cross-site coverage at `recal_n=0` < 0.80, cross-site coverage at `recal_n=100` ≥ 0.85, and that recalibration images are not in the scored set (`n` drops by `recal_n`).
- [ ] **Step 2: Run** — fail. **Step 3: Implement.** **Step 4: Run** — pass. **Step 5: Run on real predictions; commit.**

### Task 11: E4 heatmaps against lesion outlines (stretch; do only after Task 10)

**Files:**
- Create: `experiments/e4_heatmaps.py`
- Modify: `src/datasets.py` (`lesion_mask(site, root, path) -> np.ndarray | None` from the VIA JSON polygons of `smartom` and `code`), `tests/test_datasets.py`, `WHY.md`

**Interfaces:**
- Produces: `gradcam(checkpoint: Path, image: PIL.Image) -> np.ndarray` (H×W in [0, 1], last ResNet stage; written by hand with one forward and one backward hook) — reused by the demo.
- Produces: `results/e4_pointing.csv` — per (train site, test site) the pointing-game hit rate: fraction of true-refer test images whose heatmap maximum falls inside a lesion polygon; `results/e4_examples.png` — 8 examples.

- [ ] **Step 1: Failing tests**: `lesion_mask` on a hand-written VIA JSON with one square polygon returns a mask whose true area equals the square; an image without annotation returns `None`.
- [ ] **Step 2–4:** implement, pass. Runs in Colab (needs checkpoints) as one more notebook cell.
- [ ] **Step 5:** Commit CSV and figure.

### Task 12: Demo, report and README

**Files:**
- Create: `app/demo.py`, `REPORT.md`, `README.md`, `LICENSE` (MIT)
- Modify: `WHY.md`

**Interfaces:**
- Consumes: the best cross-site model from `results/e1_matrix.csv` (checkpoint or probe saved by Task 8 to `checkpoints/demo/`, fetched from Drive), its threshold and its conformal quantile from the training site's `val`.

- [ ] **Step 1: Write `app/demo.py`**: upload one photo (jpg/png, downscaled to at most 1024 px on the long side), show one of "Refer", "Do not refer", "Not sure — refer", the probability, and the Grad-CAM overlay if the demo model is the ResNet. A fixed banner states: research demo, not a medical device, trained on a few thousand photos from named sites, see a dentist or doctor for any mouth sore lasting more than two weeks. Run `streamlit run app/demo.py` on one image from each site and check the page renders.
- [ ] **Step 2: Write `REPORT.md`**: question; datasets and their limits (copy the "Known problems" list from the spec and update with what Tasks 2, 3 and 8 found); method; E1–E4 tables and figures with intervals; what helped, what did not; limits; references. Every number is copied from a CSV in `results/`. Run the `humanizer:humanizer` skill over the prose.
- [ ] **Step 3: Write `README.md`**: two-paragraph summary, the E1 heatmap, the headline numbers, how to rerun (local tests; Colab notebook; table scripts), dataset credits and licences, "not a medical device".
- [ ] **Step 4: Check `WHY.md`** has an entry for every `.py` file: `git ls-files '*.py'` against the headings in `WHY.md`.
- [ ] **Step 5: Final check**: `.venv/Scripts/python -m pytest -q` passes; `git status` shows no images tracked. Commit.
- [ ] **Step 6: Publish** — Sachchidananda creates `oral-screen-shift` on GitHub and pushes; done only on his explicit go-ahead.
