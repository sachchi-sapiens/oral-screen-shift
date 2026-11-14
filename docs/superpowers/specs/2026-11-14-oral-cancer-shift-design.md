# Oral cancer screening from phone photos: does it survive a change of hospital?

Design document, 14 November 2026. Status: waiting for approval.

## Why

India has roughly a third of the world's oral cancer cases and most are found late. Early lesions are visible in the mouth, so a health worker with a phone could flag them. Published models report accuracy above 90%, but each one is tested on photos from the same hospital, camera protocol and population it was trained on. A screening tool will be used somewhere else. Nobody has published a clean test of what happens then.

The project answers one question: **when a model trained on one site's oral photos is used on another site's, how much does it lose, and what recovers it?**

It is also meant to be a piece of work Sachchidananda can put on a CV and discuss in a masters or PhD application: a stated hypothesis, public data, baselines, ablations and honest reporting.

## Goals

1. Measure the drop. Train on each dataset, test on every dataset, and report the full train/test matrix.
2. Compare a small number of fixes and say which ones help.
3. Make the model able to say "not sure, refer to a doctor", and check whether that safety guarantee still holds at a new site.
4. Publish code that reruns every table from a single command per experiment, a short paper-style report, and a small demo.

## Non-goals

- A new network architecture.
- Lesion segmentation or detection as a trained task.
- Histopathology (microscope) images.
- Any clinical claim. This is research code, not a screening product.

## What would make this a failure even if it runs

- Images of one patient on both sides of a split. This inflates every number and is the most common flaw in the papers we are checking.
- Picking thresholds or hyperparameters on the test site.
- Reporting only the experiments that came out well.
- A repository full of unused abstractions. Each file must earn its place.

## Data

| Name | Source | Size | Labels | Access |
|---|---|---|---|---|
| Sri Lanka | Zenodo 10664056, Piyarathne et al., Oral Oncology 2024 | 3,000 images, 714 patients | healthy, benign, OPMD, cancer; lesion outlines (COCO) | Restricted, request on Zenodo. CC BY-NC-ND 4.0 |
| SMART (Tamil Nadu) | Zenodo 16899860, ICMR SMITA project | 2,640 images, 356 patients | normal, variation from normal, OPMD | Restricted, request on Zenodo. CC BY 4.0 |
| Karnataka | Mendeley Data mhjyrn35p4 | 323 original images | benign lesion, malignant lesion | Open. CC BY 4.0 |

OPMD means "oral potentially malignant disorder": a lesion that is not cancer yet but can become it.

**Common task.** The three label sets differ, so every dataset is mapped to the decision a screener actually makes: *refer* (OPMD or cancer) versus *do not refer* (healthy, normal variation, benign). The mapping lives in one file and is tested.

**Known problems, stated up front.**
- Two datasets need an access request from Sachchidananda's Zenodo account. Approval is up to the owners. Work starts on Karnataka so the pipeline is ready when access arrives.
- Karnataka has no healthy mouths and (as far as we can tell) no patient ids, and it is small. It is used as a test site and for pipeline development, not as a main training site. If patient ids are missing, images are split individually and the report says so.
- SMART has no cancer class, so "refer" there means OPMD only. This is a label shift on top of the image shift; the report separates the two where it can.
- Sri Lanka's licence forbids redistributing derivatives. The repository ships no images and no cached features, only download scripts and code.
- If only one restricted dataset is granted, the study runs with two sites plus Karnataka. If neither is granted, we stop and choose again; a single 323-image dataset cannot support the question.
- A fourth dataset, SMART-OM (2,469 images), is described in a preprint but we found no download location. It is added only if one turns up.

## Experiments

Each experiment is one script that writes one results table (CSV) and one figure.

**E1. The matrix.** For each training site and each test site, report AUROC, sensitivity at the threshold chosen on the training site's validation split, specificity, and calibration error (ECE). Two models:
- a ResNet-50 pretrained on ImageNet and fine-tuned end to end (the kind of model the published papers use);
- a linear classifier on frozen DINOv2 features (a general-purpose pretrained vision model).

Diagonal cells are in-site results; off-diagonal cells are the answer to the research question.

**E2. Fixes.** Three, chosen because each tests a different explanation for the drop:
- colour and lighting augmentation (is the drop about camera and lighting?);
- training on all sites but one and testing on the held-out site (does variety in training help?);
- frozen foundation features versus fine-tuning (already in E1; does a model that never saw the site's quirks overfit to them less?).

**E3. Safe refusal.** Split conformal prediction on the training site gives a rule of the form "answer only when confident; otherwise refer", with a promised error rate. Report whether the promise holds in-site and at the other sites, plus risk-coverage curves. Then recalibrate using a small labelled sample from the new site (25, 50, 100 images) and report how many are needed to restore the promise. This is the practical result: how much local labelling a clinic would need before using the tool.

**E4. Where the model looks (stretch).** Grad-CAM heatmaps on Sri Lanka, scored against the expert lesion outlines by the fraction of heatmap peaks that fall inside a lesion. Done only if E1 to E3 are finished.

**Statistics.** Three random seeds per trained model. 95% confidence intervals by bootstrap over patients, not images. Splits are 70/15/15 by patient, fixed by seed and saved to disk.

## How it is built (builder's calls, not yet agreed by Sachchidananda)

- Python 3.11+, PyTorch, `timm` for model weights, scikit-learn for the linear probe and metrics, pandas, matplotlib, pytest. Streamlit for the demo, matching his earlier project.
- No LLM is used. The DeepSeek key from the earlier project is not needed.
- GPU work (ResNet fine-tuning, one-off DINOv2 feature extraction) runs on free Google Colab through a single notebook that calls the same scripts. Features are cached to disk, so the linear-probe, conformal and plotting work runs on a laptop CPU.
- Conformal prediction is written by hand (about 20 lines) rather than adding a library.
- Config is a plain YAML file per experiment. No experiment-tracking service.
- Repository name: `oral-screen-shift`. Licence: MIT for code. The repository root is the existing folder `Desktop/medical domain AI project`.
- The demo loads whichever model has the best cross-site AUROC in E1, with the E3 refusal rule.
- Karnataka appears in the E1 matrix as a test column only.

Planned layout:

```
oral-screen-shift/
  README.md            what, why, results tables, how to rerun
  WHY.md               one entry per code file: why it exists, what would break without it
  REPORT.md            paper-style write-up: question, data, method, results, limits
  data/README.md       how to request and place each dataset (no images in git)
  src/
    datasets.py        load each dataset, map labels to refer / do not refer
    splits.py          patient-level splits, saved to disk
    features.py        extract and cache frozen features
    train.py           fine-tune baseline; fit linear probe
    evaluate.py        metrics, bootstrap intervals, the train/test matrix
    conformal.py       calibration, refusal rule, recalibration
  experiments/         e1_matrix.py, e2_fixes.py, e3_refusal.py, e4_heatmaps.py + YAML configs
  notebooks/colab_gpu.ipynb
  app/demo.py          upload a photo, get refer / do not refer / not sure, with heatmap
  tests/
```

Tests cover what can silently corrupt results: no patient appears in two splits, the label mapping for every class of every dataset, metric functions against hand-computed values, and conformal coverage on synthetic data where the true answer is known. Tests use tiny fake images and need no dataset or GPU.

## What Sachchidananda needs to do

1. Request access to Zenodo records 10664056 and 16899860 from his own account, giving the research purpose honestly.
2. Run the Colab notebook when it is ready (his Google account, his Drive for the data).
3. Create the GitHub repository and push when the work is done.

## Left to the builder

Learning rates, image size, augmentation strengths, bootstrap count, plot styling, and the exact wording of the report. Any of these that turn out to change a conclusion get reported as a sensitivity check.

## What done looks like

- `pytest` passes offline.
- Each experiment script regenerates its table and figure from cached features.
- `REPORT.md` states the answer to the research question with intervals, including results that went against expectation.
- `WHY.md` has an entry for every code file.
- The demo runs locally on one photo.
