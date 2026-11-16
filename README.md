# Oral screen shift

Does a model that flags suspicious mouth lesions from phone photos still work when it is moved to a different hospital?

Oral cancer is common in South Asia and is usually found late, although early lesions can be seen in the mouth. Published models report accuracy above 90%, but each is tested on photos from the same place it was trained on. This project trains on one public dataset and tests on the others, measures the drop, tries three fixes, and checks whether a "not sure, see a doctor" rule keeps its promised error rate after the move.

**Status: work in progress.** The code and tests are here; the experiments are being run and the results are not in yet. This page will be replaced with the findings.

**Not a medical device.** Research code only.

## What is here

| Path | What it is |
|---|---|
| `WHY.md` | One entry per code file: why it exists |
| `data/README.md` | The datasets, their licences, and what was found inside them |
| `src/` | Dataset indexing, patient-level splits, metrics, conformal refusal, models |
| `experiments/` | The three experiments and the script that trains the models |
| `notebooks/colab_gpu.ipynb` | Runs everything on a free Colab GPU |
| `docs/superpowers/specs/` | The design document |

## Run the tests

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -q
```

Tests use tiny generated images. They need no dataset and no GPU.

## Data

No image is stored in this repository. `scripts/download.py` fetches each dataset from its owner. One dataset is restricted and needs its owner's approval. See `data/README.md`.

## Licence

Code: MIT. The datasets keep their own licences.
