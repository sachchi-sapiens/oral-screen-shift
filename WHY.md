# Why each file exists

This project asks one question: when a model that flags suspicious mouth lesions from phone photos is trained at one site and used at another, how much worse does it get, and what brings it back? Every code file below is here because answering that question needs it. If a file has no entry, it should not be in the repository.

## src/datasets.py

The four datasets name their classes differently (one says "OPMD", another lists leukoplakia and lichen planus separately, one only has "benign" and "malignant"). A model cannot be moved between sites unless they share a label. This file maps every class to the decision a screener makes: refer or do not refer. Without it, each experiment would do its own mapping and a silent mismatch would look like a drop in accuracy.

## scripts/download.py

The datasets are several gigabytes and their licences are their owners', so no image is stored in this repository. This script is how anyone, including the Colab notebook, gets the same four archives from their original hosts. Without it the study could not be rerun.

## src/evaluate.py

Holds the numbers every table reports: AUROC, sensitivity, specificity and calibration error, plus the rule that picks the decision threshold on the training site's validation data. Its bootstrap resamples patients, not photos: several photos of one mouth are not independent evidence, and treating them as if they were gives intervals that are too narrow. Without one shared file, two experiments could compute "sensitivity" in two ways.

## src/conformal.py

A screening tool should be able to say "not sure, see a doctor". Split conformal prediction gives that rule together with a promised error rate. Experiment 3 asks whether the promise survives a move to a new site, so the rule has to be written out and tested on data where the right answer is known. It is about twenty lines, so it is written here instead of pulling in a library.

## src/splits.py

The most common flaw in papers on this topic is photos of one patient landing in both training and test data, which makes a model look better than it is. This file puts each patient wholly in train, validation or test. Two datasets do not name their patients, so it also fingerprints every photo (a 64-bit difference hash) and treats near-identical photos as one patient. The same fingerprints find photos shared between two datasets, which would otherwise make a "new site" test partly a test on training images. The fingerprint is about five lines, so no hashing library is used.

## src/features.py

Two jobs. `load_rgb` is the single way a photo is opened: phone photos carry a rotation flag and arrive as greyscale, CMYK or with transparency, and a model fed sideways or four-channel images fails in ways that look like a site effect. `extract` runs a frozen pretrained model over every photo once and saves the result, so the linear-probe and refusal experiments rerun in seconds on a laptop instead of needing a GPU each time.

## src/train.py

The two models being compared. The fine-tuned ResNet-50 stands for what published papers do. The linear probe on frozen DINOv2 features tests whether a model that never adjusts to one site's photos travels better. The colour-augmentation switch tests whether the cross-site drop is about cameras and lighting. Hyperparameters are fixed here, not tuned per site, so differences between cells of the results matrix come from the data.

## scripts/prepare.py

Runs once after downloading: lists every usable photo, groups photos by patient, reports photos that appear in two datasets and removes the second copy, and writes the three split files that every later step reads. Keeping this in one script means the Colab run and the laptop run cannot end up with different splits.

## experiments/common.py

Things all experiment scripts must agree on: which sites are trained on, how a prediction file is named, and that a decision threshold comes from the training site's validation data and never from the test site. Also the plot colours, so a method has the same colour in every figure.

## experiments/run_gpu.py

Trains each model and saves its refer probability for every validation and test photo. Saving predictions separates the slow part (needs a GPU, runs in Colab) from the analysis (runs anywhere in seconds), and lets a reader regenerate every table without retraining. It skips anything already saved, because free Colab sessions disconnect.

## experiments/e1_matrix.py

Experiment 1, the headline: train on one site, test on every site. The diagonal is what papers report; the off-diagonal is the answer to this project's question.

## experiments/e2_fixes.py

Experiment 2: compares the candidate fixes (colour augmentation, frozen pretrained features, training on two sites) at sites the model never saw.

## experiments/e3_refusal.py

Experiment 3: checks whether the "not sure, refer" rule keeps its 90% promise after a move, and how many labelled local photos are needed to restore it. This is the number a clinic would need before using such a tool.

## scripts/shrink.py

Phone photos are several megabytes and the models see 224 pixels. Decoding full-size photos every epoch would make the Colab run take most of a day; with small copies it takes about an hour. Relative paths are unchanged, so the same split files apply.

## notebooks/colab_gpu.ipynb

The laptop this was built on has no GPU. This notebook is the exact sequence of commands that fine-tunes the ResNet models on a free Colab GPU and saves their predictions to Google Drive. Each cell is one command from this repository, so nothing about the results lives only in a notebook.

## experiments/e4_site.py

Experiment 4, added after the first results. A model trained on one dataset scored healthy mouths from another dataset as lesions. This script asks why: it trains a classifier to say which dataset a healthy photo came from. If that is easy, the photos carry a "site signature" (camera, lighting, framing), and because the datasets differ so much in how many lesions they hold, a model can score well in-site by learning the signature instead of the lesion. It replaces the heatmap experiment in the original plan, because it explains the failure more directly.

## app/demo.py

A small page where one photo gets "refer", "do not refer" or "not sure", using the same probe and the same refusal rule as the experiments. It exists so the result can be shown to someone in a minute. Its warning banner is part of the code on purpose: the experiments show this model is not reliable on photos from a new place.

## tests/

Tests use tiny generated images and made-up predictions, so they need no dataset, no GPU and no network. They pin the things that would silently corrupt results: a patient in two splits, a wrong label mapping, a threshold that peeked at the test site, a refusal rule whose coverage is off, a download that reports success with no files.
