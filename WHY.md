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
