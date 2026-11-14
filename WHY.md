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
