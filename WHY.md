# Why each file exists

This project asks one question: when a model that flags suspicious mouth lesions from phone photos is trained at one site and used at another, how much worse does it get, and what brings it back? Every code file below is here because answering that question needs it. If a file has no entry, it should not be in the repository.

## src/datasets.py

The four datasets name their classes differently (one says "OPMD", another lists leukoplakia and lichen planus separately, one only has "benign" and "malignant"). A model cannot be moved between sites unless they share a label. This file maps every class to the decision a screener makes: refer or do not refer. Without it, each experiment would do its own mapping and a silent mismatch would look like a drop in accuracy.
