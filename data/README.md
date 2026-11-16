# Data

No image is stored in this repository. `python scripts/download.py --site <name> --out data` fetches each dataset from its owner's host. Please cite the owners if you use their data.

| Site name | Dataset | Host | Licence |
|---|---|---|---|
| `smartom` | SMART-OM: a smartphone-based expert-annotated dataset of oral mucosa images (Madan Kumar, Ranganathan et al., 2026) | figshare, doi:10.6084/m9.figshare.31341790 | CC BY 4.0 |
| `code` | CODE: Comprehensive Oral mucosa Database with Explanations, version 5 (same research network as SMART-OM) | figshare, doi:10.6084/m9.figshare.30550889 | CC BY 4.0 |
| `china` | Dataset of oral mucosal diseases (Yu, Huang, Wang et al., BMC Oral Health 26, 169, 2026) | Hugging Face `HRruiH/Dataset-of-oral-mucosal-diseases` | Apache 2.0 |
| `karnataka` | Oral Images Dataset (hospitals and colleges in Karnataka, India) | Mendeley Data `mhjyrn35p4`, version 2 | CC BY 4.0 |

## What is inside, as found on 14 November 2026

### smartom

```
SMART-OM/<NN. class>/01. Unannotated/<NN. mouth region>/<file>
SMART-OM/<NN. class>/02. Region annotation | 03. Full annotation | 04. Lesion annotation/...
```

Only `01. Unannotated` is used. The other folders hold the same photos with outlines drawn on them, which would give the answer away.

Most files are named `SMITA00126_R_LB.jpeg` (patient, phone type, mouth region). 53 are not: `5 - DT.jpg` style (a patient number, kept as `n5`) and `Ca 1.JPG` or `6026.jpg` style (no patient given; these are grouped by photo similarity instead).

| Class | Photos | Patients |
|---|---|---|
| normal | 2,145 | 337 |
| variation from normal | 179 | 111 |
| OPMD | 125 | 64 |
| oral cancer | 20 | 20 |

A patient can appear in more than one class (healthy regions plus a lesion). 331 patient groups in total; 84 of them have at least one refer photo.

### code

```
<top>/<NN. CLASS IMAGES>/<SMITA patient id>/<UNANNOTATED | ANNOTATED | JSON>/<file>
```

The three sub-folder names are spelt about twenty ways (`UN ANNOTATED`, `Unannoated`, `UNANNODATED`, ...). A photo is used when the letters of its folder name start with `UN`. `ANNOTATED` copies have the lesion outline drawn on the photo and are never used.

| Class | Photos in the archive | After removing photos shared with `smartom` |
|---|---|---|
| normal | 5,431 | 5,391 |
| variations from normal | 166 | 163 |
| OPMD | 14 | 13 |
| oral cancer | 7 | 0 |

20 files could not be opened as images and are skipped.

**After cleaning, `code` has 13 refer photos from 6 patients.** That is far too few to train on, so `code` is a test site only. Its value is the other 5,554 photos: how often a model raises a false alarm on healthy mouths at a centre it has never seen. Sensitivity on `code` rests on 13 photos and is reported with its interval.

### china

```
mouth_data/<Normal | OLK | OLP | OSF | OCA>/<class number>-<photo number>.jpg
```

| Class | Photos |
|---|---|
| normal | 286 |
| OLK (leukoplakia) | 253 |
| OLP (lichen planus) | 310 |
| OSF (submucous fibrosis) | 225 |
| OCA (cancer) | 274 |

No patient ids. 79% of photos are refer cases, against 6% in `smartom`: the two main sites differ in case mix as well as in cameras. Photos come from phones, cameras and published papers. 41 photos were merged into groups with a near-identical photo, leaving 1,307 groups.

### karnataka

```
Oral Images Dataset/original_data/<benign_lesions | malignant_lesions>/<n>.jpg
Oral Images Dataset/augmented_data/...
```

Only `original_data` is used: 165 benign and 158 malignant. `augmented_data` holds flipped and rotated copies. No patient ids, no healthy mouths. Test site only.

## Photos shared between datasets

Every photo gets a 64-bit difference hash; two photos within 4 bits of each other are treated as the same picture.

**`china` and `karnataka`: 10 photos, all 0 bits apart.** `china` labels them leukoplakia or lichen planus; `karnataka` labels the same photos malignant. Both datasets appear to have taken them from a common published source. They are removed from `karnataka`, leaving 148 malignant photos.

**`smartom` and `code`: 33 photos.** `code`'s description calls the two datasets independent. Seven pairs are 0 bits apart, and those are all seven of `code`'s oral cancer photos, which also sit in `smartom`'s cancer folder under different file names. One more pair is 1 bit apart. The remaining 25 pairs are 3 or 4 bits apart, mostly healthy mouths; some of those may be different photos that happen to look alike under the same photo protocol, but removing them is the safe choice. All 33 are removed from `code`, because `smartom` is a training site. No patient id appears in both datasets.

No photo is shared between any other pair of datasets.

## Splits

`python scripts/prepare.py` writes `splits/seed0.csv`, `seed1.csv`, `seed2.csv`. Each seed is a different 70/15/15 split by patient group inside `smartom` and `china`. Every `code` and `karnataka` photo is marked `test`.

Each site and class is shuffled with its own seed, so adding or removing a dataset never changes another dataset's split.

Seed 0, photos (refer photos):

| Site | train | val | test |
|---|---|---|---|
| smartom | 1,737 (105) | 374 (20) | 358 (20) |
| china | 943 (743) | 202 (159) | 203 (160) |
| code | | | 5,567 (13) |
| karnataka | | | 313 (148) |

The `smartom` test split has about 20 refer photos and `code` has 13. Sensitivity figures for those two carry wide intervals and are reported with them.
