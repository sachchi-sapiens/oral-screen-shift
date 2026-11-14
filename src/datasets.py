"""Find every usable photo in each dataset and give it one screening label: refer or do not refer."""

import re
from pathlib import Path

import pandas as pd
from PIL import Image

# 1 = refer (pre-cancer or cancer), 0 = do not refer. Keys are lower-cased class names.
REFER = {
    "smartom": {"normal": 0, "variation from normal": 0, "opmd": 1, "oral cancer": 1},
    "code": {"normal": 0, "variations from normal": 0, "opmd": 1, "oral cancer": 1},
    "china": {"normal": 0, "olk": 1, "olp": 1, "osf": 1, "oca": 1},
    "karnataka": {"benign": 0, "malignant": 1},
}
IMAGE = re.compile(r"\.(jpe?g|png)$", re.I)


def refer_label(site: str, orig_class: str) -> int:
    key = orig_class.strip().lower()
    if key not in REFER[site]:
        raise KeyError(f"{site}: no refer mapping for class '{key}'")
    return REFER[site][key]


def _class_from_folder(name: str) -> str:
    """'03. OPMD IMAGES' -> 'opmd', 'benign_lesions' -> 'benign'."""
    name = re.sub(r"^\d+\.\s*", "", name).lower()
    return re.sub(r"(\s+images|_lesions)$", "", name)


# Each function below takes a path split into parts (relative to the site folder) and returns
# (class folder name, patient id or None), or None when the file must not be used.

def _china(parts):          # mouth_data/<class>/<file>
    return parts[-2], None


def _karnataka(parts):      # Oral Images Dataset/original_data/<class>_lesions/<file>
    # augmented_data holds flipped and rotated copies; using them would put test photos in training
    return (parts[-2], None) if "original_data" in parts else None


def _smartom(parts):        # SMART-OM/<class>/01. Unannotated/<mouth region>/<file>
    if "01. Unannotated" not in parts:
        return None         # the other folders are the same photos with outlines drawn on them
    name = parts[-1]
    patient = re.match(r"SMITA\d+", name) or re.match(r"(\d+)\s*-\s*[A-Z]{2}\b", name)
    if patient is None:
        return parts[1], None
    return parts[1], patient.group(0) if name.startswith("SMITA") else f"n{patient.group(1)}"


def _code(parts):           # <top>/<class>/<patient>/<UNANNOTATED, spelt many ways>/<file>
    if not re.sub(r"[^A-Z]", "", parts[-2].upper()).startswith("UN"):
        return None         # ANNOTATED copies have the lesion outline drawn on the photo
    patient = next((p for p in parts if re.fullmatch(r"SMITA\d+", p)), None)
    return parts[1], patient


READERS = {"china": _china, "karnataka": _karnataka, "smartom": _smartom, "code": _code}


def _readable(path: Path) -> bool:
    try:
        Image.open(path).verify()
        return True
    except Exception:
        return False


def build_index(site: str, data_root: Path) -> pd.DataFrame:
    """One row per usable photo under data_root/site. Paths are relative to data_root."""
    data_root = Path(data_root)
    rows, skipped = [], 0
    for path in sorted((data_root / site).rglob("*")):
        if not path.is_file() or not IMAGE.search(path.name):
            continue
        found = READERS[site](path.relative_to(data_root / site).parts)
        if found is None:
            continue
        if not _readable(path):
            skipped += 1
            continue
        orig_class = _class_from_folder(found[0])
        rows.append({"path": path.relative_to(data_root).as_posix(), "site": site, "patient_id": found[1],
                     "orig_class": orig_class, "label": refer_label(site, orig_class)})
    print(f"{site}: {len(rows)} photos, skipped {skipped} unreadable")
    return pd.DataFrame(rows, columns=["path", "site", "patient_id", "orig_class", "label"])


def load_index(data_root: Path, sites: list[str]) -> pd.DataFrame:
    return pd.concat([build_index(s, data_root) for s in sites], ignore_index=True)
