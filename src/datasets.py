"""Turn each dataset's own class names into one screening decision: refer or do not refer."""

# 1 = refer (pre-cancer or cancer), 0 = do not refer. Keys are lower-cased class names.
REFER = {
    "smartom": {"normal": 0, "variations from normal": 0, "opmd": 1, "oc": 1},
    "code": {"normal mucosa": 0, "variations from normal": 0, "opmd": 1, "oral cancer": 1},
    "china": {"normal": 0, "olk": 1, "olp": 1, "osf": 1, "oca": 1},
    "karnataka": {"benign": 0, "malignant": 1},
}


def refer_label(site: str, orig_class: str) -> int:
    key = orig_class.strip().lower()
    if key not in REFER[site]:
        raise KeyError(f"{site}: no refer mapping for class '{key}'")
    return REFER[site][key]
