import pandas as pd
import pytest

from PIL import Image

from src.datasets import build_index, load_index, refer_label


@pytest.mark.parametrize("site,cls,expected", [
    ("smartom", "normal", 0), ("smartom", "variation from normal", 0),
    ("smartom", "opmd", 1), ("smartom", "oral cancer", 1),
    ("code", "normal", 0), ("code", "variations from normal", 0),
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


def tree(root, files):
    """Create tiny real images at the given relative paths."""
    for f in files:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4)).save(root / f)
    return root


@pytest.fixture
def fake_china(tmp_path):
    return tree(tmp_path, ["china/mouth_data/Normal/1-1.jpg", "china/mouth_data/Normal/1-2.png",
                           "china/mouth_data/OLK/2-1.jpg", "china/mouth_data/OCA/5-1.jpg"])


def test_build_index_columns_and_labels(fake_china):
    df = build_index("china", fake_china)
    assert list(df.columns) == ["path", "site", "patient_id", "orig_class", "label"]
    assert sorted(df.label) == [0, 0, 1, 1] and (df.site == "china").all() and df.patient_id.isna().all()
    assert (fake_china / df.path[0]).exists()          # paths are relative to the data root


def test_unreadable_file_skipped(fake_china, capsys):
    (fake_china / "china/mouth_data/Normal/Thumbs.jpg").write_bytes(b"junk")
    n = len(build_index("china", fake_china))
    assert "skipped 1" in capsys.readouterr().out and n == 4


def test_karnataka_ignores_augmented(tmp_path):
    base = "karnataka/Oral Images Dataset/"
    tree(tmp_path, [base + "original_data/benign_lesions/1.jpg", base + "original_data/malignant_lesions/1.jpg",
                    base + "augmented_data/augmented_benign/1.jpg"])
    df = build_index("karnataka", tmp_path)
    assert sorted(df.orig_class) == ["benign", "malignant"] and not df.path.str.contains("augmented").any()


def test_smartom_uses_unannotated_only_and_reads_patient_ids(tmp_path):
    base = "smartom/SMART-OM/"
    tree(tmp_path, [base + "01. Normal/01. Unannotated/03. Left buccal mucosa/SMITA00126_R_LB.jpeg",
                    base + "01. Normal/01. Unannotated/01. Dorsal tongue/5 - DT.jpg",
                    base + "03. OPMD/01. Unannotated/04. Right buccal mucosa/SMITA00126_R_RB.jpg",
                    base + "04. Oral Cancer/01. Unannotated/Ca 1.JPG",
                    base + "03. OPMD/04. Lesion annotation/04. Right buccal mucosa/SMITA00126_R_RB.jpg"])
    df = build_index("smartom", tmp_path).sort_values("path").reset_index(drop=True)
    assert len(df) == 4 and not df.path.str.contains("annotation").any()
    by_name = {p.split("/")[-1]: (pid, label) for p, pid, label in zip(df.path, df.patient_id, df.label)}
    assert by_name["SMITA00126_R_LB.jpeg"] == ("SMITA00126", 0)
    assert by_name["SMITA00126_R_RB.jpg"] == ("SMITA00126", 1)   # same patient, a lesion at another site
    assert by_name["5 - DT.jpg"] == ("n5", 0)
    assert pd.isna(by_name["Ca 1.JPG"][0]) and by_name["Ca 1.JPG"][1] == 1


def test_code_skips_drawn_on_copies_and_tolerates_misspelt_folders(tmp_path):
    base = "code/CODE - x_Version_5/"
    tree(tmp_path, [base + "01. NORMAL IMAGES/SMITA03453/UNANNOTATED/SMITA03453_KSR_DT.jpg",
                    base + "01. NORMAL IMAGES/SMITA03453/ANNOTATED/SMITA03453_KSR_DT.jpg",
                    base + "03. OPMD IMAGES/SMITA03515/UN ANNOTATED/SMITA03515_KSR_RB.jpg",
                    base + "04. ORAL CANCER IMAGES/SMITA09999/Unannoated/SMITA09999_KSR_LB.jpg",
                    base + "02. VARIATIONS FROM NORMAL IMAGES/SMITA07745/UNANNOTATED IMAGES/SMITA07745_KSR_LB.jpg"])
    df = build_index("code", tmp_path)
    assert len(df) == 4 and sorted(df.label) == [0, 0, 1, 1]
    assert set(df.patient_id) == {"SMITA03453", "SMITA03515", "SMITA09999", "SMITA07745"}


def test_load_index_joins_sites(fake_china, tmp_path):
    tree(tmp_path, ["karnataka/Oral Images Dataset/original_data/benign_lesions/1.jpg"])
    assert set(load_index(tmp_path, ["china", "karnataka"]).site) == {"china", "karnataka"}
