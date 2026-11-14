import pytest

from src.datasets import refer_label


@pytest.mark.parametrize("site,cls,expected", [
    ("smartom", "normal", 0), ("smartom", "variations from normal", 0),
    ("smartom", "opmd", 1), ("smartom", "oc", 1),
    ("code", "normal mucosa", 0), ("code", "variations from normal", 0),
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
