import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import download


def fake_zenodo(monkeypatch, entries, bodies):
    """Stand in for zenodo.org: one listing call, then one call per file. Records the headers sent."""
    seen = []

    def urlopen(request):
        seen.append(request)
        if request.full_url.endswith("/files"):
            return io.BytesIO(json.dumps({"entries": entries}).encode())
        return io.BytesIO(bodies[request.full_url])

    monkeypatch.setattr(download.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("ZENODO_TOKEN", "t0ken")
    return seen


def test_restricted_record_files_are_saved_and_zips_unpacked(tmp_path, monkeypatch):
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, "w") as z:
        z.writestr("Images/a.jpg", b"jpeg bytes")
    entries = [{"key": "Images.zip", "links": {"content": "https://zenodo.org/x/Images.zip/content"}},
               {"key": "Imagewise_data.csv", "links": {"content": "https://zenodo.org/x/Imagewise_data.csv/content"}}]
    seen = fake_zenodo(monkeypatch, entries, {entries[0]["links"]["content"]: packed.getvalue(),
                                              entries[1]["links"]["content"]: b"id,category\n"})
    download.download("srilanka", tmp_path)
    assert (tmp_path / "srilanka/Images/a.jpg").read_bytes() == b"jpeg bytes"
    assert (tmp_path / "srilanka/Imagewise_data.csv").exists() and not (tmp_path / "srilanka/Images.zip").exists()
    assert all(r.get_header("Authorization") == "Bearer t0ken" for r in seen)


def test_empty_listing_stops_with_a_message_and_leaves_no_folder(tmp_path, monkeypatch):
    fake_zenodo(monkeypatch, [], {})
    with pytest.raises(SystemExit, match="no files"):
        download.download("srilanka", tmp_path)
    assert not (tmp_path / "srilanka").exists()


def test_an_empty_folder_from_a_failed_run_is_not_treated_as_done(tmp_path, monkeypatch):
    (tmp_path / "srilanka").mkdir()
    entries = [{"key": "a.csv", "links": {"content": "https://zenodo.org/x/a.csv/content"}}]
    fake_zenodo(monkeypatch, entries, {entries[0]["links"]["content"]: b"x"})
    download.download("srilanka", tmp_path)
    assert (tmp_path / "srilanka/a.csv").exists()
