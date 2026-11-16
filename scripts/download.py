"""Fetch and unpack one of the datasets.

    python scripts/download.py --site china --out data
"""

import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

FIGSHARE = {"smartom": (31341790, "SMART-OM"), "code": (30550889, "Version_5")}
DIRECT = {
    "china": "https://huggingface.co/datasets/HRruiH/Dataset-of-oral-mucosal-diseases/resolve/main/mouth_data.zip",
    "karnataka": "https://data.mendeley.com/public-api/zip/mhjyrn35p4/download/2",
}
ZENODO = {"srilanka": 10664056}  # restricted: the owner must have accepted your access request


def zenodo_files(record: int, out: Path) -> None:
    """Fetch every file of a restricted Zenodo record, using a personal token from the environment."""
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("Set ZENODO_TOKEN to a Zenodo personal access token (Account > Applications).")
    auth = {"Authorization": f"Bearer {token}"}
    with urllib.request.urlopen(urllib.request.Request(f"https://zenodo.org/api/records/{record}", headers=auth)) as r:
        files = json.load(r)["files"]
    out.mkdir(parents=True)
    for f in files:
        request = urllib.request.Request(f["links"]["self"], headers=auth)
        with urllib.request.urlopen(request) as r, open(out / f["key"], "wb") as target:
            shutil.copyfileobj(r, target, 1 << 20)
        if f["key"].endswith(".zip"):
            with zipfile.ZipFile(out / f["key"]) as z:
                z.extractall(out)
            (out / f["key"]).unlink()


def zip_url(site: str) -> str:
    if site in DIRECT:
        return DIRECT[site]
    article, wanted = FIGSHARE[site]
    with urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{article}") as r:
        files = json.load(r)["files"]
    return next(f["download_url"] for f in files if wanted in f["name"])


def download(site: str, out: Path) -> None:
    target = out / site
    if target.exists():
        print(f"{site}: {target} already exists, skipped")
        return
    out.mkdir(parents=True, exist_ok=True)
    partial = out / f"{site}.partial"  # so an interrupted run is not mistaken for a finished one
    if site in ZENODO:
        shutil.rmtree(partial, ignore_errors=True)
        zenodo_files(ZENODO[site], partial)
        partial.rename(target)
        print(f"{site}: saved to {target}")
        return
    archive = out / f"{site}.zip"
    req = urllib.request.Request(zip_url(site), headers={"User-Agent": "oral-screen-shift"})
    with urllib.request.urlopen(req) as r, open(archive, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    with zipfile.ZipFile(archive) as z:
        z.extractall(partial)
    partial.rename(target)
    archive.unlink()
    print(f"{site}: unpacked to {target}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--site", required=True, choices=[*FIGSHARE, *DIRECT, *ZENODO])
    p.add_argument("--out", type=Path, default=Path("data"))
    a = p.parse_args()
    download(a.site, a.out)
