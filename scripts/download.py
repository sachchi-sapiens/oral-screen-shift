"""Fetch and unpack one of the four public datasets.

    python scripts/download.py --site china --out data
"""

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

FIGSHARE = {"smartom": (31341790, "SMART-OM"), "code": (30550889, "Version_5")}
DIRECT = {
    "china": "https://huggingface.co/datasets/HRruiH/Dataset-of-oral-mucosal-diseases/resolve/main/mouth_data.zip",
    "karnataka": "https://data.mendeley.com/public-api/zip/mhjyrn35p4/download/2",
}


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
    archive = out / f"{site}.zip"
    req = urllib.request.Request(zip_url(site), headers={"User-Agent": "oral-screen-shift"})
    with urllib.request.urlopen(req) as r, open(archive, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    partial = out / f"{site}.partial"  # so an interrupted unzip is not mistaken for a finished one
    with zipfile.ZipFile(archive) as z:
        z.extractall(partial)
    partial.rename(target)
    archive.unlink()
    print(f"{site}: unpacked to {target}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--site", required=True, choices=[*FIGSHARE, *DIRECT])
    p.add_argument("--out", type=Path, default=Path("data"))
    a = p.parse_args()
    download(a.site, a.out)
