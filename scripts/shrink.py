"""Save a small copy of every indexed photo, so training reads them fast.

Phone photos are several megabytes each and the models only see 224 pixels, so decoding the
originals every epoch is most of the training time. Relative paths are kept, so the split files
work unchanged with `--data <out>`.

    python scripts/shrink.py --data data --out data_small
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import load_rgb
from src.splits import load_splits

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("data_small"))
    p.add_argument("--side", type=int, default=320)
    a = p.parse_args()

    paths = load_splits(0).path
    for n, rel in enumerate(paths, 1):
        target = a.out / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            image = load_rgb(a.data / rel)
            image.thumbnail((a.side, a.side))
            image.save(target, quality=95)
        print(f"{n}/{len(paths)}", end="\r")
    print(f"\n{len(paths)} photos in {a.out}")
