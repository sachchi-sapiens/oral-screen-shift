"""Build the photo index, group photos by patient, report duplicates, and write the splits.

    python scripts/prepare.py --data data --sites smartom china karnataka
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import load_index
from src.splits import add_groups, cross_site_duplicates, make_splits, save_splits

TEST_ONLY = {"code", "karnataka"}  # too few refer cases (code) or too few photos (karnataka) to train on

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--sites", nargs="+", default=["smartom", "china", "code", "karnataka"])
    a = p.parse_args()

    df = add_groups(load_index(a.data, a.sites), a.data)

    shared = cross_site_duplicates(df)
    print(f"\nphotos that appear in two datasets: {len(shared)}")
    if len(shared):
        print(shared.to_string(index=False))
        # keep the copy in the first-listed site, drop the other, so no test site holds a training photo
        df = df[~df.path.isin(shared.path_b)]

    print("\nphotos per site and class")
    print(df.groupby(["site", "orig_class"]).size().to_string())
    print("\nper site: photos, groups, refer photos, groups with a refer photo")
    refer = df[df.label == 1]
    print(pd.DataFrame({"photos": df.groupby("site").size(), "groups": df.groupby("site").group.nunique(),
                        "refer": refer.groupby("site").size(), "refer_groups": refer.groupby("site").group.nunique()}).to_string())

    for seed in (0, 1, 2):
        split = make_splits(df, seed)
        split.loc[split.site.isin(TEST_ONLY), "split"] = "test"
        save_splits(split, seed)
        if seed == 0:
            print("\nseed 0 split sizes (photos / refer photos)")
            print(split.groupby(["site", "split"]).label.agg(["size", "sum"]).to_string())
