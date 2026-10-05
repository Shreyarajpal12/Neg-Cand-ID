"""KuaiRec 2.0 and KuaiRand-Pure: interaction tables with a primary category.

    python -m prep.kuairec_kuairand

Inputs
    data/kuairec/{big_matrix.csv, small_matrix.csv, item_categories.csv}
    data/KuaiRand-Pure/data/{log_standard_4_08_to_4_21_pure.csv,
        log_standard_4_22_to_5_08_pure.csv, log_random_4_22_to_5_08_pure.csv,
        video_features_basic_pure.csv}

Outputs (columns user_id, video_id, label, category; KuaiRec also watch_ratio)
    data/kuairec/prepared/{big, small}.parquet
    data/kuairand/prepared/{standard_train, standard_val, random}.parquet

KuaiRec: label = 1 when watch_ratio >= 1.0; duplicate (user, video) rows are
dropped. KuaiRand: label = is_click. The category of a video is the first id
in its feature (KuaiRec) or tag (KuaiRand) list, -1 when absent.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from paths import DATA, kuairand_table, kuairec_table

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def parse_feat(v):
    s = str(v).strip().strip("[]").replace(" ", "")
    return [int(x) for x in s.split(",") if x not in ("", "nan")]


def main():
    os.makedirs(os.path.dirname(kuairec_table("big")), exist_ok=True)
    os.makedirs(os.path.dirname(kuairand_table("random")), exist_ok=True)

    # ---------------- KuaiRec ----------------
    cats = pd.read_csv(f"{DATA}/kuairec/item_categories.csv")
    cats["feat_list"] = cats["feat"].map(parse_feat)
    cats["category"] = cats["feat_list"].map(lambda l: int(l[0]) if l else -1)
    cat_map = cats.set_index("video_id")["category"]

    for name, f in (("big", "big_matrix.csv"), ("small", "small_matrix.csv")):
        d = pd.read_csv(f"{DATA}/kuairec/{f}", usecols=["user_id", "video_id", "watch_ratio"])
        d["label"] = (d["watch_ratio"] >= 1.0).astype(np.int8)
        d["category"] = d["video_id"].map(cat_map).fillna(-1).astype(np.int16)
        d = d.drop_duplicates(["user_id", "video_id"])
        d.to_parquet(kuairec_table(name), index=False)
        log(f"kuairec {name}: {len(d):,} rows, {d.user_id.nunique():,} users, "
            f"{d.video_id.nunique():,} items, pos_rate={d.label.mean():.4f}, "
            f"cats={d.category.nunique()}")

    big = pd.read_parquet(kuairec_table("big"), columns=["user_id", "video_id"])
    small = pd.read_parquet(kuairec_table("small"), columns=["user_id", "video_id"])
    su, bu = set(small.user_id.unique()), set(big.user_id.unique())
    log(f"kuairec small users also in big: {len(su & bu)}/{len(su)}")
    ov = pd.merge(big, small, on=["user_id", "video_id"], how="inner")
    log(f"kuairec (user, video) pairs present in both big and small: {len(ov):,} "
        f"({100*len(ov)/len(small):.2f}% of small)")

    # ---------------- KuaiRand ----------------
    KD = f"{DATA}/KuaiRand-Pure/data"
    vb = pd.read_csv(f"{KD}/video_features_basic_pure.csv", usecols=["video_id", "tag"])
    vb["category"] = vb["tag"].map(lambda v: parse_feat(v)[0] if parse_feat(v) else -1).astype(np.int16)
    tag_map = vb.set_index("video_id")["category"]
    COLS = ["user_id", "video_id", "is_click"]
    for name, f in (("standard_train", "log_standard_4_08_to_4_21_pure.csv"),
                    ("standard_val", "log_standard_4_22_to_5_08_pure.csv"),
                    ("random", "log_random_4_22_to_5_08_pure.csv")):
        d = pd.read_csv(f"{KD}/{f}", usecols=COLS)
        d["label"] = d["is_click"].astype(np.int8); d = d.drop(columns=["is_click"])
        d["category"] = d["video_id"].map(tag_map).fillna(-1).astype(np.int16)
        d.to_parquet(kuairand_table(name), index=False)
        log(f"kuairand {name}: {len(d):,} rows, {d.user_id.nunique():,} users, "
            f"{d.video_id.nunique():,} items, click_rate={d.label.mean():.4f}, "
            f"cats={d.category.nunique()}")
    log("done")


if __name__ == "__main__":
    main()
