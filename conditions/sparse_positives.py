"""KuaiRec conditions with sparse clean positives: about 5%, 2% and 1% positive.

    python -m conditions.sparse_positives

Input:  data/kuairec/prepared/{big, small}.parquet (prep.kuairec_kuairand)
Output: data/conditions/kuairec/{pos05, pos02, pos01}/{train, val, test}.parquet
        results/summary/sparse_positives_stats.csv

Splits before thinning: train = big matrix minus a 10% validation slice drawn
with seed 42, val = that slice, test = small matrix. Each split is thinned
independently in two steps; labels are never changed.

  1. Category-volume thinning: keep an interaction of category c with
     probability (N_c / N_max)^(GAMMA - 1), GAMMA = 1.5, where N_c is the
     number of training interactions in c.
  2. Per-category positive skew: keep a positive of category c with
     probability min(1, s * (N_c / N_max)^DELTA), DELTA = 1. High-volume
     categories keep most of their positives and tail categories approach
     zero. s is solved per split by bisection so that the split's positive
     rate equals the target (a split already at or below the target is left
     unchanged).

The uniform draws of each split come from one generator seeded per split and
shared across targets, so the positives kept at a lower target are a subset
of those kept at a higher target.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from paths import SUMMARY, condition_dir, kuairec_table

SEED = 42
GAMMA = 1.5
DELTA = 1.0
STREAM = 1                                    # offset of the per-split random streams
TARGETS = {"pos05": 0.05, "pos02": 0.02, "pos01": 0.01}


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if x.sum() else float("nan")


def solve_scale(pos_w, n_neg, pi):
    """Scale s with sum(min(1, s * w)) over positives == pi * n_neg / (1 - pi)."""
    target = pi * n_neg / (1 - pi)
    if len(pos_w) <= target:
        return np.inf
    lo, hi = 1e-12, 1e12
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if np.minimum(1.0, mid * pos_w).sum() < target:
            lo = mid
        else:
            hi = mid
    return hi


def main():
    big = pd.read_parquet(kuairec_table("big"))
    m = np.random.default_rng(SEED).random(len(big)) < 0.10
    splits = {"train": big[~m].reset_index(drop=True), "val": big[m].reset_index(drop=True),
              "test": pd.read_parquet(kuairec_table("small"))}
    N = splits["train"].groupby("category").size()
    keep = (N / N.max()) ** (GAMMA - 1); floor = float(keep.min())
    wpos = (N / N.max()) ** DELTA; wfloor = float(wpos.min())
    rows, kept = [], {}
    for name, pi in TARGETS.items():
        out = condition_dir("kuairec", name); os.makedirs(out, exist_ok=True)
        for si, (split, df) in enumerate(splits.items()):
            rng = np.random.default_rng(SEED + 100 * STREAM + si)
            d = df[rng.random(len(df)) < df.category.map(keep).fillna(floor).to_numpy()]
            u = rng.random(len(d))
            lab = d.label.to_numpy() == 1
            w = d.category.map(wpos).fillna(wfloor).to_numpy()
            s = solve_scale(w[lab], int((~lab).sum()), pi)
            d = d[~(lab & (u >= np.minimum(1.0, s * w)))].reset_index(drop=True)
            d.to_parquet(f"{out}/{split}.parquet", index=False)
            kept[(name, split)] = set(zip(d.user_id[d.label == 1], d.video_id[d.label == 1]))
            g = d.groupby("category").label.agg(n="size", pos="sum").reindex(N.index).fillna(0)
            pos = g.pos.to_numpy()
            rows.append(dict(condition=name, target_pos_rate=pi, split=split, scale=s,
                             interactions=len(d), positives=int(d.label.sum()),
                             positive_rate=d.label.mean(),
                             gini_volume=gini(g.n), gini_positives=gini(g.pos),
                             categories_pos_ge100=int((pos >= 100).sum()),
                             categories_pos_ge10=int((pos >= 10).sum()),
                             categories_pos_lt10=int((pos < 10).sum())))
    for split in splits:
        a, b, c = (kept[(n, split)] for n in ("pos01", "pos02", "pos05"))
        print(f"{split}: positives nested 1% <= 2% <= 5%: {a <= b <= c}")
    T = pd.DataFrame(rows)
    os.makedirs(SUMMARY, exist_ok=True)
    T.to_csv(f"{SUMMARY}/sparse_positives_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
