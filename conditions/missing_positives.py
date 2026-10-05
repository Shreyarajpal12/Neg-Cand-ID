"""KuaiRec 5% + MNAR-75: hide 75% of the training positives, missing not at random.

    python -m conditions.missing_positives

Input:  data/conditions/kuairec/pos05 (conditions.sparse_positives)
Output: data/conditions/kuairec/pos05_mnar75/{train, val, test}.parquet,
        labels_clean.parquet, and results/summary/missing_positives_stats.csv

Only the training split is changed; validation and test are copied, so the
task and the evaluation distribution are the same as in pos05.

A training positive (u, i) is hidden with probability proportional to
(volume_i / max_j volume_j)^ALPHA, ALPHA = 0.25, capped at CAP = 0.95, where
volume_i is the number of training interactions of item i. The scale is
solved by bisection so that 75% of the positives are hidden in expectation.
The mild tilt toward high-volume items corrupts the item-level positive rates
that BNS estimates without removing all positives of the head items.

A hidden positive receives label 0 and a watch_ratio redrawn from the
training negatives' watch_ratio values, so it is indistinguishable from an
observed negative on every feature available to the samplers (watch_ratio
feeds the user feature `mean_watch_ratio_train`). The true labels are written
to labels_clean.parquet, which no sampler or rule reads; only the
candidate-quality analysis (analysis.negative_precision) uses it.

Rules are re-discovered on the corrupted training split. The random stream is
seeded from the tag "p05_mnar75".
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import pandas as pd

from paths import SUMMARY, condition_dir

SEED = 42
BASE = condition_dir("kuairec", "pos05")
NAME = "pos05_mnar75"
SEED_TAG = "p05_mnar75"
RHO = 0.75
ALPHA = 0.25
CAP = 0.95


def tag_seed(tag):
    """Deterministic seed from a tag string (independent of Python's hash())."""
    return SEED + sum(ord(c) * (i + 1) for i, c in enumerate(tag)) % 100000


def solve_scale(w, target, cap):
    lo, hi = 1e-12, 1e12
    for _ in range(300):
        mid = np.sqrt(lo * hi)
        if np.minimum(cap, mid * w).sum() < target:
            lo = mid
        else:
            hi = mid
    return hi


def main():
    tr = pd.read_parquet(f"{BASE}/train.parquet")
    vol = tr.groupby("video_id").size()
    negwr = tr.loc[tr.label == 0, "watch_ratio"].to_numpy()
    head = vol.sort_values(ascending=False).head(50).index

    out = condition_dir("kuairec", NAME); os.makedirs(out, exist_ok=True)
    for s in ("val", "test"):
        shutil.copy(f"{BASE}/{s}.parquet", f"{out}/{s}.parquet")
    d = tr.copy(); clean = d.label.to_numpy().copy()
    pos = clean == 1; npos = int(pos.sum())
    rng = np.random.default_rng(tag_seed(SEED_TAG)); u = rng.random(len(d))
    w = (d.video_id.map(vol).fillna(1).to_numpy().astype(float) / vol.max()) ** ALPHA
    c = solve_scale(w[pos], RHO * npos, CAP)
    p = np.minimum(CAP, c * w)
    drop = pos.copy(); drop[pos] = u[pos] < p[pos]
    lab = d.label.to_numpy().copy(); lab[drop] = 0; d["label"] = lab
    wr = d.watch_ratio.to_numpy().copy()
    wr[drop] = rng.choice(negwr, size=int(drop.sum()), replace=True)
    d["watch_ratio"] = wr
    pd.DataFrame({"label_clean": clean}).to_parquet(f"{out}/labels_clean.parquet", index=False)
    d.to_parquet(f"{out}/train.parquet", index=False)
    leak = int(((d.watch_ratio >= 1.0) & (d.label == 0)).sum())
    if leak:
        print(f"LEAK DETECTED in {NAME}: {leak} rows"); sys.exit(1)

    g = d.assign(label_clean=clean).groupby("video_id").agg(pos_obs=("label", "sum"),
                                                            pos_true=("label_clean", "sum"))
    ih = g.index.isin(head)
    T = pd.DataFrame([dict(
        condition=NAME, rho=RHO, pos_true=npos, pos_kept=int(d.label.sum()),
        drop_frac=drop.sum() / npos, obs_rate=d.label.mean(),
        head50_drop=1 - g.loc[ih, "pos_obs"].sum() / max(g.loc[ih, "pos_true"].sum(), 1),
        tail_drop=1 - g.loc[~ih, "pos_obs"].sum() / max(g.loc[~ih, "pos_true"].sum(), 1),
        head_items_zeroed=int((g.loc[ih, "pos_obs"] == 0).sum()),
        items_lt10_obs=int((g.pos_obs < 10).sum()), items_lt10_true=int((g.pos_true < 10).sum()))])
    os.makedirs(SUMMARY, exist_ok=True)
    T.to_csv(f"{SUMMARY}/missing_positives_stats.csv", index=False)
    pd.set_option("display.width", 230)
    print(T.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nbase: rows={len(tr):,} positives={int(tr.label.sum()):,} rate={tr.label.mean():.4f}; "
          f"val/test copied unchanged")


if __name__ == "__main__":
    main()
