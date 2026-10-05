"""MIND-small: impression-level click data with temporal splits.

    python -m prep.mind

Input:  data/mind/MINDsmall_train/{behaviors.tsv, news.tsv}
Output: data/mind/prepared/{train, val, test}.parquet and users.pkl

Task: predict a click on a (user, article) pair that was shown in an
impression. Non-clicks are observed negatives (shown but not clicked).

Splits: a single temporal split of behaviors.tsv by impression timestamp,
70 / 15 / 15. Only MINDsmall_train is used, so validation and test are
distinct time windows of that file.

User features (users.pkl) come from the `history` field only, which is the
user's reading before the impression window, never from the labels being
predicted: hist_len, hist_cats, top_cat (categorical), top_cat_share,
cat_entropy. The category of an article is its MIND top-level category.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from paths import DATA

M = f"{DATA}/mind"; OUT = f"{M}/prepared"
NEWS = ["nid", "cat", "subcat", "title", "abstract", "url", "tent", "aent"]
BEH = ["impid", "user", "time", "history", "impressions"]


def main():
    os.makedirs(OUT, exist_ok=True)
    news = pd.read_csv(f"{M}/MINDsmall_train/news.tsv", sep="\t", header=None, names=NEWS,
                       usecols=["nid", "cat", "subcat"])
    cidx = {c: i for i, c in enumerate(sorted(news.cat.unique()))}
    n2c = dict(zip(news.nid, news.cat.map(cidx)))
    nidx = {n: i for i, n in enumerate(sorted(news.nid))}
    print(f"news: {len(nidx):,} articles, {len(cidx)} categories")

    b = pd.read_csv(f"{M}/MINDsmall_train/behaviors.tsv", sep="\t", header=None, names=BEH)
    b["ts"] = pd.to_datetime(b.time, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    b = b.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    q70, q85 = b.ts.quantile(0.70), b.ts.quantile(0.85)
    b["split"] = np.where(b.ts <= q70, "train", np.where(b.ts <= q85, "val", "test"))
    print(f"temporal cuts: train <= {q70}  val <= {q85}  test > {q85}")
    print(b.split.value_counts().to_dict())

    uidx = {u: i for i, u in enumerate(sorted(b.user.unique()))}
    rows = []
    for u, imp, sp in zip(b.user, b.impressions, b.split):
        ui = uidx[u]
        for tok in imp.split():
            nid, lab = tok.rsplit("-", 1)
            v = nidx.get(nid); c = n2c.get(nid)
            if v is None or c is None:
                continue
            rows.append((ui, v, int(lab), c, sp))
    d = pd.DataFrame(rows, columns=["user_id", "video_id", "label", "category", "split"])

    feat = []
    hist = b.dropna(subset=["history"]).drop_duplicates("user")
    for u, h in zip(hist.user, hist.history):
        arts = h.split(); cs = [n2c[a] for a in arts if a in n2c]
        vc = pd.Series(cs).value_counts() if cs else pd.Series(dtype=int)
        feat.append(dict(user_id=uidx[u], hist_len=len(arts), hist_cats=int(len(vc)),
                         top_cat=int(vc.index[0]) if len(vc) else -1,
                         top_cat_share=float(vc.iloc[0] / len(cs)) if cs else 0.0,
                         cat_entropy=float(-(vc / len(cs) * np.log(vc / len(cs) + 1e-12)).sum()) if cs else 0.0))
    U = pd.DataFrame(feat)
    miss = [i for i in uidx.values() if i not in set(U.user_id)]
    if miss:
        U = pd.concat([U, pd.DataFrame({"user_id": miss, "hist_len": 0, "hist_cats": 0,
                                        "top_cat": -1, "top_cat_share": 0.0, "cat_entropy": 0.0})],
                      ignore_index=True)
    U.sort_values("user_id").to_pickle(f"{OUT}/users.pkl")

    for name in ("train", "val", "test"):
        s = d[d.split == name][["user_id", "video_id", "label", "category"]].reset_index(drop=True)
        s.to_parquet(f"{OUT}/{name}.parquet", index=False)
        print(f"{name}: rows={len(s):,} users={s.user_id.nunique():,} items={s.video_id.nunique():,} "
              f"pos={int(s.label.sum()):,} rate={s.label.mean()*100:.2f}%")
    tr = d[d.split == "train"]; te = d[d.split == "test"]
    print(f"\nusers in both train and test: {len(set(tr.user_id) & set(te.user_id)):,}")
    mg = tr.merge(U, on="user_id", how="left")
    print("\nmean user feature by training label (leakage check):")
    print(mg.groupby("label")[["hist_len", "hist_cats", "top_cat_share", "cat_entropy"]].mean()
          .to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
