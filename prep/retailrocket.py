"""RetailRocket: active visitors, purchase labels and temporal splits.

    python -m prep.retailrocket

Input:  data/retailrocket/{events.csv, item_properties_part1.csv,
        item_properties_part2.csv, category_tree.csv}
Output: data/retailrocket/prepared/{train, val, test}.parquet and users.pkl

  * Active visitors: visitors with at least 5 events (81,620 visitors). Most
    visitors in the full log have a single event, which leaves nothing to
    condition a rule on.
  * Label: a (visitor, item) pair is positive if it ever reaches a
    `transaction` event. The furthest funnel stage (0 view, 1 add-to-cart,
    2 transaction) is kept in the `evidence` column; it is not a feature.
  * Category: the latest `categoryid` of each item from the item properties;
    items without a category are dropped. Leaf categories are mapped to their
    immediate parent in category_tree.csv (268 parent categories; a category
    without a parent keeps its own id) and renumbered densely. After this
    mapping, purchases are 3.10% of the training interactions.
  * Splits: temporal on the first event time of each pair, 70 / 15 / 15.
  * User features: computed from a strictly earlier event window (events up
    to the 40th percentile of event time), so they contain no information
    from the pairs being scored: h_events, h_items, h_cats, h_top_cat
    (categorical), h_top_share. The script prints the mean of each feature
    for buyers and non-buyers as a leakage check.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from paths import DATA

D = f"{DATA}/retailrocket"; OUT = f"{D}/prepared"
MIN_EVENTS = 5


def main():
    os.makedirs(OUT, exist_ok=True)
    cats = []
    for p in ("item_properties_part1.csv", "item_properties_part2.csv"):
        for ch in pd.read_csv(f"{D}/{p}", chunksize=2_000_000):
            c = ch[ch.property == "categoryid"][["timestamp", "itemid", "value"]]
            if len(c):
                cats.append(c)
    cat = pd.concat(cats).sort_values("timestamp").drop_duplicates("itemid", keep="last")
    i2c = {int(i): int(v) for i, v in zip(cat.itemid, pd.to_numeric(cat.value, errors="coerce").fillna(-1))
           if v == v and v >= 0}

    e = pd.read_csv(f"{D}/events.csv", usecols=["timestamp", "visitorid", "event", "itemid"])
    g = e.groupby("visitorid").size(); active = set(g[g >= MIN_EVENTS].index)
    e = e[e.visitorid.isin(active)]
    e["category"] = e.itemid.map(i2c)
    e = e.dropna(subset=["category"]); e["category"] = e.category.astype(int)
    # Leaf category -> immediate parent, then dense renumbering in sorted id order.
    tree = pd.read_csv(f"{D}/category_tree.csv")
    parent = {int(c): int(p) for c, p in zip(tree.categoryid, tree.parentid) if p == p}
    e["category"] = e.category.map(lambda c: parent.get(c, c))
    e["category"] = e.category.rank(method="dense").astype(int) - 1
    print(f"active visitors {len(active):,}; events with category {len(e):,}")

    rank = {"view": 0, "addtocart": 1, "transaction": 2}
    e["ev"] = e.event.map(rank)
    pair = e.groupby(["visitorid", "itemid"]).agg(ts=("timestamp", "min"), evidence=("ev", "max"),
                                                  category=("category", "first")).reset_index()
    pair["label"] = (pair.evidence == 2).astype(np.int8)
    q70, q85 = pair.ts.quantile(0.70), pair.ts.quantile(0.85)
    pair["split"] = np.where(pair.ts <= q70, "train", np.where(pair.ts <= q85, "val", "test"))
    uidx = {u: i for i, u in enumerate(sorted(pair.visitorid.unique()))}
    iidx = {i: j for j, i in enumerate(sorted(pair.itemid.unique()))}
    pair["user_id"] = pair.visitorid.map(uidx); pair["video_id"] = pair.itemid.map(iidx)

    # User features from the earlier event window only.
    _e = pd.read_csv(f"{D}/events.csv", usecols=["timestamp", "visitorid", "event", "itemid"])
    _g = _e.groupby("visitorid").size(); _e = _e[_e.visitorid.isin(set(_g[_g >= MIN_EVENTS].index))]
    _cut = _e.timestamp.quantile(0.40)
    _h = _e[_e.timestamp <= _cut].copy()
    _h["user_id"] = _h.visitorid.map(uidx); _h["category"] = _h.itemid.map(i2c)
    _h = _h.dropna(subset=["user_id"])
    f = _h.groupby("user_id").agg(h_events=("itemid", "size"), h_items=("itemid", "nunique"),
                                  h_cats=("category", "nunique"))
    _top = _h.dropna(subset=["category"]).groupby(["user_id", "category"]).size().reset_index(name="n")
    _top = _top.sort_values("n", ascending=False).drop_duplicates("user_id").set_index("user_id")
    f["h_top_cat"] = _top.category; f["h_top_share"] = _top.n / f.h_events
    U = pd.DataFrame({"user_id": sorted(uidx.values())}).merge(f.reset_index(), on="user_id", how="left")
    U = U.fillna({"h_events": 0, "h_items": 0, "h_cats": 0, "h_top_cat": -1, "h_top_share": 0.0})
    print(f"feature window ends {pd.to_datetime(_cut, unit='ms')}; events in it: {len(_h):,}")
    U.to_pickle(f"{OUT}/users.pkl")

    for name in ("train", "val", "test"):
        s = pair[pair.split == name][["user_id", "video_id", "label", "category", "evidence"]].reset_index(drop=True)
        s.to_parquet(f"{OUT}/{name}.parquet", index=False)
        print(f"{name}: rows={len(s):,} users={s.user_id.nunique():,} items={s.video_id.nunique():,} "
              f"pos={int(s.label.sum()):,} rate={s.label.mean()*100:.3f}%  "
              f"cart-or-better={float((s.evidence>=1).mean())*100:.2f}%")
    print(f"\ncategories: {pair.category.nunique():,}; user frame {len(U):,} rows, cols {list(U.columns)}")
    mg = pair[pair.split == "train"].merge(U, on="user_id", how="left")
    print("\nmean user feature by training label (leakage check):")
    print(mg.groupby("label")[["h_events", "h_items", "h_cats", "h_top_share"]].mean()
          .to_string(float_format=lambda x: f"{x:.3f}"))
    _r = mg.groupby("label")[["h_events", "h_items", "h_cats", "h_top_share"]].mean()
    _ratio = (_r.loc[1] / _r.loc[0].replace(0, np.nan))
    print("\nbuyer / non-buyer ratio (close to 1 means no leakage):",
          {k: round(float(v), 2) for k, v in _ratio.items()})


if __name__ == "__main__":
    main()
