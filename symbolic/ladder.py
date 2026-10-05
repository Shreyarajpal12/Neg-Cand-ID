"""Gate-ranked negative ladder: the tiered negative budget used by the samplers.

Each (user, category) receives the best level of any rule that nominates it.
Levels follow the order in which the gates are applied:

    0  S+E+H+R   passed every statistical gate
    1  S+E+H     passed E and H, failed R
    2  S+E       passed E, failed H
    3  E>=0.35   failed the E threshold, moderate evidence
    4  E>=0.20   broad evidence
    5  E>=0      non-negative evidence only
    6  fill      no rule nominates this (user, category)

The Negative Symbolic sampler fills each positive's K negatives from level 0
downward and draws uniformly within a level. Every positive receives exactly K
distinct negatives from the same pool that the other samplers use.
"""
from __future__ import annotations

import numpy as np

LEVEL_NAMES = ["S+E+H+R", "S+E+H", "S+E", "E>=0.35", "E>=0.20", "E>=0", "random fill"]
FILL = 6


def gate_level_table(rules):
    """(rule_id, category, level, E[, rule]) for every applicable pair with level >= 0."""
    r = rules[rules["status"] == "APPLICABLE"].copy()
    A = (r["stage_A"] == True).to_numpy(); B = (r["stage_B"] == True).to_numpy()
    C = (r["stage_C"] == True).to_numpy(); E = r["E"].astype(float).to_numpy()
    r["level"] = np.select([C, B & ~C, A & ~B, (E >= .35) & ~A,
                            (E >= .20) & (E < .35), (E >= 0) & (E < .20)],
                           [0, 1, 2, 3, 4, 5], default=-1)
    # The rule string is kept so that Santander can re-evaluate each rule on
    # the held-out transition's features.
    cols = ["rule_id", "category", "level", "E"] + (["rule"] if "rule" in r.columns else [])
    return r[r["level"] >= 0][cols].reset_index(drop=True)


def user_category_rank(glt, fire, users, all_cats):
    """BL[u, c] = best level of any rule nominating (u, c); BE[u, c] = that rule's E.

    Returns (BL, BE, cat2col), where cat2col[c + 1] is the column of category c.
    """
    cats = np.unique(np.asarray(all_cats)[np.asarray(all_cats) >= 0]).astype(np.int64)
    cat2col = np.full(int(cats.max()) + 2, -1, dtype=np.int64)
    cat2col[cats + 1] = np.arange(len(cats))
    BL = np.full((len(users), len(cats)), FILL, dtype=np.int8)
    BE = np.zeros((len(users), len(cats)), dtype=np.float32)
    for row in glt.sort_values(["level", "E"], ascending=[True, False]).itertuples():
        c = int(row.category)
        if c + 1 >= len(cat2col) or cat2col[c + 1] < 0 or row.rule_id not in fire:
            continue
        j = cat2col[c + 1]; f = fire[row.rule_id]
        better = f & ((row.level < BL[:, j]) | ((row.level == BL[:, j]) & (row.E > BE[:, j])))
        BL[better, j] = row.level; BE[better, j] = row.E
    return BL, BE, cat2col


def item_levels(pool_cat, blr, ber, cat2col):
    """Ladder level and E of every pool item for one user (row blr / ber)."""
    pc = np.asarray(pool_cat, dtype=np.int64)
    ok = (pc + 1 >= 0) & (pc + 1 < len(cat2col))
    col = np.full(len(pc), -1, dtype=np.int64); col[ok] = cat2col[pc[ok] + 1]
    lev = np.full(len(pc), FILL, dtype=np.int8); Ev = np.zeros(len(pc), dtype=np.float32)
    if blr is not None:
        h = col >= 0
        lev[h] = blr[col[h]]; Ev[h] = ber[col[h]]
    return lev, Ev


def ladder_pick(rng, pool, pool_cat, blr, ber, cat2col, a, K):
    """(a x k) distinct picks per positive, best gate level first, uniform within a level.

    Returns (items, levels, E) arrays of shape (a, k) with k = min(K, |pool|).
    """
    pool = np.asarray(pool); k = min(K, len(pool))
    lev, Ev = item_levels(pool_cat, blr, ber, cat2col)
    cut = np.partition(lev, k - 1)[k - 1]
    m = lev <= cut
    sub, sl, se = pool[m], lev[m], Ev[m]
    base = sl.astype(np.float64) * 10.0
    keys = base[None, :] + rng.random((a, len(sub))) * 1e-7
    idx = np.argpartition(keys, k - 1, axis=1)[:, :k]
    idx = np.take_along_axis(idx, np.argsort(np.take_along_axis(keys, idx, 1), axis=1), 1)
    return sub[idx], sl[idx], se[idx]


def distinct_draw(rng, pool, a, K, w=None):
    """(a x k) distinct picks per positive; weighted without replacement when w is given.

    Weighted draws use exponential keys divided by the weights (Efraimidis and
    Spirakis), so each row is a weighted sample without replacement.
    """
    pool = np.asarray(pool); k = min(K, len(pool))
    if w is None:
        keys = rng.random((a, len(pool)))
    else:
        keys = rng.exponential(size=(a, len(pool))) / np.clip(np.asarray(w, float), 1e-12, None)[None, :]
    return pool[np.argpartition(keys, k - 1, axis=1)[:, :k]]
