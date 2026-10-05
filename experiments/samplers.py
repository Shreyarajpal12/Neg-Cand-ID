"""Negative samplers that use symbolic evidence, and their diagnostics.

Pure array helpers; experiments.train_recommender calls them per user.

Sampler names
    random                uniform over the pool
    bns                   item-level BNS weights, 1 - positives_i / interactions_i
    bns_smooth            BNS-S, 1 - (positives_i + 0.5) / (interactions_i + 1)
    dns                   top-K of M = 20 pool candidates by the current model
    negative_symbolic     the gate-ranked negative ladder (symbolic.ladder)
    positive_symbolic     protection tiers from positive evidence, safest first
    symbolic_random       uniform inside the symbolic pool, then wider tiers
    symbolic_bns          BNS weights inside the symbolic pool, then wider tiers
    symbolic_bns_smooth   BNS-S weights inside the symbolic pool, then wider tiers
    symbolic_dns          DNS whose candidate search is restricted to the symbolic pool

The pool of a user is the whole catalogue minus that user's observed training
positives, identical for every sampler.

Symbolic pool. The negative ladder levels are 0 S+E+H+R, 1 S+E+H, 2 S+E,
3 E>=0.35, 4 E>=0.20, 5 E>=0 and 6 no rule. The symbolic layer samplers keep
the base sampler's rule and change only its candidate set: they draw first
from levels 0-2 (the symbolic pool), then from levels 3-4, then from the rest
of the pool. symbolic_dns searches levels 0-2 when that set holds at least
M = 20 items, otherwise levels 0-4, otherwise the full pool (plain DNS).

Positive evidence. E_pos (experiments.positive_rules) gives each
(user, category) a protection tier P: 0 none, 1 weak (0 < E_pos < 0.20),
2 medium (0.20 <= E_pos < 0.45), 3 strong (E_pos >= 0.45). positive_symbolic
samples tier 0 first and reaches strongly supported positive regions only
when nothing else is left. Positive rules never add positives.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

RANDOM = "random"
BNS = "bns"
BNS_SMOOTH = "bns_smooth"
DNS = "dns"
NEGATIVE_SYMBOLIC = "negative_symbolic"
POSITIVE_SYMBOLIC = "positive_symbolic"
SYMBOLIC_RANDOM = "symbolic_random"
SYMBOLIC_BNS = "symbolic_bns"
SYMBOLIC_BNS_SMOOTH = "symbolic_bns_smooth"
SYMBOLIC_DNS = "symbolic_dns"

ALL_SAMPLERS = [RANDOM, BNS, BNS_SMOOTH, DNS, NEGATIVE_SYMBOLIC, POSITIVE_SYMBOLIC,
                SYMBOLIC_RANDOM, SYMBOLIC_BNS, SYMBOLIC_BNS_SMOOTH, SYMBOLIC_DNS]

SYMBOLIC_MAX = 2          # symbolic pool: ladder levels 0-2
WIDER_MAX = 4             # second tier: ladder levels 3-4
NEG_FILL = 6
POS_EDGES = (1e-12, 0.20, 0.45)
DNS_M = 20

# Samplers whose negatives are pre-selected by `plan` + `tiered_draw`.
TIERED = {POSITIVE_SYMBOLIC, SYMBOLIC_RANDOM, SYMBOLIC_BNS, SYMBOLIC_BNS_SMOOTH}
DNS_SAMPLERS = {DNS, SYMBOLIC_DNS}
# Samplers for which pool and fallback diagnostics are written.
DIAGNOSED = TIERED | DNS_SAMPLERS
NEEDS_POSITIVE = {POSITIVE_SYMBOLIC}

# Slot tier at or beyond which a negative counts as fallback (it did not come
# from the sampler's primary pool).
FALLBACK_TIER = {
    SYMBOLIC_RANDOM: 2, SYMBOLIC_BNS: 2, SYMBOLIC_BNS_SMOOTH: 2,
    POSITIVE_SYMBOLIC: 3,
    SYMBOLIC_DNS: 2,
}
_OFFSET = 1e12   # tier separation; weighted keys stay below 1e11 with weights clipped at 1e-9


# ------------------------------------------------------------------ evidence
def positive_tiers(PE, u, cats):
    """Protection tier P (0..3) for every candidate category of user u.

    PE maps user -> {category: strongest E_pos}. Raises if the positive
    rules have not been discovered for this cell.
    """
    if PE is None:
        raise RuntimeError("positive_symbolic requires positive_levels.parquet for this cell; "
                           "run experiments.positive_rules first")
    cats = np.asarray(cats)
    m = PE.get(int(u))
    if not m:
        return np.zeros(len(cats), dtype=np.int8)
    hi = int(max(max(m), int(cats.max(initial=0)))) + 2
    lut = np.zeros(hi, dtype=float)
    for c, e in m.items():
        if 0 <= c < hi:
            lut[c] = e
    idx = np.where(cats >= 0, cats, hi - 1)          # unknown category -> no evidence
    return np.digitize(lut[idx], POS_EDGES).astype(np.int8)


def symbolic_tiers(lev):
    """0 = symbolic pool (levels 0-2), 1 = levels 3-4, 2 = everything else."""
    return np.where(lev <= SYMBOLIC_MAX, 0, np.where(lev <= WIDER_MAX, 1, 2)).astype(np.int8)


def plan(sampler, lev, P, w_bns, w_bns_smooth):
    """-> (tier per candidate, within-tier weights or None, primary-pool mask)."""
    if sampler == SYMBOLIC_RANDOM:
        return symbolic_tiers(lev), None, lev <= SYMBOLIC_MAX
    if sampler == SYMBOLIC_BNS:
        return symbolic_tiers(lev), w_bns, lev <= SYMBOLIC_MAX
    if sampler == SYMBOLIC_BNS_SMOOTH:
        return symbolic_tiers(lev), w_bns_smooth, lev <= SYMBOLIC_MAX
    if sampler == POSITIVE_SYMBOLIC:
        return P.astype(np.int8), None, P == 0
    raise KeyError(sampler)


# ------------------------------------------------------------------ drawing
def tiered_draw(rng, a, K, tier, w=None):
    """(a x K) distinct candidate indices per positive.

    Every candidate in a lower tier outranks every candidate in a higher tier.
    Within a tier the draw is uniform (w is None) or weighted without
    replacement with exponential keys, as in symbolic.ladder.distinct_draw.
    """
    n = len(tier); k = min(K, n)
    if w is None:
        keys = rng.random((a, n))
    else:
        keys = rng.exponential(size=(a, n)) / np.clip(np.asarray(w, float), 1e-9, None)[None, :]
    keys = keys + tier.astype(np.float64)[None, :] * _OFFSET
    return np.argpartition(keys, k - 1, axis=1)[:, :k]


def dns_pool(sampler, pool, lev, M):
    """Candidate pool for symbolic_dns: the smallest symbolic set holding >= M items.

    Returns (candidates, relaxation level): 0 = levels 0-2, 1 = levels 0-4,
    2 = full pool.
    """
    if sampler == SYMBOLIC_DNS:
        for lvl, m in ((0, lev <= SYMBOLIC_MAX), (1, lev <= WIDER_MAX)):
            if m.sum() >= M:
                return pool[m], lvl
        return pool, 2
    raise KeyError(sampler)


def slot_labels(tier, P, lev):
    """Per-slot label 't<tier>|p<positive tier>|n<negative level>' for candidate files."""
    return np.array([f"t{a}|p{b}|n{c}" for a, b, c in zip(tier, P, lev)], dtype=object)


# ------------------------------------------------------------------ diagnostics
class Diagnostics:
    """Per-sampler candidate-pool and fallback statistics. No labels are used."""
    def __init__(self):
        self.d = {}

    def start(self, s):
        self.d[s] = dict(users=0, users_primary_short=0, slots=0, pool_sum=0, primary_sum=0,
                         cov_sum=0.0, remain_sum=0, tier=np.zeros(32, np.int64),
                         slotP=np.zeros(4, np.int64), poolP=np.zeros(4, np.int64),
                         dns_level=np.zeros(8, np.int64), dns_anchor_level=np.zeros(8, np.int64))

    def add(self, s, pool_n, primary, K, slot_tier, slot_P, pool_P):
        r = self.d[s]
        pn = int(primary.sum())
        r["users"] += 1; r["pool_sum"] += pool_n; r["primary_sum"] += pn
        r["cov_sum"] += pn / max(pool_n, 1); r["users_primary_short"] += int(pn < K)
        r["remain_sum"] += int((pool_P < 3).sum())
        r["slots"] += slot_tier.size
        r["tier"] += np.bincount(slot_tier.reshape(-1).astype(int), minlength=32)[:32]
        r["slotP"] += np.bincount(slot_P.reshape(-1).astype(int), minlength=4)[:4]
        r["poolP"] += np.bincount(pool_P.astype(int), minlength=4)[:4]

    def add_dns(self, s, pool_n, cand_n, n_anchor, level, K, pool_P=None):
        r = self.d[s]
        r["users"] += 1; r["pool_sum"] += pool_n; r["primary_sum"] += cand_n
        r["cov_sum"] += cand_n / max(pool_n, 1)
        r["users_primary_short"] += int(level > 0)
        r["slots"] += n_anchor * K
        r["dns_level"][level] += 1; r["dns_anchor_level"][level] += n_anchor
        r["tier"][level] += n_anchor * K
        if pool_P is not None:
            r["poolP"] += np.bincount(pool_P.astype(int), minlength=4)[:4]

    def summary(self, s):
        r = self.d.get(s)
        if not r or r["users"] == 0:
            return {}
        fbt = FALLBACK_TIER.get(s)
        tiers = np.arange(32)
        pool_tot = max(r["poolP"].sum(), 1); slots = max(r["slots"], 1)
        out = dict(
            sampled_negatives=int(r["slots"]),
            users=int(r["users"]),
            mean_pool_size=r["pool_sum"] / r["users"],
            mean_safe_pool_size=r["primary_sum"] / r["users"],
            symbolic_pool_coverage=r["cov_sum"] / r["users"],
            users_primary_pool_short_frac=r["users_primary_short"] / r["users"],
            fallback_fraction=(float(r["tier"][tiers >= fbt].sum()) / slots) if fbt is not None else np.nan,
            mean_symbolic_tier=float((r["tier"] * tiers).sum()) / slots,
            tier_hist="|".join(str(int(x)) for x in r["tier"][:max(8, int(np.flatnonzero(r["tier"]).max(initial=-1)) + 1)]),
        )
        if r["poolP"].sum() > 0:
            out.update(
                pct_candidates_pos_evidence=float(r["poolP"][1:].sum()) / pool_tot,
                pct_candidates_strong_pos=float(r["poolP"][3]) / pool_tot,
                mean_candidates_after_strong_removal=r["remain_sum"] / r["users"] if r["remain_sum"] else np.nan,
            )
        if r["slotP"].sum() > 0:
            out.update(
                pct_sampled_pos_evidence=float(r["slotP"][1:].sum()) / slots,
                pct_sampled_strong_pos=float(r["slotP"][3]) / slots,
            )
        if r["dns_level"].sum() > 0:
            out["dns_user_level_hist"] = "|".join(str(int(x)) for x in r["dns_level"])
        return out


# ------------------------------------------------------------------ DNS logging
def dns_select_logged(dns, ep, score_fn, store, nI, ref_M=20):
    """Call DNSSelector.select unchanged and record what it picked.

    At each refresh this records the score of every picked negative,
    re-derives the pick as select() does and asserts that it matches, and
    measures hardness as the percentile of each picked negative's score among
    ref_M reference items scored by the same model at the same moment:

      hardness_percentile     references drawn uniformly from the catalogue
      hardness_pct_eligible   references drawn from the sampler's own pool

    Both reference draws use their own random generators (store["ref_rng"],
    store["ref_rng2"]), so the selector's random stream and picks are
    unchanged. Per-pick percentiles and scores are kept for the
    contamination-by-hardness analysis.
    """
    rec = {}

    def wrapped(u, i):
        s = score_fn(u, i); rec["i"] = i; rec["s"] = s
        return s
    fresh = dns.cache is None or (ep - dns.epoch) >= dns.refresh_every
    nu, ni = dns.select(ep, wrapped)
    if fresh and "s" in rec:
        n, M, K = len(dns.au), dns.M, dns.K
        s = rec["s"].reshape(n, M); cand = rec["i"].reshape(n, M)
        top = np.argpartition(-s, min(K, M - 1) - 1, axis=1)[:, :K]
        ps = np.take_along_axis(s, top, axis=1)
        pk = np.take_along_axis(cand, top, axis=1)
        assert np.array_equal(pk.reshape(-1), np.asarray(ni)), "logged pick != DNS pick"
        ref = store["ref_rng"].integers(0, nI, size=(n, ref_M))
        rs = score_fn(np.repeat(dns.au, ref_M), ref.reshape(-1)).reshape(n, ref_M)
        pct = (rs[:, None, :] < ps[:, :, None]).mean(axis=2)
        ref_e = dns.pool_fn(dns.au, ref_M, store["ref_rng2"])
        rse = score_fn(np.repeat(dns.au, ref_M), ref_e.reshape(-1)).reshape(n, ref_M)
        pct_e = (rse[:, None, :] < ps[:, :, None]).mean(axis=2)
        store["epochs"].append(dict(
            epoch=int(ep), picked_mean_score=float(ps.mean()),
            candidate_mean_score=float(s.mean()),
            hardness_percentile=float(pct.mean()),
            picked_median_score=float(np.median(ps)),
            hardness_percentile_median=float(np.median(pct)),
            hardness_pct_eligible=float(pct_e.mean()),
            hardness_pct_eligible_median=float(np.median(pct_e))))
        store["pct_cat"][int(ep)] = pct.reshape(-1).astype(np.float32)
        store["pct_elig"][int(ep)] = pct_e.reshape(-1).astype(np.float32)
        store["score"][int(ep)] = ps.reshape(-1).astype(np.float32)
        store["picks"][int(ep)] = np.asarray(ni, dtype=np.int64)
    return nu, ni


def write_diagnostics(path, row):
    """Append one (sampler, seed) row to a diagnostics CSV, replacing an older row."""
    df = pd.DataFrame([row])
    if os.path.exists(path) and os.path.getsize(path) > 0:
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.drop_duplicates(["sampler", "seed"], keep="last").to_csv(path, index=False)
