"""Candidate rules and the S / E / H / R sweep over (rule, category) pairs.

The statistics come from symbolic.tristate, symbolic.evidence and
symbolic.scoring. This module supplies the data interface:

    product p        a category (the primary category of an item, or a
                     Santander product)
    outcome unit     one training interaction with an item of category p
    conversion       label == 1 (a positive interaction)
    q_0p             positive rate over all training interactions in p
                     whose user is observable under the rule
    q_rp             positive rate over the training interactions in p of
                     users satisfying rule r

A rule with high E is therefore a user condition under which interactions with
category p are less often positive than the category's own base rate.

The outcome unit is an interaction rather than a user because a user has many
interactions per category in these datasets; one bit per (user, category)
would discard most of the signal and push base rates toward 1.

Per-category counts are stored as two (n_categories x n_users) matrices, so
every (rule, category) arm is a masked sum.

Rule graph. Rules have at most two predicates, so the only edges are from an
arity-2 rule to each of its two arity-1 conjuncts. These are
predicate-extension edges; containment of both the satisfying and the
observable sets is checked explicitly. An arity-1 rule has no parent, so its H
is not estimable (status ROOT_NO_PARENT) and it cannot pass the H gate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from symbolic.evidence import beta_binomial_effect
from symbolic.scoring import (
    MIN_ARM, hierarchical_gain, node_H_from_edges, product_relevance,
)
from symbolic.tristate import (
    TriStateSignature, canonical_rule_key, eval_rule_tristate,
)

SEED = 42
DRAWS = 10_000
PERCENTILE = 10
TAU_E = 0.45          # Evidence Strength threshold; not retuned per dataset

# KuaiRec / KuaiRand profile features.
KUAI_NUMERIC_FEATURES = ["is_lowactive_period", "is_live_streamer", "is_video_author",
                         "follow_user_num", "fans_user_num", "friend_user_num",
                         "register_days"]
KUAI_CATEGORICAL_FEATURES = ["user_active_degree"]


# ---------------------------------------------------------------------------
# Feature frame and predicates
# ---------------------------------------------------------------------------

def build_user_frame(users, user_ids, numeric=None, categorical=None, behavioural=None):
    """One row per user in `user_ids`, interpretable columns only.

    numeric      numeric columns of `users` (default: KuaiRec profile features)
    categorical  categorical columns of `users`, factorized to integer codes so
                 that `col==k` predicates can be evaluated; a missing value
                 stays NaN (UNKNOWN), never a code
    behavioural  optional per-user table from the discovery split (see
                 symbolic.discovery.behavioural_from); its columns are added
                 as numeric features

    Returns (frame, codes) where codes maps each categorical column to
    {code: label}, so that rules can be reported with readable values. The
    numeric feature list is stored in frame.attrs["numeric_features"].
    """
    numeric = list(KUAI_NUMERIC_FEATURES if numeric is None else numeric)
    categorical = list(KUAI_CATEGORICAL_FEATURES if categorical is None else categorical)
    u = users.set_index("user_id")
    frame = pd.DataFrame(index=pd.Index(user_ids, name="user_id"))
    for c in numeric:
        frame[c] = pd.to_numeric(u[c].reindex(user_ids), errors="coerce").to_numpy()
    if behavioural is not None:
        b = behavioural.set_index("user_id").reindex(user_ids)
        beh_cols = list(b.columns)
        for c in beh_cols:
            frame[c] = pd.to_numeric(b[c], errors="coerce").to_numpy()
        numeric = numeric + beh_cols
    frame.attrs["numeric_features"] = numeric
    codes = {}
    for c in categorical:
        raw = u[c].reindex(user_ids).astype("string")
        cat = pd.Categorical(raw)
        codes[c] = {int(i): str(v) for i, v in enumerate(cat.categories)}
        vals = cat.codes.astype(float)
        vals[cat.codes < 0] = np.nan
        frame[c + "_code"] = vals
    return frame.reset_index(), codes


def build_predicates(frame, codes):
    """`col==0` and `col<=q25/q50/q75` for numeric columns; `col==k` for categorical.

    Quantiles are computed over the users of the discovery split. A quantile
    equal to 0 is skipped (it would restate `col==0`) and duplicate
    thresholds within a column are collapsed.
    """
    preds, meta = [], {}
    for c in frame.attrs["numeric_features"]:
        v = frame[c].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        if v.size == 0:
            continue
        seen = set()
        if (v == 0).any():
            preds.append(f"{c}==0"); seen.add(("==", 0.0))
            meta[f"{c}==0"] = {"column": c, "op": "==", "threshold": 0.0}
        for tag, qq in (("q25", 0.25), ("q50", 0.50), ("q75", 0.75)):
            t = float(np.quantile(v, qq))
            if t == 0.0 or ("<=", t) in seen:
                continue
            seen.add(("<=", t))
            p = f"{c}<={t}"
            preds.append(p)
            meta[p] = {"column": c, "op": "<=", "threshold": t, "quantile": tag}
    for c, labels in codes.items():
        col = c + "_code"
        for k, label in labels.items():
            p = f"{col}=={float(k)}"
            preds.append(p)
            meta[p] = {"column": col, "op": "==", "threshold": float(k),
                       "category_label": label, "source_column": c}
    return preds, meta


def build_candidates(frame, preds, max_arity=2):
    """Single predicates plus pairwise conjunctions across different columns.

    Two predicates on the same column are never conjoined. Canonical keys keep
    `A AND B` and `B AND A` from both entering the pool, empty or vacuous
    single predicates are dropped, and rules with identical tri-state
    signatures are merged into one (the others are kept as aliases).
    Each surviving rule receives an id R0000, R0001, ...
    """
    col_of = {p: p.split("<=")[0].split("==")[0] for p in preds}
    cands, by_key = [], {}
    for p in preds:
        obs, fire = eval_rule_tristate(frame, [p])
        if fire.sum() == 0 or fire.sum() == obs.sum():
            continue
        cands.append({"predicates": [p], "arity": 1,
                      "key": canonical_rule_key([p]), "obs": obs, "fire": fire})
    if max_arity >= 2:
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                a, b = preds[i], preds[j]
                if col_of[a] == col_of[b]:
                    continue
                obs, fire = eval_rule_tristate(frame, [a, b])
                if fire.sum() == 0:
                    continue
                cands.append({"predicates": [a, b], "arity": 2,
                              "key": canonical_rule_key([a, b]),
                              "obs": obs, "fire": fire})
    out = []
    for c in cands:
        sig = TriStateSignature.from_masks(c["obs"], c["fire"]).key()
        if sig in by_key:
            by_key[sig]["aliases"].append(c["key"])
            continue
        c["aliases"] = []
        by_key[sig] = c
        out.append(c)
    for k, c in enumerate(out):
        c["rule_id"] = f"R{k:04d}"
    return out


# ---------------------------------------------------------------------------
# S / E / H / R sweep
# ---------------------------------------------------------------------------

def _arm(N, P, ci, mask):
    return int(N[ci][mask].sum()), int(P[ci][mask].sum())


def score_all(cands, N, P, categories, seed=SEED, tau_e=TAU_E,
              log=lambda s: None):
    """Scope and E for every (rule, category); then H; then R, gated in that order.

    N[c, u] and P[c, u] are the training interaction and positive counts of
    user u in category c. H is computed only where E >= tau_e, and R only
    where H > 0, because R re-evaluates the critical edge on every other
    category.

    Returns (rules DataFrame, {rule key: list of parent candidates}).
    """
    by_key = {c["key"]: c for c in cands}
    parents_of = {}
    for c in cands:
        if c["arity"] != 2:
            parents_of[c["key"]] = []
            continue
        ps = []
        for p in c["predicates"]:
            pk = canonical_rule_key([p])
            if pk in by_key:
                par = by_key[pk]
                if (not np.any(c["fire"] & ~par["fire"])
                        and not np.any(c["obs"] & ~par["obs"])
                        and int(c["fire"].sum()) < int(par["fire"].sum())):
                    ps.append(par)
        parents_of[c["key"]] = ps

    rows = []
    for ci, cat in enumerate(categories):
        n_pop = int(N[ci].sum())
        if n_pop == 0:
            continue
        for c in cands:
            n_obs, c_obs = _arm(N, P, ci, c["obs"])
            n_r, c_r = _arm(N, P, ci, c["fire"])
            # Scope: the public pipeline only requires S > 0.
            if n_obs <= 0 or n_r <= 0:
                rows.append({"rule_id": c["rule_id"], "category": cat,
                             "arity": c["arity"],
                             "rule": " AND ".join(c["predicates"]),
                             "S": 0.0 if n_obs else None,
                             "status": "NOT_APPLICABLE",
                             "E": None, "H": None, "R": None,
                             "n_rule": n_r, "n_observable": n_obs,
                             "q_rule": None, "q_baseline": None})
                continue
            S = n_r / n_obs
            ev = beta_binomial_effect(c_r, n_r, c_obs, n_obs,
                                      draws=DRAWS, percentile=PERCENTILE,
                                      seed=seed)
            rows.append({
                "rule_id": c["rule_id"], "category": cat, "arity": c["arity"],
                "rule": " AND ".join(c["predicates"]),
                "S": S, "status": "APPLICABLE",
                "E": ev["E"], "D_median": ev["D_median"],
                "H": None, "H_status": None, "R": None, "R_status": None,
                "n_rule": n_r, "n_observable": n_obs,
                "q_rule": c_r / n_r, "q_baseline": c_obs / n_obs,
            })

    df = pd.DataFrame(rows)
    # Columns created as all-None would otherwise stay `object`.
    for c in ("S", "E", "D_median", "H", "R", "q_rule", "q_baseline"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("H_status", "R_status", "critical_parent"):
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].astype(object)
    log(f"  scored {len(df)} (rule,category) pairs; "
        f"{int((df['status']=='APPLICABLE').sum())} applicable")

    # ---- H, only where E qualifies ----------------------------------------
    e_ok = df["status"].eq("APPLICABLE") & df["E"].notna() & (df["E"] >= tau_e)
    log(f"  E>={tau_e}: {int(e_ok.sum())} pairs")
    cat_ix = {c: i for i, c in enumerate(categories)}
    cand_ix = {c["rule_id"]: c for c in cands}
    edge_draws = {}          # (rule_id, category) -> draws of the critical edge

    for idx in np.flatnonzero(e_ok.to_numpy()):
        r = df.iloc[idx]
        c = cand_ix[r["rule_id"]]
        ps = parents_of[c["key"]]
        if not ps:
            df.iat[idx, df.columns.get_loc("H_status")] = "ROOT_NO_PARENT"
            continue
        ci = cat_ix[r["category"]]
        edge_H = {}
        for par in ps:
            rem = par["fire"] & ~c["fire"]
            n_ch, c_ch = _arm(N, P, ci, c["fire"])
            n_rm, c_rm = _arm(N, P, ci, rem)
            h = hierarchical_gain(c_ch, n_ch, c_rm, n_rm, draws=DRAWS,
                                  percentile=PERCENTILE, seed=seed)
            if h.get("estimable"):
                edge_H[par["rule_id"]] = h
        if not edge_H:
            df.iat[idx, df.columns.get_loc("H_status")] = "REMAINDER_TOO_SMALL"
            continue
        crit, h_node, _ = node_H_from_edges(edge_H)
        df.iat[idx, df.columns.get_loc("H")] = h_node
        df.iat[idx, df.columns.get_loc("H_status")] = "ESTIMATED"
        df.iat[idx, df.columns.get_loc("critical_parent")] = crit
        edge_draws[(r["rule_id"], r["category"])] = (crit, edge_H[crit]["_draws"])

    # ---- R, only where H qualifies ----------------------------------------
    h_ok = e_ok & df["H"].notna() & (df["H"] > 0)
    log(f"  +H>0: {int(h_ok.sum())} pairs")
    for idx in np.flatnonzero(h_ok.to_numpy()):
        r = df.iloc[idx]
        c = cand_ix[r["rule_id"]]
        crit, tgt_draws = edge_draws[(r["rule_id"], r["category"])]
        par = cand_ix[crit]
        rem = par["fire"] & ~c["fire"]
        others = []
        for q, qcat in enumerate(categories):
            if qcat == r["category"]:
                continue
            n_ch, c_ch = _arm(N, P, q, c["fire"])
            n_rm, c_rm = _arm(N, P, q, rem)
            if n_ch < MIN_ARM or n_rm < MIN_ARM:
                continue          # dropped from the comparison, never imputed as 0
            h = hierarchical_gain(c_ch, n_ch, c_rm, n_rm, draws=DRAWS,
                                  percentile=PERCENTILE, seed=seed)
            if h.get("estimable"):
                others.append(h["_draws"])
        res = product_relevance(tgt_draws, others, percentile=PERCENTILE)
        if res.get("estimable"):
            df.iat[idx, df.columns.get_loc("R")] = res["R"]
            df.iat[idx, df.columns.get_loc("R_status")] = "ESTIMATED"
        else:
            df.iat[idx, df.columns.get_loc("R_status")] = \
                "CROSS_PRODUCT_NOT_ESTIMABLE"
    log(f"  +R>0: {int((h_ok & df['R'].notna() & (df['R'] > 0)).sum())} pairs")
    return df, parents_of


def assign_stages_and_tiers(df, tau_e=TAU_E):
    """Gate membership and evidence tiers.

    stage_A  S+E      (E >= tau_e)
    stage_B  S+E+H    (and H > 0)
    stage_C  S+E+H+R  (and R > 0)

    `tier` records E bands (0.35, 0.20, 0) for pairs outside stage_C. Tiers
    never make a rule statistically qualified; they are audit metadata.
    """
    ap = df["status"].eq("APPLICABLE")
    E = df["E"]
    df["stage_A"] = ap & E.notna() & (E >= tau_e)
    df["stage_B"] = df["stage_A"] & df["H"].notna() & (df["H"] > 0)
    df["stage_C"] = df["stage_B"] & df["R"].notna() & (df["R"] > 0)
    df["tier"] = np.where(
        df["stage_C"], "TIER1_QUALIFIED",
        np.where(ap & E.notna() & (E >= 0.35), "TIER2_MODERATE",
                 np.where(ap & E.notna() & (E >= 0.20), "TIER3_BROAD",
                          np.where(ap & E.notna() & (E >= 0.0),
                                   "TIER4_MAXIMUM", "UNTIERED"))))
    return df
