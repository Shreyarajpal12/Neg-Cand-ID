"""Frozen rule discovery and the budget-matched candidate-quality comparison.

Rules are discovered on a discovery log, frozen, and only then applied to an
evaluation log whose outcomes the discovery step never sees:

    KuaiRec   discovery = big matrix        evaluation = small matrix
    KuaiRand  discovery = standard log      evaluation = random-exposure log
    others    discovery = training split    evaluation = test split

Candidate quality is intrinsic: the symbolic selection is every evaluation
pair that a fully qualified (S+E+H+R) rule nominates, with no budget K and no
fallback. Every comparator receives exactly the same number of candidates.

The category-matched control draws, for each category, as many pairs as the
symbolic selection took from that category, uniformly among that category's
evaluation pairs. A remaining advantage therefore cannot be explained by the
selection simply concentrating on categories with low positive rates. Its
empirical p-value is the add-one fraction of replicates whose negative
precision reaches the symbolic value, so the smallest reportable value is
1 / (R + 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from symbolic.rules import (
    KUAI_CATEGORICAL_FEATURES, KUAI_NUMERIC_FEATURES, assign_stages_and_tiers,
    build_candidates, build_predicates, build_user_frame, score_all,
)

SEED = 42


def behavioural_from(disc, label_col="label"):
    """Per-user activity features computed from the discovery log only.

    `mean_watch_ratio_train` is added only when the log has a watch ratio; on a
    click log it would duplicate `train_positive_rate`.
    """
    g = disc.groupby("user_id")
    b = pd.DataFrame({
        "n_train_interactions": g.size(),
        "n_train_positives": g[label_col].sum(),
        "train_positive_rate": g[label_col].mean(),
        "n_distinct_categories": g["category"].nunique(),
    })
    if "watch_ratio" in disc.columns:
        b["mean_watch_ratio_train"] = g["watch_ratio"].mean()
    b.index.name = "user_id"
    return b.reset_index()


def user_feature_lists(users_raw):
    """(numeric, categorical) user-feature columns for a user table.

    KuaiRec and KuaiRand use their published profile features. For the other
    prepared datasets every column ending in `top_cat` is categorical and the
    remaining columns are numeric.
    """
    if set(KUAI_NUMERIC_FEATURES) <= set(users_raw.columns):
        return list(KUAI_NUMERIC_FEATURES), list(KUAI_CATEGORICAL_FEATURES)
    cols = [c for c in users_raw.columns if c != "user_id"]
    cat_cols = [c for c in cols if c.endswith("top_cat")]
    return [c for c in cols if c not in cat_cols], cat_cols


def candidate_rules(disc, users_raw, log=print):
    """User frame, predicates and candidate rules for a discovery log.

    Returns (frame, codes, predicates, predicate_meta, candidates, user_ids).
    The same function is used by negative discovery, positive discovery and
    the downstream samplers, so rule ids agree across them.
    """
    user_ids = np.sort(disc["user_id"].unique())
    beh = behavioural_from(disc)
    numeric, categorical = user_feature_lists(users_raw)
    frame, codes = build_user_frame(users_raw, user_ids, numeric=numeric,
                                    categorical=categorical, behavioural=beh)
    preds, meta = build_predicates(frame, codes)
    cands = build_candidates(frame, preds, max_arity=2)
    log(f"  {len(preds)} predicates -> {len(cands)} candidate rules "
        f"(arity1={sum(c['arity']==1 for c in cands)}, arity2={sum(c['arity']==2 for c in cands)})")
    return frame, codes, preds, meta, cands, user_ids


def count_matrices(disc, user_ids, categories):
    """N[c, u] = interactions of user u in category c; P[c, u] = positives."""
    uix = {u: i for i, u in enumerate(user_ids)}
    cix = {c: i for i, c in enumerate(categories)}
    N = np.zeros((len(categories), len(user_ids)), dtype=np.int64)
    P = np.zeros((len(categories), len(user_ids)), dtype=np.int64)
    ui = disc["user_id"].map(uix).to_numpy()
    ci = disc["category"].map(cix).to_numpy()
    ok = ~(pd.isna(ui) | pd.isna(ci))
    ui, ci = ui[ok].astype(np.int64), ci[ok].astype(np.int64)
    lab = disc["label"].to_numpy()[ok].astype(np.int64)
    np.add.at(N, (ci, ui), 1); np.add.at(P, (ci, ui), lab)
    return N, P


def discover(disc, users_raw, log=print):
    """Full negative-rule discovery on the discovery log only.

    Returns (rules, candidates, frame, user_ids, predicate_meta, codes).
    """
    frame, codes, preds, meta, cands, user_ids = candidate_rules(disc, users_raw, log=log)
    categories = sorted(int(c) for c in disc["category"].unique() if c >= 0)
    N, P = count_matrices(disc, user_ids, categories)
    rules, _ = score_all(cands, N, P, categories, seed=SEED, log=log)
    rules = assign_stages_and_tiers(rules)
    for s in ("stage_A", "stage_B", "stage_C"):
        log(f"  {s}: {int(rules[s].sum())} pairs, "
            f"{rules.loc[rules[s],'rule_id'].nunique()} rules, "
            f"{rules.loc[rules[s],'category'].nunique()} categories")
    return rules, cands, frame, user_ids, meta, codes


def user_category_nominations(rules, cands, user_ids, stage_col="stage_C"):
    """(user_id, category) -> (rule_id, E) for rules that pass `stage_col`.

    When several rules nominate the same pair, the one with the highest E is kept.
    """
    fire = {c["rule_id"]: c["fire"] for c in cands}
    best = {}
    sub = rules[rules[stage_col] == True].sort_values("E", ascending=False)
    for r in sub.itertuples():
        for u in user_ids[fire[r.rule_id]]:
            k = (int(u), int(r.category))
            if k not in best:
                best[k] = (r.rule_id, float(r.E))
    return best


# ---------------------------------------------------------------------------
# Candidate quality
# ---------------------------------------------------------------------------

def symbolic_select(eval_df, nominations):
    """Every evaluation pair whose (user, category) is nominated. No K, no fallback."""
    keys = pd.MultiIndex.from_arrays([eval_df["user_id"], eval_df["category"]])
    idx = pd.MultiIndex.from_tuples(list(nominations.keys())) if nominations else \
        pd.MultiIndex.from_tuples([], names=["u", "c"])
    hit = keys.isin(idx)
    sel = eval_df.loc[hit].copy()
    sel["rule_id"] = [nominations[(int(u), int(c))][0]
                      for u, c in zip(sel["user_id"], sel["category"])]
    return sel


def precision(labels):
    labels = np.asarray(labels)
    return float(1.0 - labels.mean()) if len(labels) else float("nan")


def matched_replicates(eval_df, sel, keys, rng, R):
    """Draw the symbolic per-group counts uniformly within each group, R times."""
    counts = sel.groupby(keys, sort=False).size()
    pools, sizes = [], []
    grouped = eval_df.groupby(keys, sort=False).indices
    for k, n in counts.items():
        idx = grouped.get(k)
        if idx is None or len(idx) == 0:
            continue
        pools.append(idx); sizes.append(min(int(n), len(idx)))
    y = eval_df["label"].to_numpy()
    out = np.empty(R)
    for r in range(R):
        picks = [rng.choice(p, size=s, replace=False) for p, s in zip(pools, sizes)]
        out[r] = precision(y[np.concatenate(picks)]) if picks else np.nan
    return out


def candidate_quality_table(eval_df, sel, pop_weights, bns_weights, seed=SEED, R=200, log=print):
    """Budget-matched comparators for the symbolic selection.

    Rows: Symbolic (S+E+H+R), Uniform Random, Popularity, Bayesian (BNS, when
    weights are given) and Category-Matched Random (mean over R replicates,
    with an empirical p-value). All comparators draw from one random stream
    seeded with `seed`, in this order.
    """
    rng = np.random.default_rng(seed)
    y = eval_df["label"].to_numpy()
    n_elig, n_sel = len(eval_df), len(sel)
    sym_p = precision(sel["label"].to_numpy())
    rows = [dict(method="Symbolic (S+E+H+R)", candidates=n_sel,
                 coverage=n_sel / n_elig, negative_precision=sym_p,
                 positive_contamination=int(sel["label"].sum()),
                 replicates=None, p_value=None, delta_vs_matched_category=None)]

    def add(name, idx):
        p = precision(y[idx])
        rows.append(dict(method=name, candidates=len(idx), coverage=len(idx) / n_elig,
                         negative_precision=p, positive_contamination=int(y[idx].sum()),
                         replicates=None, p_value=None,
                         delta_vs_matched_category=None))
        return p

    add("Uniform Random", rng.choice(n_elig, size=n_sel, replace=False))
    pw = pop_weights / pop_weights.sum()
    add("Popularity", rng.choice(n_elig, size=n_sel, replace=False, p=pw))
    if bns_weights is not None:
        bw = np.clip(bns_weights, 1e-9, None); bw = bw / bw.sum()
        add("Bayesian (BNS)", rng.choice(n_elig, size=n_sel, replace=False, p=bw))

    label = "Category-Matched Random"
    reps = matched_replicates(eval_df, sel, "category", rng, R)
    mean, sd = float(np.nanmean(reps)), float(np.nanstd(reps))
    pv = float((np.sum(reps >= sym_p) + 1) / (R + 1))
    rows.append(dict(method=label, candidates=n_sel, coverage=n_sel / n_elig,
                     negative_precision=mean, negative_precision_sd=sd,
                     positive_contamination=int(round((1 - mean) * n_sel)),
                     replicates=R, p_value=pv,
                     delta_vs_matched_category=None))
    log(f"    {label}: {mean:.4f} +/- {sd:.4f} over {R} replicates, "
        f"symbolic={sym_p:.4f}, empirical p={pv:.4g}")

    df = pd.DataFrame(rows)
    base = df.loc[df.method == label, "negative_precision"].iloc[0]
    df["delta_vs_matched_category"] = df["negative_precision"] - base
    return df
