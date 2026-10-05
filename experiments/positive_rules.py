"""Positive rule discovery for Positive Symbolic.

    python -m experiments.positive_rules
    NEGCAND_DATASET=mind python -m experiments.positive_rules
    NEGCAND_DATASET=kuairec NEGCAND_CONDITION=pos05_mnar75 python -m experiments.positive_rules

The candidate rules are the same as in negative discovery. Each
(rule, category) pair is scored with E_pos (symbolic.evidence.positive_evidence),
which mirrors Evidence Strength with the contrast reversed:

    D_pos = q_r / q_0 - 1,    E_pos = clip(10th percentile of D_pos, 0, 1)

Both the rule arm and the category baseline need at least MIN_ARM training
interactions. Only Scope and E_pos are checked; H and R are not used.

Discovery uses the training split of the cell only (the big matrix for
natural KuaiRec).

Outputs (artifacts/<cell>/)
    positive_rules.csv        every scored (rule, category) pair
    protected_pairs.parquet   (user_id, category) covered by a rule with E_pos >= 0.45
    positive_levels.parquet   (user_id, category, e_pos): the strongest E_pos > 0
                              of any rule covering the pair; used to form the
                              protection tiers of positive_symbolic
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from paths import (ARTIFACTS, SPLIT_DATASETS, cell, cell_from_env, condition_dir,
                   kuairand_table, kuairec_table, load_users, split_table)
from symbolic.discovery import candidate_rules
from symbolic.evidence import positive_evidence

DS, CONDITION = cell_from_env()
MIN_ARM = 5
SEED = 42
E_MIN = 0.45          # same threshold as the negative Evidence Strength gate
ART = f"{ARTIFACTS}/{cell(DS, CONDITION)}"
os.makedirs(ART, exist_ok=True)


def log(m):
    print(f"  {m}", flush=True)


if CONDITION:
    tr = pd.read_parquet(f"{condition_dir(DS, CONDITION)}/train.parquet")
elif DS in SPLIT_DATASETS:
    tr = pd.read_parquet(split_table(DS, "train"))
elif DS == "kuairec":
    tr = pd.read_parquet(kuairec_table("big"))
else:
    tr = pd.read_parquet(kuairand_table("standard_train"))
users_raw = load_users(DS)

log(f"dataset={DS} condition={CONDITION or 'natural'}  train={len(tr):,}  "
    f"pos_rate={tr.label.mean():.4f}  (training split only)")

frame, codes, preds, _, cands, uids = candidate_rules(tr, users_raw, log=log)

fire = {c["rule_id"]: c["fire"] for c in cands}
u2row = {int(u): i for i, u in enumerate(uids)}
row = tr.user_id.map(u2row).to_numpy()
cat = tr.category.to_numpy()
y = tr.label.to_numpy()
cats = np.sort(pd.unique(cat))

rows = []
for c in cands:
    f = fire[c["rule_id"]]
    hit = f[row]
    if hit.sum() < MIN_ARM:
        continue
    for k in cats:
        m = cat == k
        n0 = int(m.sum())
        if n0 < MIN_ARM:
            continue
        mr = m & hit
        n_r = int(mr.sum())
        if n_r < MIN_ARM or n_r >= n0:
            continue
        c_r = int(y[mr].sum()); c0 = int(y[m].sum())
        E = positive_evidence(c_r, n_r, c0, n0, SEED)
        if E is None:
            continue
        rows.append(dict(rule_id=c["rule_id"], category=int(k),
                         rule=" AND ".join(c["predicates"]), arity=c["arity"],
                         S=n_r / n0, E_pos=E, n_rule=n_r, n_observable=n0,
                         q_rule=c_r / n_r, q_baseline=c0 / n0,
                         support_rows=n_r, positive_count=c_r, negative_count=n_r - c_r,
                         positive_rate=c_r / n_r, confidence=E,
                         tier=("strong" if E >= 0.45 else "medium" if E >= 0.20
                               else "weak" if E > 0 else "none"),
                         n_users_covered=int(f.sum())))
R = pd.DataFrame(rows)
log(f"scored {len(R):,} (rule, category) pairs")
if len(R):
    keep = R[(R.E_pos >= E_MIN) & (R.q_rule > R.q_baseline)]
    log(f"E_pos >= {E_MIN}: {len(keep):,} pairs, {keep.rule_id.nunique()} rules, "
        f"{keep.category.nunique()} categories")
    R.to_csv(f"{ART}/positive_rules.csv", index=False)
    prot = []
    for r_ in keep.itertuples():
        for u in uids[fire[r_.rule_id]]:
            prot.append((int(u), int(r_.category)))
    P = pd.DataFrame(prot, columns=["user_id", "category"]).drop_duplicates()
    P.to_parquet(f"{ART}/protected_pairs.parquet", index=False)

    # Strongest positive evidence per (user, category), over every rule with
    # E_pos > 0, so the sampler can order candidates by protection tier.
    graded = R[(R.E_pos > 0) & (R.q_rule > R.q_baseline)]
    best = {}
    for r_ in graded.itertuples():
        f = fire[r_.rule_id]
        for u in uids[f]:
            k = (int(u), int(r_.category))
            if r_.E_pos > best.get(k, 0.0):
                best[k] = float(r_.E_pos)
    if best:
        L = pd.DataFrame([(u, c, e) for (u, c), e in best.items()],
                         columns=["user_id", "category", "e_pos"])
        L.to_parquet(f"{ART}/positive_levels.parquet", index=False)
        _b = np.digitize(L.e_pos.to_numpy(), [1e-12, 0.20, 0.35, 0.45])
        log(f"pairs with any positive evidence: {len(L):,}")
        log("E_pos bands (1: <0.20, 2: <0.35, 3: <0.45, 4: >=0.45): "
            + ", ".join(f"{t}={int((_b==t).sum()):,}" for t in (1, 2, 3, 4)))
    tot = len(uids) * len(cats)
    log(f"protected (user, category) pairs: {len(P):,} of {tot:,} = {len(P)/tot:.2%}")
    log(f"mean rate in protected rules {keep.q_rule.mean():.4f} vs baseline {keep.q_baseline.mean():.4f}")
else:
    log("no scorable pairs")
