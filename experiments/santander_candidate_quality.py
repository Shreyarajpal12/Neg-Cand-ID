"""Santander: rule discovery and candidate quality of Random, Bayesian and Symbolic.

    python -m experiments.santander_candidate_quality
    python -m experiments.santander_baselines     # adds Popularity and Product-Matched Random

Requires the transition cache from prep.santander.

Data interface for the S/E/H/R statistics:
    product p       one of the 24 ind_*_ult1 products
    outcome unit    one eligible (case, p) pair on a training transition
    conversion      the product was acquired at t + 1
A rule with high E is a customer condition under which acquisition of p is
less likely than that product's own base rate.

Discovery uses a random sample of DISC_CASES training cases (seed 42).
Predicates are built from training quantiles of the numeric features and the
customer's number of owned products, plus categorical levels that cover at
least MIN_LEVEL_SUPPORT of the sample; candidate rules that hold for less
than MIN_SUPPORT of the sample are dropped. Santander's categoricals are wide,
so these two filters keep the sweep tractable; an arm that thin could not
support an estimable H comparison in any case.

Candidate quality. The frozen rules are applied to the held-out test
transition. For every test case with at least one eligible product, each
sampler selects K = 4 eligible products (all of them when fewer are eligible):
    random     uniform
    bayesian   weights 1 - (training acquisition rate of the product)
    symbolic   gate-ranked ladder (symbolic.ladder levels), uniform within a level
Negative precision is the fraction of selected pairs that were not acquired.

Outputs
    artifacts/santander/discovered_rules.csv
    artifacts/santander/selected_negative_pairs.csv
    artifacts/santander/config.json
    results/santander/candidate_quality.csv
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from paths import ARTIFACTS, DATA, RESULTS
from prep.santander import CAT, NUM, PROD
from symbolic import ladder
from symbolic.rules import assign_stages_and_tiers, build_candidates, score_all
from symbolic.tristate import eval_rule_tristate

SEED = 42
K = 4
DISC_CASES = 150_000
MIN_LEVEL_SUPPORT = 0.001
MIN_SUPPORT = 0.002
CACHE = f"{DATA}/santander/cache"
OUT = f"{RESULTS}/santander"; ART = f"{ARTIFACTS}/santander"
os.makedirs(OUT, exist_ok=True); os.makedirs(ART, exist_ok=True)
T0 = time.time(); LOG = open(f"{OUT}/santander_candidate_quality.log", "w")


def log(m):
    s = f"[{time.time()-T0:7.1f}s] {m}"; print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


man = json.loads(str(np.load(f"{CACHE}/manifest.npy", allow_pickle=True)))
nT = len(man["months"])
log(f"transitions={nT}  train t={man['split']['train'][0]}..{man['split']['train'][-1]}  "
    f"val t={man['split']['validation']}  test t={man['split']['test']}")


def load_trans(i):
    z = np.load(f"{CACHE}/trans_{i:02d}.npz")
    m = pd.read_parquet(f"{CACHE}/meta_{i:02d}.parquet")
    return m, z["own_t"], z["acq"], z["eligible"]


TR_IDX = list(range(nT - 2)); TE_IDX = nT - 1


# ---------------------------------------------------------------- features --
def frame_of(meta, own):
    f = pd.DataFrame(index=np.arange(len(meta)))
    for c in NUM:
        f[c] = pd.to_numeric(meta[c], errors="coerce").to_numpy(dtype=float)
    f["n_products_owned"] = own.sum(1).astype(float)
    return f


CATMAPS = {}


def cat_codes(meta, fit=False):
    """Integer codes per categorical column; unseen levels map to a reserved code."""
    out = {}
    for c in CAT:
        s = meta[c].astype("string").fillna("NA")
        if fit or c not in CATMAPS:
            CATMAPS[c] = {v: i for i, v in enumerate(sorted(s.unique()))}
        out[c] = s.map(CATMAPS[c]).fillna(len(CATMAPS[c])).to_numpy(np.int64)
    return out


# ------------------------------------------------- assemble training cases --
# Each transition is converted to integer codes on load and only the integer
# arrays are concatenated. The categorical vocabulary is fitted on the first
# training transition and reused for all later ones.
log("assembling training cases")
m0, _o0, _a0, _e0 = load_trans(TR_IDX[0])
cat_codes(m0, fit=True)
del _o0, _a0, _e0
nums, codes_l, owns, acqs, eligs = [], [], [], [], []
for i in TR_IDX:
    m, o, a, e = load_trans(i)
    nums.append(np.stack([pd.to_numeric(m[c], errors="coerce").to_numpy(np.float32)
                          for c in NUM], 1))
    cc = cat_codes(m)
    codes_l.append(np.stack([cc[c] for c in CAT], 1).astype(np.int32))
    owns.append(o); acqs.append(a); eligs.append(e)
    del m, cc
TR_num = np.concatenate(nums); TR_code = np.concatenate(codes_l)
TR_own = np.concatenate(owns); TR_acq = np.concatenate(acqs); TR_elig = np.concatenate(eligs)
del nums, codes_l, owns, acqs, eligs
TR_meta = pd.DataFrame(TR_num, columns=NUM)
TR_CODES = {c: TR_code[:, j].astype(np.int64) for j, c in enumerate(CAT)}
log(f"train cases={len(TR_meta):,}  eligible pairs={int(TR_elig.sum()):,}  "
    f"acquisitions={int(TR_acq.sum()):,}")
TE_meta, TE_own, TE_acq, TE_elig = load_trans(TE_IDX)
log(f"test cases={len(TE_meta):,}")

TR_f = frame_of(TR_meta, TR_own); TE_f = frame_of(TE_meta, TE_own)
NUMF = list(TR_f.columns)

# ------------------------------------------------------- symbolic discovery --
log("SYMBOLIC DISCOVERY (training transitions only)")
rng = np.random.default_rng(SEED)
sub = rng.choice(len(TR_meta), size=min(DISC_CASES, len(TR_meta)), replace=False)
sub.sort()
d_f = TR_f.iloc[sub].reset_index(drop=True)
d_codes = {c: v[sub] for c, v in TR_CODES.items()}
for c, v in d_codes.items():
    d_f[c + "_code"] = v.astype(float)

preds, meta_p = [], {}
for c in NUMF:
    v = d_f[c].to_numpy(float); v = v[~np.isnan(v)]
    if v.size == 0:
        continue
    seen = set()
    if (v == 0).any():
        preds.append(f"{c}==0"); seen.add(0.0); meta_p[f"{c}==0"] = {"column": c, "op": "=="}
    for tag, q in (("q25", .25), ("q50", .5), ("q75", .75)):
        t = float(np.quantile(v, q))
        if t == 0.0 or t in seen:
            continue
        seen.add(t); p = f"{c}<={t}"; preds.append(p)
        meta_p[p] = {"column": c, "op": "<=", "threshold": t, "quantile": tag}
for c in CAT:
    col = c + "_code"
    vals, cnts = np.unique(d_codes[c], return_counts=True)
    for v in vals[cnts >= MIN_LEVEL_SUPPORT * len(d_f)]:
        p = f"{col}=={float(v)}"; preds.append(p)
        inv = {i: k for k, i in CATMAPS[c].items()}
        meta_p[p] = {"column": col, "op": "==", "value": inv.get(int(v), "NA"), "source": c}
log(f"  {len(preds)} predicates over {len(NUMF)} numeric + {len(CAT)} categorical")
cands = build_candidates(d_f, preds, max_arity=2)
floor = int(MIN_SUPPORT * len(d_f))
cands = [c for c in cands if int(c["fire"].sum()) >= floor]
for _i, _c in enumerate(cands):
    _c["rule_id"] = f"R{_i:04d}"
log(f"  {len(cands)} candidate rules "
    f"(arity1={sum(c['arity']==1 for c in cands)}, arity2={sum(c['arity']==2 for c in cands)})")

N = TR_elig[sub].T.astype(np.int64)     # (24 products x cases) eligible
P = TR_acq[sub].T.astype(np.int64)      # (24 products x cases) acquired
rules, _ = score_all(cands, N, P, list(range(len(PROD))), seed=SEED, log=log)
rules = assign_stages_and_tiers(rules)
rules["product"] = rules["category"].map(lambda i: PROD[int(i)])
rules["predicate_meta"] = rules["rule"].map(
    lambda r: json.dumps([meta_p.get(p, {}) for p in r.split(" AND ")]))
rules.to_csv(f"{ART}/discovered_rules.csv", index=False)
for s in ("stage_A", "stage_B", "stage_C"):
    log(f"  {s}: {int(rules[s].sum())} pairs, {rules.loc[rules[s],'rule_id'].nunique()} rules, "
        f"{rules.loc[rules[s],'product'].nunique()} products")

GLT = ladder.gate_level_table(rules)
log("  gate ladder: " + ", ".join(f"{ladder.LEVEL_NAMES[l]}={int((GLT.level==l).sum())}"
                                  for l in range(6)))


def apply_rules(frame, codes):
    """Frozen rules -> per (case, product) best gate level, its E, and rule id.

    Each distinct predicate is evaluated once and rules are combined from the
    cached masks, with the same tri-state semantics as eval_rule_tristate
    (observable = AND of observables; fires = AND of fires, within observable).
    """
    f = frame.copy()
    for c, v in codes.items():
        f[c + "_code"] = v.astype(float)
    lvl = np.full((len(f), len(PROD)), ladder.FILL, dtype=np.int8)
    Ev = np.zeros((len(f), len(PROD)), dtype=np.float32)
    rule_id = np.zeros((len(f), len(PROD)), dtype=np.int32)
    ridmap, nxt, cache, pcache = {}, 1, {}, {}

    def _pred(pr):
        if pr not in pcache:
            pcache[pr] = eval_rule_tristate(f, [pr])
        return pcache[pr]
    for r in GLT.sort_values(["level", "E"], ascending=[True, False]).itertuples():
        if r.rule not in cache:
            ob = None; fi = None
            for pr in r.rule.split(" AND "):
                o_, f_ = _pred(pr)
                ob = o_.copy() if ob is None else (ob & o_)
                fi = f_.copy() if fi is None else (fi & f_)
            cache[r.rule] = fi & ob
        fires = cache[r.rule]; p = int(r.category)
        upd = fires & ((r.level < lvl[:, p]) | ((r.level == lvl[:, p]) & (r.E > Ev[:, p])))
        lvl[upd, p] = int(r.level); Ev[upd, p] = r.E
        if r.rule_id not in ridmap:
            ridmap[r.rule_id] = nxt; nxt += 1
        rule_id[upd, p] = ridmap[r.rule_id]
    return lvl, rule_id, {v: k for k, v in ridmap.items()}, Ev


TE_codes = cat_codes(TE_meta)
log("applying frozen rules to the held-out test transition")
TE_lvl, TE_rid, RIDMAP, TE_E = apply_rules(TE_f, TE_codes)
log(f"  rule-nominated eligible pairs on test: "
    f"{int(((TE_lvl < ladder.FILL) & (TE_elig == 1)).sum()):,} of {int(TE_elig.sum()):,}")

prod_rate = TR_acq.sum(0) / np.maximum(TR_elig.sum(0), 1)   # training acquisition rate
bns_w = np.clip(1.0 - prod_rate, 1e-6, None)                # BNS: P(true negative)
log("  BNS product weights from training acquisition rates only")


def pick_negatives(elig, lvl, K, sampler, seed):
    """K products per case from that case's own eligible set."""
    r = np.random.default_rng(seed)
    n_cases, n_p = elig.shape
    rows_c, rows_p = [], []
    for ci in range(n_cases):
        e = np.flatnonzero(elig[ci])
        if e.size == 0:
            continue
        if sampler == "symbolic":
            order = np.lexsort((r.random(e.size), lvl[ci, e]))
            take = e[order][:K]
        elif sampler == "bayesian":
            w = bns_w[e]; w = w / w.sum()
            take = r.choice(e, size=min(K, e.size), replace=False, p=w)
        elif sampler == "random":
            take = r.choice(e, size=min(K, e.size), replace=False)
        else:
            raise KeyError(sampler)
        rows_c.append(np.full(len(take), ci)); rows_p.append(take)
    return np.concatenate(rows_c), np.concatenate(rows_p)


log("CANDIDATE QUALITY on the frozen test transition")
act = TE_acq.sum(1) > 0                      # cases with at least one acquisition
cq_rows, sel_store = [], []
for smp in ("random", "bayesian", "symbolic"):
    c, p = pick_negatives(TE_elig, TE_lvl, K, smp, SEED)
    y = TE_acq[c, p]
    sel_store.append(pd.DataFrame({
        "customer_id": TE_meta.ncodpers.to_numpy()[c], "month": man["split"]["test"],
        "product": [PROD[i] for i in p], "sampler": smp, "K": K,
        "rule_id": [RIDMAP.get(int(TE_rid[a, b]), "") if smp == "symbolic" else ""
                    for a, b in zip(c, p)],
        "bayesian_score": np.where(smp == "bayesian", bns_w[p], np.nan),
        "seed": SEED}))
    m = act[c]
    cq_rows.append({"sampler": smp, "selected_pairs": len(c),
                    "negative_precision": float(1 - y.mean()),
                    "acquisitions_admitted": int(y.sum()),
                    "selected_pairs_active": int(m.sum()),
                    "negative_precision_active": float(1 - y[m].mean()) if m.any() else np.nan,
                    "acquisitions_admitted_active": int(y[m].sum())})
    log(f"  {smp:9} pairs={len(c):>9,} neg_prec={cq_rows[-1]['negative_precision']:.5f} "
        f"admitted={int(y.sum()):,} | acquiring-case slice prec="
        f"{cq_rows[-1]['negative_precision_active']:.5f} admitted={int(y[m].sum()):,}")
pd.concat(sel_store, ignore_index=True).to_csv(f"{ART}/selected_negative_pairs.csv", index=False)
pd.DataFrame(cq_rows).to_csv(f"{OUT}/candidate_quality.csv", index=False)

json.dump({"seed": SEED, "K": K, "split": man["split"], "products": PROD,
           "features": {"numeric": NUMF, "categorical": CAT},
           "predicates": ["x==0", "x<=q25", "x<=q50", "x<=q75", "x==value"],
           "max_rule_length": 2, "thresholds": {"S": ">0", "E": ">=0.45", "H": ">0", "R": ">0"},
           "discovery_cases_sampled": int(min(DISC_CASES, len(TR_meta))),
           "min_level_support": MIN_LEVEL_SUPPORT, "min_rule_support": MIN_SUPPORT},
          open(f"{ART}/config.json", "w"), indent=2, default=str)
log(f"wrote {OUT}/candidate_quality.csv   TOTAL {time.time()-T0:.1f}s")
