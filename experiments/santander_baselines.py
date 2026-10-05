"""Santander candidate-quality table with Popularity and Product-Matched Random.

    python -m experiments.santander_baselines

Run after experiments.santander_candidate_quality. Rules are not
re-discovered: the frozen rules in artifacts/santander/discovered_rules.csv are
re-applied to the same held-out test transition.

Rows of the table (all with the same candidate budget as the symbolic row):
    S + E + H + R            per case, K = 4 eligible products, products that a
                             fully qualified rule nominates first, uniform order
                             within the nominated and non-nominated groups
    Uniform Random           uniform over all eligible test pairs
    Popularity               weighted by training acquisition counts of the product
    Product-Matched Random   per product, as many pairs as the symbolic row took
                             from that product, uniform within the product
                             (mean over 100 replicates, with an empirical p-value)
    Bayesian                 the Bayesian row of results/santander/candidate_quality.csv

Output: results/santander/candidate_quality_full.csv
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from paths import ARTIFACTS, DATA, RESULTS
from prep.santander import CAT, NUM, PROD
from symbolic.tristate import eval_rule_tristate

SEED, K, R = 42, 4, 100
CACHE = f"{DATA}/santander/cache"
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


man = json.loads(str(np.load(f"{CACHE}/manifest.npy", allow_pickle=True)))
nT = len(man["months"]); TE = nT - 1
z = np.load(f"{CACHE}/trans_{TE:02d}.npz")
meta = pd.read_parquet(f"{CACHE}/meta_{TE:02d}.parquet")
own, acq, elig = z["own_t"], z["acq"], z["eligible"]
log(f"test transition {man['split']['test']}: cases={len(meta):,} "
    f"eligible={int(elig.sum()):,} acquisitions={int(acq.sum()):,}")

# ---- rebuild the feature frame as in discovery -----------------------------
f = pd.DataFrame(index=np.arange(len(meta)))
for c in NUM:
    f[c] = pd.to_numeric(meta[c], errors="coerce").to_numpy(float)
f["n_products_owned"] = own.sum(1).astype(float)
tr0 = pd.read_parquet(f"{CACHE}/meta_00.parquet")
for c in CAT:
    cats = sorted(tr0[c].astype("string").fillna("NA").unique())
    m = {v: i for i, v in enumerate(cats)}
    f[c + "_code"] = meta[c].astype("string").fillna("NA").map(m).fillna(len(m)).to_numpy(float)

rules = pd.read_csv(f"{ARTIFACTS}/santander/discovered_rules.csv")
QUAL = rules[rules.stage_C == True]
log(f"frozen rules: {len(QUAL)} qualified (rule, product) pairs, "
    f"{QUAL.rule_id.nunique()} rules, {QUAL.category.nunique()} products")

nominated = np.zeros((len(f), len(PROD)), dtype=bool)
for r in QUAL.itertuples():
    _, fires = eval_rule_tristate(f, r.rule.split(" AND "))
    nominated[fires, int(r.category)] = True
log(f"rule-nominated eligible pairs: {int((nominated & (elig == 1)).sum()):,}")

# ---- symbolic selection ------------------------------------------------------
rng = np.random.default_rng(SEED)
sc, sp = [], []
for ci in range(len(f)):
    e = np.flatnonzero(elig[ci])
    if e.size == 0:
        continue
    lvl = np.where(nominated[ci, e], 0, 9)
    order = np.lexsort((rng.random(e.size), lvl))
    take = e[order][:K]
    sc.append(np.full(len(take), ci)); sp.append(take)
sc = np.concatenate(sc); sp = np.concatenate(sp)
y_sym = acq[sc, sp]
BUDGET = len(sc)
log(f"symbolic selected {BUDGET:,} pairs, precision={1-y_sym.mean():.5f}, "
    f"admitted={int(y_sym.sum()):,}")

# ---- shared eligible universe -----------------------------------------------
ec, ep = np.nonzero(elig)
y_all = acq[ec, ep]
log(f"eligible universe: {len(ec):,} pairs, base acquisition rate={y_all.mean():.5f}")

rng = np.random.default_rng(SEED)
rows = [dict(method="S + E + H + R", rules=int(QUAL.rule_id.nunique()),
             candidates=BUDGET, coverage=BUDGET / len(ec),
             negative_precision=float(1 - y_sym.mean()),
             acquisition_contamination=int(y_sym.sum()),
             replicates=None, p_value=None)]


def add(name, idx):
    y = y_all[idx]
    rows.append(dict(method=name, rules="--", candidates=len(idx),
                     coverage=len(idx) / len(ec),
                     negative_precision=float(1 - y.mean()),
                     acquisition_contamination=int(y.sum()),
                     replicates=None, p_value=None))
    log(f"  {name:28} prec={1-y.mean():.5f} admitted={int(y.sum()):,}")


add("Uniform Random", rng.choice(len(ec), BUDGET, replace=False))

# Popularity: training acquisition counts per product.
pop = np.zeros(len(PROD))
for i in range(nT - 2):
    zz = np.load(f"{CACHE}/trans_{i:02d}.npz"); pop += zz["acq"].sum(0)
w = pop[ep]; w = np.clip(w, 1e-9, None); w = w / w.sum()
log(f"  popularity weights from {nT-2} training transitions")
add("Popularity", rng.choice(len(ec), BUDGET, replace=False, p=w))

# Product-matched random.
counts_p = np.bincount(sp, minlength=len(PROD))
pool_p = {p: np.flatnonzero(ep == p) for p in range(len(PROD))}
reps = np.empty(R)
for r in range(R):
    picks = [rng.choice(pool_p[p], size=min(counts_p[p], len(pool_p[p])), replace=False)
             for p in range(len(PROD)) if counts_p[p] > 0 and len(pool_p[p]) > 0]
    reps[r] = 1 - y_all[np.concatenate(picks)].mean()
mean, sd = float(reps.mean()), float(reps.std())
pv = float((np.sum(reps >= rows[0]["negative_precision"]) + 1) / (R + 1))
rows.append(dict(method="Product-Matched Random", rules="--", candidates=BUDGET,
                 coverage=BUDGET / len(ec), negative_precision=mean,
                 acquisition_contamination=int(round((1 - mean) * BUDGET)),
                 replicates=R, p_value=pv))
log(f"  Product-Matched Random       prec={mean:.5f} +/- {sd:.5f} over {R} reps, "
    f"symbolic={rows[0]['negative_precision']:.5f}, empirical p={pv:.4g}")

# Bayesian, from experiments.santander_candidate_quality.
old = pd.read_csv(f"{RESULTS}/santander/candidate_quality.csv").set_index("sampler")
b = old.loc["bayesian"]
rows.append(dict(method="Bayesian", rules="--", candidates=int(b.selected_pairs),
                 coverage=int(b.selected_pairs) / len(ec),
                 negative_precision=float(b.negative_precision),
                 acquisition_contamination=int(b.acquisitions_admitted),
                 replicates=None, p_value=None))
log(f"  Bayesian                     prec={b.negative_precision:.5f} "
    f"admitted={int(b.acquisitions_admitted):,}")

d = pd.DataFrame(rows)
d.to_csv(f"{RESULTS}/santander/candidate_quality_full.csv", index=False)
print(); print(d.to_string(index=False))
print(f"\nwrote {RESULTS}/santander/candidate_quality_full.csv")
