"""Negative rule discovery and intrinsic candidate quality for one cell.

    python -m experiments.negative_rules
    NEGCAND_DATASET=kuairand python -m experiments.negative_rules
    NEGCAND_DATASET=kuairec NEGCAND_CONDITION=pos05_mnar75 python -m experiments.negative_rules

Discovery and evaluation logs:

    kuairec    discover on the big matrix, evaluate on the small matrix. The
               two matrices share no (user, video) pair.
    kuairand   discover on the standard log (4/08-4/21), evaluate on the
               random-exposure log (4/22-5/08).
    mind, retailrocket
               discover on the training split, evaluate on the test split.
    condition  discover on the condition's training split, evaluate on its
               test split.

Leakage protections. Outcome columns are dropped from the evaluation log
before selection, the selected candidates are written to disk before labels
are joined, and labels are joined by row position (KuaiRand's random log can
show the same video to a user more than once, so (user, video) is not a key).

Outputs
    artifacts/<cell>/rules.csv                 every (rule, category) with S, E, H, R and gates
    artifacts/<cell>/feature_codes.json        categorical code -> label
    artifacts/<cell>/symbolic_candidates.csv   symbolic selection, written before labels
    results/<cell>/candidate_quality.csv       symbolic vs Uniform, Popularity,
                                               Bayesian (BNS) and Category-Matched Random
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import torch

from models.logistic_mf import LogisticMF
from paths import (ARTIFACTS, RESULTS, SPLIT_DATASETS, cell, cell_from_env, condition_dir,
                   kuairand_table, kuairec_table, load_users, split_table)
from symbolic import discovery

SEED = 42
DS, CONDITION = cell_from_env()
R = int(os.environ.get("NEGCAND_REPLICATES", "200"))
CELL = cell(DS, CONDITION)
OUT = f"{RESULTS}/{CELL}"; ART = f"{ARTIFACTS}/{CELL}"
os.makedirs(OUT, exist_ok=True); os.makedirs(ART, exist_ok=True)

T0 = time.time(); LOG = open(f"{OUT}/negative_rules.log", "w")


def log(m):
    s = f"[{time.time()-T0:7.1f}s] {m}"; print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


log(f"negative rules -- dataset={DS} condition={CONDITION or 'natural'} replicates={R}")

users_raw = load_users(DS)
if DS in SPLIT_DATASETS:
    disc = pd.read_parquet(split_table(DS, "train"))
    ev = pd.read_parquet(split_table(DS, "test"))
    OUTCOME = ["label"]
elif DS == "kuairec":
    disc = pd.read_parquet(kuairec_table("big"))
    ev = pd.read_parquet(kuairec_table("small"))
    OUTCOME = ["watch_ratio", "label"]
else:
    disc = pd.read_parquet(kuairand_table("standard_train"))
    ev = pd.read_parquet(kuairand_table("random"))
    OUTCOME = ["label"]

if CONDITION:
    _cd = condition_dir(DS, CONDITION)
    disc = pd.read_parquet(f"{_cd}/train.parquet"); ev = pd.read_parquet(f"{_cd}/test.parquet")
log(f"  discovery: {len(disc):,} rows / {disc.user_id.nunique():,} users / "
    f"pos_rate={disc.label.mean():.4f}")
log(f"  evaluation: {len(ev):,} rows / {ev.user_id.nunique():,} users / "
    f"pos_rate={ev.label.mean():.4f} (hidden until the selection is written)")

# ---- leakage checks --------------------------------------------------------
ev_keys = set(map(tuple, ev[["user_id", "video_id"]].to_numpy()[:200000]))
disc_keys = set(map(tuple, disc[["user_id", "video_id"]].to_numpy()[:200000]))
log(f"  sampled (user, item) overlap between discovery and evaluation = "
    f"{len(ev_keys & disc_keys)}")
ev_blind = ev.drop(columns=[c for c in OUTCOME if c in ev.columns])
for c in OUTCOME:
    assert c not in ev_blind.columns, f"LEAKAGE: {c} visible during selection"
log("  evaluation outcome columns dropped for selection")

# ---- frozen discovery ------------------------------------------------------
rules, cands, frame, disc_users, meta, codes = discovery.discover(disc, users_raw, log=log)
rules.to_csv(f"{ART}/rules.csv", index=False)
json.dump({str(k): v for k, v in codes.items()}, open(f"{ART}/feature_codes.json", "w"),
          default=str)
log(f"  rules frozen -> {ART}/rules.csv")

nom = discovery.user_category_nominations(rules, cands, disc_users, "stage_C")
log(f"  frozen nominations: {len(nom):,} (user, category) pairs, "
    f"{len({u for u, _ in nom}):,} distinct users")

# ---- apply to the blind evaluation log ------------------------------------
sel_blind = discovery.symbolic_select(ev_blind, nom)
log(f"  symbolic selected {len(sel_blind):,} of {len(ev_blind):,} eligible pairs "
    f"(coverage {len(sel_blind)/len(ev_blind):.4f})")

pop = disc.groupby("video_id").size()
ev_blind = ev_blind.reset_index(drop=True)
pop_w = ev_blind["video_id"].map(pop).fillna(1.0).to_numpy(dtype=float)

# ---- Bayesian (BNS) weights over the evaluation pool ----------------------
# A logistic MF trained on the discovery log scores every evaluation pair; the
# score is mapped to P(negative | score) = 1 - pi * f_pos(s) / f_all(s) using
# 100-quantile histograms of the discovery-log scores.
log("  BNS base model on the discovery log only")
uids = np.sort(disc.user_id.unique()); iids = np.sort(
    np.union1d(disc.video_id.unique(), ev.video_id.unique()))
uix = {u: i for i, u in enumerate(uids)}; iix = {v: i for i, v in enumerate(iids)}
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
base = LogisticMF(len(uids), len(iids), 32).to(dev)
opt = torch.optim.Adam(base.parameters(), lr=1e-3, weight_decay=1e-6)
lf = torch.nn.BCEWithLogitsLoss()
bu = torch.as_tensor(np.ascontiguousarray(disc.user_id.map(uix).to_numpy()), dtype=torch.long, device=dev)
bi = torch.as_tensor(np.ascontiguousarray(disc.video_id.map(iix).to_numpy()), dtype=torch.long, device=dev)
by = torch.as_tensor(np.ascontiguousarray(disc.label.to_numpy()), dtype=torch.float32, device=dev)
g = torch.Generator(); g.manual_seed(SEED)
for ep in range(3):
    perm = torch.randperm(len(bu), generator=g).to(dev); tot = 0.0
    for a in range(0, len(bu), 65536):
        b = perm[a:a+65536]; opt.zero_grad()
        l = lf(base(bu[b], bi[b]), by[b]); l.backward(); opt.step()
        tot += float(l.detach()) * len(b)
    log(f"    base epoch {ep+1} loss={tot/len(bu):.4f}")
base.eval()
with torch.no_grad():
    s_disc = np.concatenate([base(bu[a:a+2_000_000], bi[a:a+2_000_000]).cpu().numpy()
                             for a in range(0, len(bu), 2_000_000)])
    eu = torch.as_tensor(np.ascontiguousarray(ev_blind.user_id.map(uix).fillna(0).to_numpy()),
                         dtype=torch.long, device=dev)
    ei = torch.as_tensor(np.ascontiguousarray(ev_blind.video_id.map(iix).fillna(0).to_numpy()),
                         dtype=torch.long, device=dev)
    s_ev = np.concatenate([base(eu[a:a+2_000_000], ei[a:a+2_000_000]).cpu().numpy()
                           for a in range(0, len(eu), 2_000_000)])
y_disc = disc.label.to_numpy().astype(bool); pi = float(y_disc.mean())
bins = np.unique(np.quantile(s_disc, np.linspace(0, 1, 101)))
f_all, _ = np.histogram(s_disc, bins=bins, density=True)
f_pos, _ = np.histogram(s_disc[y_disc], bins=bins, density=True)
ratio = np.divide(f_pos, f_all, out=np.zeros_like(f_all), where=f_all > 0)
pneg = np.clip(1.0 - pi * ratio, 0.0, 1.0)
bns_w = pneg[np.clip(np.digitize(s_ev, bins) - 1, 0, len(pneg) - 1)]
log(f"    BNS pi={pi:.4f}, P(neg|s) in [{bns_w.min():.3f},{bns_w.max():.3f}]")

# ---- persist candidates before any label join ------------------------------
sel_out = sel_blind.copy()
sel_out["sampler"] = "symbolic"; sel_out["symbolic_stage"] = "S+E+H+R"
sel_out["fallback_used"] = False; sel_out["seed"] = SEED; sel_out["K"] = "intrinsic"
for c in OUTCOME:
    assert c not in sel_out.columns, f"LEAKAGE: {c} in persisted candidates"
sel_out.to_csv(f"{ART}/symbolic_candidates.csv", index=False)
log(f"  candidates written before labels are joined: {ART}/symbolic_candidates.csv")

# ---- join labels by row position -------------------------------------------
ev_full = ev.reset_index(drop=True)
sel = sel_blind.copy()
sel["label"] = ev_full["label"].to_numpy()[sel_blind.index.to_numpy()]
assert not sel["label"].isna().any()
log("  labels joined")

cq = discovery.candidate_quality_table(ev_full, sel, pop_w, bns_w, seed=SEED, R=R, log=log)
cq.insert(0, "dataset", DS); cq.insert(1, "condition", CONDITION or "natural")
cq.to_csv(f"{OUT}/candidate_quality.csv", index=False)
log("")
log(cq[["method", "candidates", "coverage", "negative_precision",
        "positive_contamination", "p_value", "delta_vs_matched_category"]].to_string(index=False))
log("")
log(f"wrote {OUT}/candidate_quality.csv   TOTAL {time.time()-T0:.1f}s")
