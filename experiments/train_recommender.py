"""Downstream recommender training with each negative sampler.

    NEGCAND_DATASET=kuairec NEGCAND_SEED=42 python -m experiments.train_recommender
    NEGCAND_DATASET=kuairec NEGCAND_CONDITION=pos05_mnar75 NEGCAND_SEED=1 \
        python -m experiments.train_recommender
    NEGCAND_DATASET=mind NEGCAND_SAMPLERS=random,bns,dns python -m experiments.train_recommender

Requires artifacts/<cell>/rules.csv (experiments.negative_rules) and, for
positive_symbolic, artifacts/<cell>/positive_levels.parquet
(experiments.positive_rules).

Protocol
    KuaiRec (natural and conditions)
        LightGCN, 64 dimensions, 2 layers, Adam, learning rate 5e-3.
        Natural: train on the big matrix minus a seeded 10% validation slice,
        test on the small matrix. Conditions: their own train/val/test.
    MIND, RetailRocket
        DeepFM, 32 dimensions, MLP 256-128, dropout 0.2, Adam, learning rate
        5e-4, fields: user, item, category and the user features.
        Temporal train/val/test splits.
    All
        Batch size 4,096, no weight decay, pointwise binary cross-entropy,
        K = 4 negatives per positive, at most 40 epochs, early stopping on
        validation PR-AUC with patience 15 (ties broken by validation Top-1);
        the best checkpoint is restored before testing. At most 100 training
        positives per user are used as positives. Seeds 42, 1 and 2; the seed
        controls the validation slice (natural KuaiRec), the positive subsample,
        negative sampling, initialization and batch order.

Fairness. Users, positives, splits, architecture, features, loss, optimizer,
schedule, seed and K are identical for every sampler; only the negative rows
change. Every sampler draws K distinct negatives per positive from the same
pool: the catalogue minus the user's observed training positives. DNS draws
its M = 20 candidates from that pool as well.

Symbolic samplers map the frozen rules to users by rebuilding the candidate
rules on the same discovery log that negative discovery used (the training
split, or the full big matrix for natural KuaiRec).

Outputs
    results/<cell>/downstream_s<seed>.csv            one row per sampler
    results/<cell>/sampler_diagnostics_s<seed>.csv   pool, fallback and DNS hardness
    results/<cell>/train_recommender_s<seed>.log
    artifacts/<cell>/<sampler>_candidates_K4_s<seed>.csv   pre-selected negatives
    artifacts/<cell>/dns_picks_<sampler>_K4_s<seed>.npz    DNS picks and hardness per epoch

Options (environment)
    NEGCAND_DATASET      kuairec | mind | retailrocket            (default kuairec)
    NEGCAND_CONDITION    pos05 | pos02 | pos01 | pos05_mnar75     (default: natural)
    NEGCAND_SEED         default 42
    NEGCAND_SAMPLERS     comma-separated subset of samplers.ALL_SAMPLERS (default: all)
    NEGCAND_MAX_EPOCHS, NEGCAND_PATIENCE   override 40 and 15 (for quick checks only)
"""
from __future__ import annotations

import hashlib
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from experiments import samplers as S
from models import recommenders as models
from paths import (ARTIFACTS, RESULTS, SPLIT_DATASETS, cell, cell_from_env, condition_dir,
                   kuairec_table, load_users, split_table)
from symbolic import ladder
from symbolic.discovery import behavioural_from, candidate_rules

DS, CONDITION = cell_from_env()
SEED = int(os.environ.get("NEGCAND_SEED", "42"))
K = 4
SAMPLERS = [s for s in os.environ.get("NEGCAND_SAMPLERS", ",".join(S.ALL_SAMPLERS)).split(",") if s]
MAX_EPOCHS = int(os.environ.get("NEGCAND_MAX_EPOCHS", "40"))
PATIENCE = int(os.environ.get("NEGCAND_PATIENCE", "15"))
BATCH = 4096
ANCHORS_PER_USER = 100
unknown = [s for s in SAMPLERS if s not in S.ALL_SAMPLERS]
assert not unknown, f"unknown samplers {unknown}; choose from {S.ALL_SAMPLERS}"
if DS not in SPLIT_DATASETS and DS != "kuairec":
    raise ValueError(f"downstream training supports kuairec, mind and retailrocket, not {DS!r}")

CELL = cell(DS, CONDITION)
OUT = f"{RESULTS}/{CELL}"; ART = f"{ARTIFACTS}/{CELL}"
os.makedirs(OUT, exist_ok=True); os.makedirs(ART, exist_ok=True)
SUF = f"_s{SEED}"
RESULT_FILE = f"{OUT}/downstream{SUF}.csv"
DIAG_FILE = f"{OUT}/sampler_diagnostics{SUF}.csv"
T0 = time.time(); LOG = open(f"{OUT}/train_recommender{SUF}.log", "a")


def log(m):
    s = f"[{time.time()-T0:7.1f}s] {m}"; print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


dev = "cuda" if torch.cuda.is_available() else "cpu"
log(f"train_recommender -- cell={CELL} seed={SEED} K={K} samplers={SAMPLERS} device={dev}")

# ---------------------------------------------------------------- data ----
users_raw = load_users(DS)
if DS in SPLIT_DATASETS:
    tr = pd.read_parquet(split_table(DS, "train"))
    va = pd.read_parquet(split_table(DS, "val"))
    te = pd.read_parquet(split_table(DS, "test"))
    disc_path = split_table(DS, "train")
else:
    tr = pd.read_parquet(kuairec_table("big"))
    te = pd.read_parquet(kuairec_table("small"))
    rng0 = np.random.default_rng(SEED)
    m = rng0.random(len(tr)) < 0.10
    va, tr = tr[m].reset_index(drop=True), tr[~m].reset_index(drop=True)
    disc_path = kuairec_table("big")
if CONDITION:
    _cd = condition_dir(DS, CONDITION)
    tr = pd.read_parquet(f"{_cd}/train.parquet"); va = pd.read_parquet(f"{_cd}/val.parquet")
    te = pd.read_parquet(f"{_cd}/test.parquet")
    disc_path = f"{_cd}/train.parquet"
log(f"  train={len(tr):,}  val={len(va):,}  test={len(te):,}  "
    f"pos rates {tr.label.mean():.4f}/{va.label.mean():.4f}/{te.label.mean():.4f}")

uids = np.sort(np.unique(np.concatenate([tr.user_id.unique(), va.user_id.unique(), te.user_id.unique()])))
iids = np.sort(np.unique(np.concatenate([tr.video_id.unique(), va.video_id.unique(), te.video_id.unique()])))
uix = pd.Series(np.arange(len(uids)), index=uids); iix = pd.Series(np.arange(len(iids)), index=iids)
nU, nI = len(uids), len(iids)
log(f"  vocab: {nU:,} users, {nI:,} items")
icat = pd.concat([tr[["video_id", "category"]], te[["video_id", "category"]]]) \
         .drop_duplicates("video_id").set_index("video_id")["category"]
item_cat = icat.reindex(iids).fillna(-1).astype(int).to_numpy()

# ------------------------------------------------- frozen symbolic rules ----
rules = pd.read_csv(f"{ART}/rules.csv")
disc = pd.read_parquet(disc_path)
_, _, _, _, cands, disc_users = candidate_rules(disc, users_raw, log=log)
del disc
fire = {c["rule_id"]: c["fire"] for c in cands}
GATE_LVL = ladder.gate_level_table(rules)
BL, BE, CAT2COL = ladder.user_category_rank(
    GATE_LVL, fire, disc_users, np.concatenate([item_cat, rules.category.to_numpy()]))
DUROW = {int(u_): i_ for i_, u_ in enumerate(disc_users)}
log("  gate ladder: " + ", ".join(f"{ladder.LEVEL_NAMES[l]}={int((GATE_LVL.level==l).sum())}"
                                  for l in range(6)) + " (rule, category) pairs")

# ------------------------------------------------------------- positives ----
pos = tr[tr.label == 1]
rng = np.random.default_rng(SEED)
parts = []
for u, g in pos.groupby("user_id", sort=True):
    v = g.video_id.to_numpy()
    if len(v) > ANCHORS_PER_USER:
        v = rng.choice(v, ANCHORS_PER_USER, replace=False)
    parts.append(pd.DataFrame({"user_id": u, "video_id": v}))
anchors = pd.concat(parts, ignore_index=True)
au = uix.reindex(anchors.user_id).to_numpy(); ai = iix.reindex(anchors.video_id).to_numpy()
log(f"  positives: {len(anchors):,} over {anchors.user_id.nunique():,} users")

# Pool of a user: every item without an observed training positive from that user.
pos_by_user = {int(u): set(iix.reindex(g.video_id).to_numpy())
               for u, g in pos.groupby("user_id", sort=True)}
all_items = np.arange(nI)
# Training positives as (user_index * nI + item_index), so DNS can exclude them.
POS_KEYS = np.unique(uix.reindex(pos.user_id).to_numpy().astype(np.int64) * nI
                     + iix.reindex(pos.video_id).to_numpy().astype(np.int64))

# ---- BNS weights from training-split item rates ----------------------------
it = iix.reindex(tr.video_id).to_numpy(); y_tr = tr.label.to_numpy()
npos = np.bincount(it, weights=y_tr, minlength=nI); ntot = np.bincount(it, minlength=nI)
rate = np.divide(npos, ntot, out=np.full(nI, tr.label.mean()), where=ntot > 0)
bns_item = np.clip(1.0 - rate, 1e-6, None)
# BNS-S: Jeffreys posterior mean instead of the raw rate.
bns_smooth = np.clip(1.0 - (npos + 0.5) / (ntot + 1.0), 1e-6, None)
log("  BNS and BNS-S item weights from training-split rates")

# ---- positive evidence (experiments.positive_rules) -------------------------
PE = None
try:
    _pl = pd.read_parquet(f"{ART}/positive_levels.parquet")
    PE = {}
    for _u, _c, _e in zip(_pl.user_id.to_numpy(), _pl.category.to_numpy(), _pl.e_pos.to_numpy()):
        PE.setdefault(int(_u), {})[int(_c)] = float(_e)
    log(f"  positive evidence on {len(_pl):,} (user, category) pairs")
except Exception as _e:
    log(f"  positive evidence unavailable: {_e}")


def _hash(x):
    return hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()[:12]


# Hashes show that the positives and the BNS weights are identical across samplers.
HASHES = dict(anchors_hash=_hash(np.stack([au, ai]).astype(np.int64)),
              bns_item_hash=_hash(bns_item), bns_smooth_hash=_hash(bns_smooth))
log(f"  hashes: {HASHES}")
DIAG = S.Diagnostics(); DNSLOG = {}

u_groups = anchors.groupby("user_id", sort=True).indices


def user_pool(u):
    banned = pos_by_user.get(int(u), set())
    return all_items if not banned else \
        all_items[~np.isin(all_items, np.fromiter(banned, dtype=np.int64))]


def user_levels(u, pool_u):
    row_ = DUROW.get(int(u))
    return ladder.item_levels(item_cat[pool_u],
                              BL[row_] if row_ is not None else None,
                              BE[row_] if row_ is not None else None, CAT2COL)


def select(sampler, seed=SEED):
    """Pre-selected negatives for every positive (DNS samplers are handled in training).

    Returns (user index per row, item index per row, fallback flag, slot label).
    """
    r = np.random.default_rng(seed + 7919 * K)
    N = len(anchors)
    neg = np.empty(N * K, dtype=np.int64)
    lvl_of = np.full(N * K, 3, dtype=np.int8)
    p_of = np.full(N * K, -1, dtype=np.int8); n_of = np.full(N * K, -1, dtype=np.int8)
    for u, idx in u_groups.items():
        a = len(idx)
        slots = (np.asarray(idx)[:, None] * K + np.arange(K)[None, :]).reshape(-1)
        pool_u = user_pool(u)
        if sampler == S.NEGATIVE_SYMBOLIC:
            row_ = DUROW.get(int(u))
            pk, lv, _E = ladder.ladder_pick(
                r, pool_u, item_cat[pool_u],
                BL[row_] if row_ is not None else None,
                BE[row_] if row_ is not None else None, CAT2COL, a, K)
            neg[slots] = pk.reshape(-1); lvl_of[slots] = lv.reshape(-1)
        elif sampler == S.BNS:
            neg[slots] = ladder.distinct_draw(r, pool_u, a, K, bns_item[pool_u]).reshape(-1)
        elif sampler == S.BNS_SMOOTH:
            neg[slots] = ladder.distinct_draw(r, pool_u, a, K, bns_smooth[pool_u]).reshape(-1)
        elif sampler in S.TIERED:
            lev, _E = user_levels(u, pool_u)
            P_ = S.positive_tiers(PE, u, item_cat[pool_u]) if sampler in S.NEEDS_POSITIVE \
                else np.zeros(len(pool_u), dtype=np.int8)
            tier_, w_, prim_ = S.plan(sampler, lev, P_, bns_item[pool_u], bns_smooth[pool_u])
            ix_ = S.tiered_draw(r, a, K, tier_, w_)
            neg[slots] = pool_u[ix_].reshape(-1)
            lvl_of[slots] = tier_[ix_].reshape(-1)
            p_of[slots] = P_[ix_].reshape(-1); n_of[slots] = lev[ix_].reshape(-1)
            DIAG.add(sampler, len(pool_u), prim_, K, tier_[ix_], P_[ix_], P_)
        elif sampler == S.RANDOM:
            neg[slots] = ladder.distinct_draw(r, pool_u, a, K).reshape(-1)
        else:
            raise KeyError(sampler)
    if sampler == S.NEGATIVE_SYMBOLIC:
        fb = lvl_of >= ladder.FILL
        label = np.array(ladder.LEVEL_NAMES, dtype=object)[np.clip(lvl_of, 0, ladder.FILL)]
    elif sampler in S.TIERED:
        fb = lvl_of >= S.FALLBACK_TIER.get(sampler, 99)
        label = S.slot_labels(lvl_of, p_of, n_of)
    else:
        fb = np.zeros(N * K, dtype=bool); label = np.array([""] * (N * K), dtype=object)
    return np.repeat(au, K), neg, fb, label


# ------------------------------------------------------------ evaluation ---
def idx_of(df):
    return (uix.reindex(df.user_id).to_numpy(), iix.reindex(df.video_id).to_numpy(),
            df.label.to_numpy())


VA, TE = idx_of(va), idx_of(te)

ARCH = "deepfm" if DS in SPLIT_DATASETS else "lightgcn"
log(f"  backbone = {ARCH}")
if ARCH == "deepfm":
    u_feat = users_raw.set_index("user_id")
    fields = ["user_id", "video_id", "category"] + [c for c in users_raw.columns if c != "user_id"]
    fdims, ucols = [], {}
    for c in fields[3:]:
        raw = u_feat[c].reindex(uids).astype("string").fillna("NA")
        cat = pd.Categorical(raw); ucols[c] = cat.codes.astype(np.int64).clip(0)
        fdims.append(int(cat.categories.size) + 1)
    ncat = int(item_cat.max()) + 2
    field_dims = [nU, nI, ncat] + fdims
    ucol_mat = np.stack([ucols[c] for c in fields[3:]], axis=1)   # nU x F'

    def make_x(uarr, iarr):
        return np.concatenate([uarr[:, None], iarr[:, None],
                               (item_cat[iarr] + 1)[:, None], ucol_mat[uarr]], axis=1)
    log(f"  DeepFM fields={fields} dims={field_dims}")


def run_one(sampler):
    torch.manual_seed(SEED); np.random.seed(SEED)
    t_start = time.time(); DIAG.start(sampler); DNSLOG.clear()
    DNSLOG.update(ref_rng=np.random.default_rng(SEED + 424242),
                  ref_rng2=np.random.default_rng(SEED + 525252),
                  epochs=[], picks={}, pct_cat={}, pct_elig={}, score={})
    dns = None
    if sampler in S.DNS_SAMPLERS:
        if sampler == S.SYMBOLIC_DNS:
            # The symbolic layer decides which candidates DNS may search;
            # DNS decides hardness with its unchanged scorer (M = 20).
            by_user = pd.Series(np.arange(len(au))).groupby(au).indices
            uni = {}
            for uu, _ix in by_user.items():
                orig = int(uids[uu])
                pool_ = user_pool(orig)
                lev, _E = user_levels(orig, pool_)
                cand_, lvl_ = S.dns_pool(sampler, pool_, lev, S.DNS_M)
                uni[uu] = cand_
                DIAG.add_dns(sampler, len(pool_), len(cand_), len(_ix), lvl_, K, None)
            log(f"    {sampler}: " + ", ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in DIAG.summary(sampler).items()))

            def pool_fn(anchor_u, M, rr):
                out = np.empty((len(anchor_u), M), dtype=np.int64)
                for uu, _ix in by_user.items():
                    c_ = uni[uu]
                    out[_ix] = c_[rr.integers(0, len(c_), size=(len(_ix), M))]
                return out
        else:
            def pool_fn(anchor_u, M, rr):
                c = rr.integers(0, nI, size=(len(anchor_u), M))
                for _ in range(8):      # resample training positives (same pool as every sampler)
                    bad = np.isin(anchor_u[:, None].astype(np.int64) * nI + c, POS_KEYS)
                    if not bad.any():
                        break
                    c[bad] = rr.integers(0, nI, size=int(bad.sum()))
                return c
        dns = models.DNSSelector(au, pool_fn, K, M=S.DNS_M, seed=SEED)
        nu, ni_, fb = np.repeat(au, K), np.zeros(len(au) * K, dtype=np.int64), \
            np.zeros(len(au) * K, bool)
    else:
        nu, ni_, fb, label = select(sampler)
        pd.DataFrame({"user_id": uids[nu], "item_id": iids[ni_],
                      "category": item_cat[ni_], "sampler": sampler, "K": K,
                      "slot": label, "fallback_used": fb, "seed": SEED}).to_csv(
            f"{ART}/{sampler}_candidates_K{K}{SUF}.csv", index=False)
        if sampler == S.NEGATIVE_SYMBOLIC:
            vals, cnts = np.unique(label.astype(str), return_counts=True)
            log(f"    {len(nu):,} negatives by ladder level: "
                + ", ".join(f"{k}={v:,} ({100*v/len(label):.1f}%)" for k, v in zip(vals, cnts)))
        else:
            log(f"    wrote {len(nu):,} {sampler} negatives")

    if ARCH == "lightgcn":
        A, ne = models.build_norm_adj(uix.reindex(pos.user_id).to_numpy(),
                                      iix.reindex(pos.video_id).to_numpy(), nU, nI, dev)
        net = models.LightGCN(nU, nI, A, dim=64, layers=2).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=5e-3)
        log(f"    LightGCN graph {ne:,} edges")

        def score_np(uarr, iarr):
            net.eval()
            with torch.no_grad():
                E = net.propagate(); out = []
                for a in range(0, len(uarr), 1_000_000):
                    out.append(net(torch.as_tensor(np.ascontiguousarray(uarr[a:a+1_000_000]), dtype=torch.long, device=dev),
                                   torch.as_tensor(np.ascontiguousarray(iarr[a:a+1_000_000]), dtype=torch.long, device=dev), E).cpu().numpy())
            return np.concatenate(out)

        def forward(uarr, iarr):
            return net(torch.as_tensor(np.ascontiguousarray(uarr), dtype=torch.long, device=dev),
                       torch.as_tensor(np.ascontiguousarray(iarr), dtype=torch.long, device=dev))
    else:
        net = models.DeepFM(field_dims, dim=32, hidden=(256, 128), dropout=0.2).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=5e-4)

        def score_np(uarr, iarr):
            net.eval(); out = []
            with torch.no_grad():
                for a in range(0, len(uarr), 500_000):
                    x = torch.as_tensor(make_x(uarr[a:a+500_000], iarr[a:a+500_000]), dtype=torch.long, device=dev)
                    out.append(net(x).cpu().numpy())
            return np.concatenate(out)

        def forward(uarr, iarr):
            return net(torch.as_tensor(make_x(uarr, iarr), dtype=torch.long, device=dev))
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator(); g.manual_seed(SEED)
    best, best_tie, best_state, bad, best_ep = -np.inf, -np.inf, None, 0, -1
    best_val = {}
    for ep in range(MAX_EPOCHS):
        if dns is not None:
            nu, ni_ = S.dns_select_logged(dns, ep, score_np, DNSLOG, nI)
        tu = np.concatenate([au, nu]); ti = np.concatenate([ai, ni_])
        ty = np.concatenate([np.ones(len(au), np.float32), np.zeros(len(nu), np.float32)])
        n = len(tu); net.train()
        perm = torch.randperm(n, generator=g).numpy()
        tot = 0.0
        for a in range(0, n, BATCH):
            b = perm[a:a+BATCH]
            out = forward(tu[b], ti[b])
            yb = torch.as_tensor(ty[b], dtype=torch.float32, device=dev)
            opt.zero_grad()
            l = lossf(out, yb)
            l.backward(); opt.step()
            tot += float(l.detach()) * len(b)
        vs = score_np(VA[0], VA[1])
        v_pr = models.pr_auc(VA[2], vs)
        v_t1, n_u, avg_pos = models.top1_hit(VA[0], VA[2], vs)
        log(f"      epoch {ep+1}: loss={tot/n:.4f} val_pr_auc={v_pr:.4f} "
            f"val_top1={v_t1:.4f} (n_users={n_u:,}, avg_pos/user={avg_pos:.1f})")
        # Primary criterion: validation PR-AUC; validation Top-1 breaks exact ties.
        if (v_pr > best) or (v_pr == best and v_t1 > best_tie):
            best, best_tie, bad, best_ep = v_pr, v_t1, 0, ep + 1
            best_val = {"val_pr_auc": v_pr, "val_top1_hit": v_t1,
                        "val_top1_users": n_u, "val_avg_positives_per_user": avg_pos}
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f"      early stop at epoch {ep+1}"); break
    if best_state is not None:
        net.load_state_dict(best_state)
    ts = score_np(TE[0], TE[1])
    nd, rc = models.rank_metrics(TE[0], TE[2], ts, 10)
    t_t1, t_nu, t_avg = models.top1_hit(TE[0], TE[2], ts)
    cats_te = item_cat[TE[1]]; aps = []
    for c_ in np.unique(cats_te):
        m_ = cats_te == c_; y_ = TE[2][m_]
        if 0 < y_.sum() < len(y_):
            aps.append(models.pr_auc(y_, ts[m_]))
    macro_ap = float(np.mean(aps)) if aps else float("nan")
    if DNSLOG.get("epochs"):
        # DNS diagnostics averaged over the refreshes up to the selected epoch.
        eps = [e for e in DNSLOG["epochs"] if e["epoch"] <= max(best_ep, 0)] or DNSLOG["epochs"][:1]
        DNSLOG["summary"] = dict(
            dns_epochs_logged=len(eps),
            **{k: float(np.mean([e[k] for e in eps])) for k in
               ("hardness_percentile", "picked_mean_score", "picked_median_score",
                "hardness_percentile_median", "hardness_pct_eligible",
                "hardness_pct_eligible_median")})
        kp = sorted(k for k in DNSLOG["picks"] if k <= max(best_ep, 0)) or sorted(DNSLOG["picks"])[:1]
        if len(kp) > 5:
            kp = [kp[i] for i in np.linspace(0, len(kp) - 1, 5).astype(int)]
        extra = {f"{nm}{k}": DNSLOG[nm][k] for k in kp for nm in ("pct_cat", "pct_elig", "score")
                 if k in DNSLOG[nm]}
        np.savez_compressed(f"{ART}/dns_picks_{sampler}_K{K}{SUF}.npz",
                            au=np.repeat(au, K).astype(np.int64), uids=uids, iids=iids,
                            **{f"ep{k}": DNSLOG["picks"][k] for k in kp}, **extra)
    return {"dataset": DS, "condition": CONDITION or "natural", "K": K, "sampler": sampler,
            "seed": SEED, "best_epoch": best_ep, **best_val,
            "test_pr_auc": models.pr_auc(TE[2], ts),
            "test_top1_hit": t_t1, "test_top1_users": t_nu,
            "test_avg_positives_per_user": t_avg,
            "test_roc_auc": models.roc_auc(TE[2], ts),
            "test_log_loss": models.log_loss(TE[2], ts),
            "test_ndcg_at_10": nd, "test_recall_at_10": rc,
            "test_macro_ap": macro_ap,
            "n_train_rows": int(len(au) + len(nu)),
            "backbone": "LightGCN(64,2)" if ARCH == "lightgcn" else "DeepFM(32,[256,128])",
            "fallback_rate": float(fb.mean()) if sampler not in S.DNS_SAMPLERS else 0.0,
            "train_time_s": float(time.time() - t_start)}


RES = []
if os.path.exists(RESULT_FILE) and os.path.getsize(RESULT_FILE) > 0:
    try:
        RES = pd.read_csv(RESULT_FILE).to_dict("records")
    except Exception as e:
        log(f"  ignoring unreadable {RESULT_FILE}: {e!r}")
for s in SAMPLERS:
    log(f"  === {CELL} seed={SEED} {s} ===")
    try:
        r = run_one(s); RES.append(r)
        if s in S.DIAGNOSED:
            S.write_diagnostics(DIAG_FILE, dict(
                dataset=DS, condition=CONDITION or "natural", sampler=s, seed=SEED, K=K,
                **HASHES, **DIAG.summary(s), **DNSLOG.get("summary", {})))
        log(f"    best_ep={r['best_epoch']} val_pr={r['val_pr_auc']:.4f} test_pr={r['test_pr_auc']:.4f} "
            f"val_top1={r['val_top1_hit']:.4f} test_top1={r['test_top1_hit']:.4f} "
            f"roc={r['test_roc_auc']:.4f} ndcg@10={r['test_ndcg_at_10']:.4f}")
        log(f"  >> {s}: COMPLETED")
    except Exception as e:
        log(f"  >> {s}: FAILED {e!r}"); log(traceback.format_exc())
    pd.DataFrame(RES).drop_duplicates(["K", "sampler", "seed"], keep="last").to_csv(RESULT_FILE, index=False)
log(f"wrote {RESULT_FILE}  TOTAL {time.time()-T0:.1f}s")
