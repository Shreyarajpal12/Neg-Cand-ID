"""Quality of the negatives each sampler selected, from saved sampler outputs only.

    python -m analysis.negative_precision

No model is retrained. Inputs per (cell, sampler, seed), written by
experiments.train_recommender:
    pre-selected samplers   artifacts/<cell>/<sampler>_candidates_K4_s<seed>.csv
    dns, symbolic_dns       artifacts/<cell>/dns_picks_<sampler>_K4_s<seed>.npz
                            (picks at up to five refreshes before the selected epoch)
    pool, fallback, DNS hardness
                            results/<cell>/sampler_diagnostics_s<seed>.csv

Labels are joined here, after sampling. Each sampled (user, item) pair is
looked up in the union of the cell's train, validation and test rows. In the
missing-positive condition, training rows use labels_clean.parquet, so a
hidden positive counts as a positive. A pair that appears in no split has no
label (unlabelled). For natural KuaiRec, the test split is the fully observed
small matrix; MIND labels are impression clicks; RetailRocket labels are
purchases.

    negative_precision              n_label0 / n_selected
    positive_contamination          n_label1 / n_selected
    *_labeled                       the same ratios over labelled candidates only
    label_coverage                  labelled / n_selected
    candidate_coverage              share of each user's pool kept by the symbolic layer

The labelled ratios are the measure reported in the paper; the unlabelled
pairs have an unknown outcome and are not counted as errors.

Protected-set metrics for positive_symbolic are computed over every labelled
pool pair (catalogue minus the user's observed training positives), because
protection acts on the pool rather than on the sampled negatives. The
reported protected region is the strong tier (P = 3).

Outputs (results/summary/)
    candidate_quality.csv           per seed
    candidate_quality_mean.csv      mean and SD over seeds
    protected_set.csv               positive-symbolic protection regions
    candidate_quality_missing.csv   inputs that were not found
"""
import glob
import os

import numpy as np
import pandas as pd

from experiments import samplers as S
from paths import ARTIFACTS, RESULTS, SUMMARY, condition_dir, kuairec_table, split_table

SEEDS = [42, 1, 2]
SAMPLERS = S.ALL_SAMPLERS
DNS_SAMPLERS = S.DNS_SAMPLERS
FILTERED = {S.SYMBOLIC_RANDOM, S.SYMBOLIC_BNS, S.SYMBOLIC_BNS_SMOOTH, S.SYMBOLIC_DNS,
            S.POSITIVE_SYMBOLIC, S.NEGATIVE_SYMBOLIC}
# (dataset label, condition label, cell directory, label source)
CELLS = [("KuaiRec", "1% clean", "kuairec_pos01", ("condition", "pos01")),
         ("KuaiRec", "5% clean", "kuairec_pos05", ("condition", "pos05")),
         ("KuaiRec", "5% + MNAR-75", "kuairec_pos05_mnar75", ("condition", "pos05_mnar75")),
         ("KuaiRec", "natural", "kuairec", ("kuairec", None)),
         ("MIND", "natural", "mind", ("split", "mind")),
         ("RetailRocket", "natural", "retailrocket", ("split", "retailrocket"))]
POS_EDGES = S.POS_EDGES
BIG = np.int64(10) ** 9
DIAG_COLS = ("symbolic_pool_coverage", "fallback_fraction", "mean_pool_size", "mean_safe_pool_size",
             "hardness_percentile", "hardness_percentile_median", "hardness_pct_eligible",
             "hardness_pct_eligible_median", "picked_mean_score", "picked_median_score",
             "pct_candidates_pos_evidence", "pct_candidates_strong_pos")


def suf(seed):
    return f"_s{seed}"


def label_table(src):
    """(user, item) -> label over every split; clean labels in the missing-positive condition."""
    kind, name = src
    parts = []
    if kind == "condition":
        b = condition_dir("kuairec", name)
        tr = pd.read_parquet(f"{b}/train.parquet", columns=["user_id", "video_id", "label", "category"])
        tr["obs"] = tr.label
        if os.path.exists(f"{b}/labels_clean.parquet"):
            tr["label"] = pd.read_parquet(f"{b}/labels_clean.parquet").iloc[:, 0].to_numpy()
        parts.append(tr.assign(src="train"))
        for s in ("val", "test"):
            x = pd.read_parquet(f"{b}/{s}.parquet", columns=["user_id", "video_id", "label", "category"])
            parts.append(x.assign(obs=x.label, src=s))
    elif kind == "kuairec":
        for f, s in ((kuairec_table("big"), "train"), (kuairec_table("small"), "test")):
            x = pd.read_parquet(f, columns=["user_id", "video_id", "label", "category"])
            parts.append(x.assign(obs=x.label, src=s))
    else:
        for s in ("train", "val", "test"):
            x = pd.read_parquet(split_table(name, s), columns=["user_id", "video_id", "label", "category"])
            parts.append(x.assign(obs=x.label, src=s))
    L = pd.concat(parts, ignore_index=True)
    L["key"] = L.user_id.astype(np.int64) * BIG + L.video_id.astype(np.int64)
    # A pair is positive if it was ever (clean-)positive; train_obs_pos marks the
    # observed training positives that every sampler excludes from its pool.
    L["train_obs_pos"] = (L.src == "train") & (L.obs == 1)
    L["heldout"] = L.src != "train"
    G = L.groupby("key").agg(label=("label", "max"), train_obs_pos=("train_obs_pos", "max"),
                             user_id=("user_id", "first"), category=("category", "first"))
    hp = L[L.heldout & (L.label == 1)].key.unique()
    G["pos_heldout"] = G.index.isin(hp)
    tp = L[(~L.heldout) & (L.label == 1) & (L.obs == 0)].key.unique()   # hidden training positives
    G["pos_hidden"] = G.index.isin(tp)
    return G.sort_index()


def lookup(G, users, items):
    k = users.astype(np.int64) * BIG + items.astype(np.int64)
    pos = np.searchsorted(G.index.values, k)
    pos = np.clip(pos, 0, len(G) - 1)
    hit = G.index.values[pos] == k
    lab = np.where(hit, G.label.values[pos], -1)
    hid = hit & G.pos_hidden.values[pos]
    hel = hit & G.pos_heldout.values[pos]
    return lab, hid, hel


def quality(lab, hid, hel):
    n = len(lab); n0 = int((lab == 0).sum()); n1 = int((lab == 1).sum()); nl = n0 + n1
    return dict(n_selected=n, n_label0=n0, n_label1=n1, n_unlabeled=n - nl,
                negative_precision=n0 / max(n, 1), positive_contamination=n1 / max(n, 1),
                negative_precision_labeled=n0 / max(nl, 1), positive_contamination_labeled=n1 / max(nl, 1),
                label_coverage=nl / max(n, 1),
                contamination_hidden_train=float(hid.sum()) / max(n, 1),
                contamination_heldout=float(hel.sum()) / max(n, 1))


def diagnostics_table(cell_dir):
    fs = glob.glob(f"{RESULTS}/{cell_dir}/sampler_diagnostics_s*.csv")
    return pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(["sampler", "seed"], keep="last") \
        if fs else pd.DataFrame(columns=["sampler", "seed"])


def protected_rows(ds, cond, G, levels_path):
    """Positive precision, base rate, lift and recall of the protection tiers."""
    P = pd.read_parquet(levels_path)
    P["tier"] = np.digitize(P.e_pos.to_numpy(), POS_EDGES)
    pk = P.user_id.astype(np.int64) * 100000 + P.category.astype(np.int64)
    tmap = pd.Series(P.tier.to_numpy(), index=pk)
    pool = G[~G.train_obs_pos.astype(bool)]
    ck = pool.user_id.astype(np.int64) * 100000 + pool.category.fillna(-1).astype(np.int64)
    tier = tmap.reindex(ck.to_numpy()).fillna(0).to_numpy().astype(int)
    y = pool.label.to_numpy() == 1
    allpos = int(y.sum())
    sets = [("strong (P=3)", tier == 3), ("any positive evidence (P>=1)", tier >= 1)] + \
           [(f"tier P={t} only", tier == t) for t in range(4)]
    out = []
    for name, msk in sets:
        tp = int((msk & y).sum()); n = int(msk.sum())
        out.append(dict(dataset=ds, condition=cond, protected_set=name,
                        labelled_pool_pairs=len(pool), protected_n=n, protected_true_positive=tp,
                        protected_positive_precision=tp / max(n, 1),
                        base_positive_rate=allpos / max(len(pool), 1),
                        precision_lift=(tp / max(n, 1)) / max(allpos / max(len(pool), 1), 1e-12),
                        protected_positive_recall=tp / max(allpos, 1),
                        pct_labelled_pool_protected=n / max(len(pool), 1)))
    return out


def main():
    rows, prot_rows, missing = [], [], []
    for ds, cond, cdir, src in CELLS:
        print(f"== {ds} {cond}", flush=True)
        G = label_table(src)
        M = diagnostics_table(cdir)
        art = f"{ARTIFACTS}/{cdir}"
        for seed in SEEDS:
            m_seed = M[M.seed == seed].set_index("sampler")
            pool_before = next((m_seed.loc[s, "mean_pool_size"]
                                for s in (S.SYMBOLIC_RANDOM, S.POSITIVE_SYMBOLIC)
                                if s in m_seed.index), np.nan)
            for s in SAMPLERS:
                r = dict(dataset=ds, condition=cond, sampler=s, seed=seed)
                if s in DNS_SAMPLERS:
                    f = f"{art}/dns_picks_{s}_K4{suf(seed)}.npz"
                    if not os.path.exists(f):
                        missing.append((ds, cond, s, seed)); continue
                    z = np.load(f, allow_pickle=True)
                    uids, iids, au = z["uids"], z["iids"], z["au"]
                    eps = sorted(k for k in z.files if k.startswith("ep"))
                    qs = [quality(*lookup(G, uids[au], iids[z[k]])) for k in eps]
                    q = {k: (float(np.mean([x[k] for x in qs])) if k != "n_selected" else qs[-1][k]) for k in qs[0]}
                    r.update(q, dns_epochs_logged=len(eps))
                else:
                    f = f"{art}/{s}_candidates_K4{suf(seed)}.csv"
                    if not os.path.exists(f):
                        missing.append((ds, cond, s, seed)); continue
                    c = pd.read_csv(f, usecols=["user_id", "item_id", "fallback_used"])
                    r.update(quality(*lookup(G, c.user_id.to_numpy(), c.item_id.to_numpy())))
                    if s == S.NEGATIVE_SYMBOLIC:
                        r["fallback_fraction"] = float(c.fallback_used.astype(bool).mean())
                r["mean_pool_size"] = pool_before
                if s in m_seed.index:
                    mm = m_seed.loc[s]
                    for k in DIAG_COLS:
                        if k in mm and pd.notna(mm[k]):
                            r[k] = float(mm[k])
                if s not in FILTERED:
                    r["symbolic_pool_coverage"] = 1.0
                    r["mean_safe_pool_size"] = r["mean_pool_size"]
                rows.append(r)

        lv = f"{art}/positive_levels.parquet"
        if os.path.exists(lv):
            prot_rows += protected_rows(ds, cond, G, lv)
        del G

    os.makedirs(SUMMARY, exist_ok=True)
    Q = pd.DataFrame(rows)
    PR = pd.DataFrame(prot_rows)
    if len(PR):
        strong = PR[PR.protected_set == "strong (P=3)"].set_index(["dataset", "condition"])
        for k in ("protected_n", "protected_true_positive", "protected_positive_precision",
                  "protected_positive_recall"):
            Q[k] = [strong.loc[(d, c), k] if (s == S.POSITIVE_SYMBOLIC and (d, c) in strong.index) else np.nan
                    for d, c, s in zip(Q.dataset, Q.condition, Q.sampler)]
    Q = Q.rename(columns={"symbolic_pool_coverage": "candidate_coverage",
                          "mean_safe_pool_size": "mean_pool_size_after_filter"})
    lead = [c for c in ["dataset", "condition", "sampler", "seed", "n_selected",
                        "negative_precision_labeled", "positive_contamination_labeled",
                        "negative_precision", "positive_contamination", "label_coverage",
                        "candidate_coverage", "fallback_fraction", "mean_pool_size",
                        "hardness_pct_eligible", "hardness_percentile"] if c in Q.columns]
    Q = Q[lead + [c for c in Q.columns if c not in lead]]
    Q.to_csv(f"{SUMMARY}/candidate_quality.csv", index=False)
    PR.to_csv(f"{SUMMARY}/protected_set.csv", index=False)
    num = [c for c in Q.columns if c not in ("dataset", "condition", "sampler", "seed") and Q[c].dtype != object]
    A = Q.groupby(["dataset", "condition", "sampler"], sort=False)[num].agg(["mean", "std"])
    A.columns = [f"{a}_{b}" for a, b in A.columns]
    A.reset_index().to_csv(f"{SUMMARY}/candidate_quality_mean.csv", index=False)
    pd.DataFrame(missing, columns=["dataset", "condition", "sampler", "seed"]).to_csv(
        f"{SUMMARY}/candidate_quality_missing.csv", index=False)
    print(f"rows={len(Q)} missing={len(missing)} -> {SUMMARY}/candidate_quality*.csv")


if __name__ == "__main__":
    main()
