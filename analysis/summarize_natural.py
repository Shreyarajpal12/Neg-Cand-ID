"""Downstream results on the natural datasets: KuaiRec, MIND and RetailRocket.

    python -m analysis.summarize_natural

Reads results/<dataset>/downstream_s<seed>.csv and
results/<dataset>/sampler_diagnostics_s<seed>.csv (seeds 42, 1, 2) and writes
(results/summary/)

    natural_per_seed.csv      every (dataset, sampler, seed) row
    natural_summary.csv       mean and sample SD over seeds per (dataset, sampler)
    natural_paired.csv        paired comparisons (symbolic layer vs base sampler,
                              positive vs negative symbolic)
    natural_diagnostics.csv   pool coverage, fallback and positive-evidence shares
    NATURAL.md                the same tables in Markdown

Natural data has no hidden-positive labels; see analysis.negative_precision
for candidate quality.
"""
import glob
import os

import numpy as np
import pandas as pd

from analysis.summarize_sparse_mnar import METRICS, SEEDS, load, md, summarize
from experiments import samplers as S
from paths import RESULTS, SUMMARY

DATASETS = [("kuairec", "KuaiRec"), ("mind", "MIND"), ("retailrocket", "RetailRocket")]
PAIRS = [(S.RANDOM, S.SYMBOLIC_RANDOM), (S.BNS, S.SYMBOLIC_BNS),
         (S.BNS_SMOOTH, S.SYMBOLIC_BNS_SMOOTH), (S.DNS, S.SYMBOLIC_DNS),
         (S.RANDOM, S.POSITIVE_SYMBOLIC), (S.NEGATIVE_SYMBOLIC, S.POSITIVE_SYMBOLIC)]
DIAG = ["symbolic_pool_coverage", "mean_safe_pool_size", "fallback_fraction", "mean_symbolic_tier",
        "pct_candidates_pos_evidence", "pct_candidates_strong_pos", "pct_sampled_pos_evidence",
        "users_primary_pool_short_frac", "hardness_percentile", "hardness_pct_eligible"]


def main():
    os.makedirs(SUMMARY, exist_ok=True)
    per_seed, summ, pairs, diags = [], [], [], []
    L = ["# Natural datasets", "", "Mean ± sample SD over seeds 42, 1 and 2.", ""]
    for ds, label in DATASETS:
        d = load(ds)
        if d is None:
            L += [f"## {label}", "", "_no results_", ""]
            continue
        per_seed.append(d.assign(dataset_label=label))
        t = summarize(d)
        summ.append(t.reset_index().assign(dataset=ds, dataset_label=label))
        show = pd.DataFrame({"sampler": t.index, "seeds": t.seeds.to_numpy()})
        for m, name in METRICS:
            show[name] = [f"{a:.4f} ± {b:.4f}" if b == b else f"{a:.4f}"
                          for a, b in zip(t[f"{m}_mean"], t[f"{m}_sd"])]
        L += [f"## {label}", "", md(show), ""]

        rows = []
        for b, f in PAIRS:
            if not ((d.sampler == b).any() and (d.sampler == f).any()):
                continue
            x = d[d.sampler == b].set_index("seed"); y = d[d.sampler == f].set_index("seed")
            sd = x.index.intersection(y.index)
            r = {"dataset": label, "base": b, "variant": f}
            for m, name in METRICS[:2]:
                dl = (y.loc[sd, m] - x.loc[sd, m])
                r[f"{name} base"] = x.loc[sd, m].mean(); r[f"{name} variant"] = y.loc[sd, m].mean()
                r[f"{name} diff"] = dl.mean(); r[f"{name} higher"] = f"{int((dl > 0).sum())}/{len(sd)}"
            rows.append(r)
        if rows:
            P = pd.DataFrame(rows); pairs.append(P)
            L += ["### Paired comparisons (diff = variant - base)", "", md(P.drop(columns="dataset")), ""]

        fs = sorted(glob.glob(f"{RESULTS}/{ds}/sampler_diagnostics_s*.csv"))
        if fs:
            mm = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
            mm = mm[mm.seed.isin(SEEDS)]
            cols = [c for c in DIAG if c in mm]
            M = mm.groupby("sampler")[cols].mean().reset_index()
            diags.append(M.assign(dataset=label))
            L += ["### Sampler diagnostics (mean over seeds)", "", md(M), ""]

    if per_seed:
        pd.concat(per_seed).to_csv(f"{SUMMARY}/natural_per_seed.csv", index=False)
        pd.concat(summ).to_csv(f"{SUMMARY}/natural_summary.csv", index=False)
    if pairs:
        pd.concat(pairs).to_csv(f"{SUMMARY}/natural_paired.csv", index=False)
    if diags:
        pd.concat(diags).to_csv(f"{SUMMARY}/natural_diagnostics.csv", index=False)
    open(f"{SUMMARY}/NATURAL.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"wrote {SUMMARY}/NATURAL.md")


if __name__ == "__main__":
    main()
