"""Downstream results on the sparse and missing-positive KuaiRec conditions.

    python -m analysis.summarize_sparse_mnar

Reads results/kuairec_<condition>/downstream_s<seed>.csv for the conditions
pos05 (5% clean), pos02 (2% clean), pos01 (1% clean) and pos05_mnar75
(5% + MNAR-75), seeds 42, 1 and 2, and writes (results/summary/)

    sparse_mnar_per_seed.csv   every (condition, sampler, seed) row
    sparse_mnar_summary.csv    mean and sample SD over seeds per (condition, sampler)
    sparse_mnar_figure.csv     test PR-AUC of random, bns, dns and negative_symbolic
                               per condition (data of the sparse/MNAR figure)
    SPARSE_MNAR.md             one table per condition
"""
import glob
import os

import numpy as np
import pandas as pd

from experiments import samplers as S
from paths import RESULTS, SUMMARY

SEEDS = [42, 1, 2]
CONDITIONS = [("pos05", "5% clean"), ("pos02", "2% clean"), ("pos01", "1% clean"),
              ("pos05_mnar75", "5% + MNAR-75")]
FIGURE_SAMPLERS = [S.RANDOM, S.BNS, S.DNS, S.NEGATIVE_SYMBOLIC]
METRICS = [("test_pr_auc", "Test PR-AUC"), ("test_top1_hit", "Top-1"),
           ("val_pr_auc", "Val PR-AUC"), ("best_epoch", "Best epoch")]


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("-" if (isinstance(v, float) and np.isnan(v)) else
                                     (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def load(cell_dir):
    fs = sorted(glob.glob(f"{RESULTS}/{cell_dir}/downstream_s*.csv"))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return d[d.seed.isin(SEEDS)].drop_duplicates(["sampler", "seed"], keep="last")


def summarize(d):
    """Mean and sample SD over seeds per sampler, in the canonical sampler order."""
    g = d.groupby("sampler")
    t = pd.DataFrame({"seeds": g.seed.nunique()})
    for m, _ in METRICS:
        t[f"{m}_mean"] = g[m].mean(); t[f"{m}_sd"] = g[m].std(ddof=1)
    order = [s for s in S.ALL_SAMPLERS if s in t.index]
    return t.loc[order]


def main():
    os.makedirs(SUMMARY, exist_ok=True)
    per_seed, summ, L = [], [], ["# KuaiRec: sparse and missing positive feedback", "",
                                 "Mean ± sample SD over seeds 42, 1 and 2.", ""]
    for cond, label in CONDITIONS:
        d = load(f"kuairec_{cond}")
        if d is None:
            L += [f"## {label}", "", "_no results_", ""]
            continue
        per_seed.append(d.assign(condition_label=label))
        t = summarize(d)
        summ.append(t.reset_index().assign(condition=cond, condition_label=label))
        show = pd.DataFrame({"sampler": t.index, "seeds": t.seeds.to_numpy()})
        for m, name in METRICS:
            show[name] = [f"{a:.4f} ± {b:.4f}" if b == b else f"{a:.4f}"
                          for a, b in zip(t[f"{m}_mean"], t[f"{m}_sd"])]
        L += [f"## {label}", "", md(show), ""]
    if per_seed:
        pd.concat(per_seed).to_csv(f"{SUMMARY}/sparse_mnar_per_seed.csv", index=False)
        T = pd.concat(summ)
        T.to_csv(f"{SUMMARY}/sparse_mnar_summary.csv", index=False)
        F = T[T.sampler.isin(FIGURE_SAMPLERS)][["condition", "condition_label", "sampler", "seeds",
                                                "test_pr_auc_mean", "test_pr_auc_sd"]]
        F.to_csv(f"{SUMMARY}/sparse_mnar_figure.csv", index=False)
    open(f"{SUMMARY}/SPARSE_MNAR.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"wrote {SUMMARY}/SPARSE_MNAR.md")


if __name__ == "__main__":
    main()
