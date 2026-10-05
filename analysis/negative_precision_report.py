"""Report on negative precision, positive protection and DNS hardness.

    python -m analysis.negative_precision_report      # after analysis.negative_precision

Reads results/summary/{candidate_quality.csv, protected_set.csv} and the DNS
pick files, and writes

    results/summary/dns_hardness_bands.csv    contamination of DNS picks by hardness band
    results/summary/NEGATIVE_PRECISION.md     tables per cell

Hardness of a DNS pick is the percentile of its score among 20 reference
items scored by the same model at the same refresh. "Eligible pool" draws the
references from the pool the sampler searches (the full pool for dns, the
symbolic pool for symbolic_dns); "catalogue" draws them uniformly from all
items. Values are averaged over the logged refreshes up to the selected epoch.
"""
import os

import numpy as np
import pandas as pd

from analysis.negative_precision import CELLS, SEEDS, label_table, lookup, suf
from experiments import samplers as S
from paths import ARTIFACTS, SUMMARY

BANDS = [0.0, 0.5, 0.8, 0.9, 0.95, 1.0001]
PAIRS = [(S.RANDOM, S.SYMBOLIC_RANDOM), (S.BNS, S.SYMBOLIC_BNS),
         (S.BNS_SMOOTH, S.SYMBOLIC_BNS_SMOOTH), (S.DNS, S.SYMBOLIC_DNS)]


def pm(x, d=4):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return "-"
    return f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x.mean():.{d}f}"


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("-" if (isinstance(v, float) and np.isnan(v)) else
                                     (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def hardness_bands():
    """Contamination of DNS picks in each eligible-pool hardness band, per seed."""
    bands = []
    for ds, cond, cdir, src in CELLS:
        G = None
        for s in sorted(S.DNS_SAMPLERS):
            for seed in SEEDS:
                f = f"{ARTIFACTS}/{cdir}/dns_picks_{s}_K4{suf(seed)}.npz"
                if not os.path.exists(f):
                    continue
                if G is None:
                    G = label_table(src)
                z = np.load(f, allow_pickle=True)
                uids, iids, au = z["uids"], z["iids"], z["au"]
                for k in sorted(x for x in z.files if x.startswith("ep")):
                    ep = k[2:]
                    if f"pct_elig{ep}" not in z.files:
                        continue
                    lab, _, _ = lookup(G, uids[au], iids[z[k]])
                    b = np.digitize(z[f"pct_elig{ep}"], BANDS) - 1
                    for bi in range(len(BANDS) - 1):
                        mk = b == bi
                        n = int(mk.sum()); n1 = int((lab[mk] == 1).sum()); nl = int((lab[mk] >= 0).sum())
                        bands.append(dict(dataset=ds, condition=cond, sampler=s, seed=seed, epoch=int(ep),
                                          band=f"{BANDS[bi]:.2f}-{min(BANDS[bi + 1], 1):.2f}", n=n,
                                          contamination=n1 / max(n, 1),
                                          contamination_labeled=n1 / max(nl, 1)))
        del G
    B = pd.DataFrame(bands)
    if len(B):
        B = B.groupby(["dataset", "condition", "sampler", "seed", "band"], as_index=False) \
             .agg(n=("n", "mean"), contamination=("contamination", "mean"),
                  contamination_labeled=("contamination_labeled", "mean"))
    return B


def main():
    Q = pd.read_csv(f"{SUMMARY}/candidate_quality.csv")
    PR = pd.read_csv(f"{SUMMARY}/protected_set.csv")
    B = hardness_bands()
    if len(B):
        B.to_csv(f"{SUMMARY}/dns_hardness_bands.csv", index=False)
    cells = [(c[0], c[1]) for c in CELLS]

    L = ["# Quality of selected negatives", "",
         "Computed from the saved negatives of the downstream runs (seeds 42, 1, 2). Labels are "
         "joined after sampling. `labelled` columns use only candidates with a known outcome; "
         "unlabelled candidates are not counted as errors.", ""]

    L += ["## 1. Negative precision per cell (mean ± SD over seeds)", ""]
    for ds, cond in cells:
        rows = []
        for s in S.ALL_SAMPLERS:
            x = Q[(Q.dataset == ds) & (Q.condition == cond) & (Q.sampler == s)]
            if len(x) == 0:
                rows.append({"sampler": s, "n_selected": "missing"}); continue
            rows.append({"sampler": s, "n_selected": f"{x.n_selected.mean():,.0f}",
                         "neg. precision (labelled)": pm(x.negative_precision_labeled),
                         "contamination (labelled)": pm(x.positive_contamination_labeled),
                         "contamination (all selected)": pm(x.positive_contamination, 5),
                         "label coverage": pm(x.label_coverage),
                         "candidate coverage": pm(x.get("candidate_coverage"), 3),
                         "fallback": pm(x.get("fallback_fraction"))})
        L += [f"### {ds}, {cond}", "", md(pd.DataFrame(rows)), ""]

    L += ["## 2. Base sampler vs the same sampler inside the symbolic pool", "",
          "Contamination among labelled candidates, mean over seeds; `lower` counts the seeds "
          "where the symbolic layer is lower.", ""]
    rows = []
    for ds, cond in cells:
        for b, f in PAIRS:
            xb = Q[(Q.dataset == ds) & (Q.condition == cond) & (Q.sampler == b)].set_index("seed")
            xf = Q[(Q.dataset == ds) & (Q.condition == cond) & (Q.sampler == f)].set_index("seed")
            sd = xb.index.intersection(xf.index)
            if len(sd) == 0:
                continue
            for col, name in (("positive_contamination_labeled", "labelled"),
                              ("positive_contamination", "all selected")):
                d = xf.loc[sd, col] - xb.loc[sd, col]
                rows.append({"cell": f"{ds} {cond}", "pair": f"{b} -> {f}", "contamination": name,
                             "base": xb.loc[sd, col].mean(), "symbolic layer": xf.loc[sd, col].mean(),
                             "lower": f"{int((d < 0).sum())}/{len(sd)}"})
    L += [md(pd.DataFrame(rows)) if rows else "_no paired results_", ""]

    L += ["## 3. Positive Symbolic protection region", "",
          "Over every labelled pool pair (catalogue minus the user's observed training positives). "
          "`lift` = protected positive precision / base positive rate. Identical across seeds.", ""]
    if len(PR):
        P2 = PR.copy(); P2["cell"] = P2.dataset + " " + P2.condition
        L += [md(P2[["cell", "protected_set", "protected_n", "protected_true_positive",
                     "protected_positive_precision", "base_positive_rate", "precision_lift",
                     "protected_positive_recall", "pct_labelled_pool_protected"]]), ""]
    else:
        L += ["_no positive levels found_", ""]

    L += ["## 4. DNS hardness and contamination", ""]
    rows = []
    for ds, cond in cells:
        for s in (S.DNS, S.SYMBOLIC_DNS):
            x = Q[(Q.dataset == ds) & (Q.condition == cond) & (Q.sampler == s)]
            if len(x) == 0:
                continue
            rows.append({"cell": f"{ds} {cond}", "sampler": s,
                         "pct vs eligible pool (mean)": pm(x.get("hardness_pct_eligible"), 3),
                         "pct vs eligible pool (median)": pm(x.get("hardness_pct_eligible_median"), 3),
                         "pct vs catalogue (mean)": pm(x.get("hardness_percentile"), 3),
                         "contamination (all selected)": pm(x.positive_contamination, 5),
                         "contamination (labelled)": pm(x.positive_contamination_labeled)})
    L += [md(pd.DataFrame(rows)) if rows else "_no DNS results_", ""]
    if len(B):
        bb = B.groupby(["dataset", "condition", "sampler", "band"], sort=False).agg(
            share=("n", "sum"), contamination=("contamination", "mean"),
            contamination_labeled=("contamination_labeled", "mean")).reset_index()
        bb["share"] = bb.share / bb.groupby(["dataset", "condition", "sampler"]).share.transform("sum")
        L += ["### Contamination by eligible-pool hardness band", "", md(bb), ""]
    open(f"{SUMMARY}/NEGATIVE_PRECISION.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"wrote {SUMMARY}/NEGATIVE_PRECISION.md")


if __name__ == "__main__":
    main()
