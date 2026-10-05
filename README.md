# Implicit Negative Candidate Discovery with Symbolic Evidence

Recommender systems learn from observed interactions between users and items,
but they rarely observe explicit negative feedback. Negative samplers therefore
treat unobserved user-item interactions as negatives, although an unobserved
interaction does not show that the user is uninterested. This code first tests
whether an unobserved interaction is a defensible negative. Observed behavior
is expressed as short symbolic rules over user features, and each
rule-category pair is checked with four separate criteria: **Scope** (the rule
applies to users with observed outcomes), **Evidence Strength** (a
conservative Beta-Binomial lower bound on how much lower the positive rate is
under the rule), **Hierarchical Contribution** (the rule isolates a more
negative group than its simpler parent rule leaves behind), and **Product
Relevance** (the effect is specific to this category rather than general low
engagement). An optional LLM domain review then judges whether an admitted
rule has a plausible meaning; it never sees the statistics. Validated
candidates fill a tiered negative budget, and existing samplers such as
Random, BNS and DNS can operate inside the admitted candidate set. A mirrored
Positive Symbolic formulation finds regions where positives are more likely
and protects those candidates from being sampled as negatives.

The paper also reports results on a proprietary industrial dataset, which is
not included. This repository reproduces the public-dataset experiments on
KuaiRec, KuaiRand, MIND, RetailRocket and Santander.

## Repository layout

| Path | Contents |
|---|---|
| `paths.py` | data, artifact and result locations; `NEGCAND_ROOT` override |
| `symbolic/tristate.py` | missing-aware (observable, satisfied) rule evaluation |
| `symbolic/evidence.py` | Evidence Strength E and its positive mirror E_pos |
| `symbolic/scoring.py` | Hierarchical Contribution H and Product Relevance R |
| `symbolic/rules.py` | predicates, candidate rules, rule graph, S/E/H/R sweep, gates |
| `symbolic/discovery.py` | frozen discovery and the budget-matched candidate-quality comparison |
| `symbolic/ladder.py` | gate-ranked negative ladder (tiered negative budget) |
| `prep/` | preparation of KuaiRec, KuaiRand, MIND, RetailRocket and Santander |
| `conditions/` | KuaiRec sparse-positive and missing-positive training conditions |
| `models/recommenders.py` | LightGCN, DeepFM, DNS selector, metrics |
| `models/logistic_mf.py` | logistic MF used by the BNS candidate-quality comparator |
| `experiments/negative_rules.py` | negative rule discovery and intrinsic candidate quality |
| `experiments/positive_rules.py` | positive rule discovery (protection tiers) |
| `experiments/samplers.py` | symbolic samplers and sampler diagnostics |
| `experiments/train_recommender.py` | downstream training for every sampler |
| `experiments/llm_domain_review.py` | optional LLM domain review of admitted rules |
| `experiments/santander_candidate_quality.py`, `santander_baselines.py` | Santander candidate quality |
| `analysis/` | negative precision, protection quality, DNS hardness, result tables |

## Installation

Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

PyTorch uses a GPU when one is available. `openai` is needed only for the LLM
domain review and `kaggle` only for the Santander download.

All commands are run from the repository root as modules (`python -m ...`).
Data, artifacts and results are placed under the repository root; set
`NEGCAND_ROOT=/path` to keep `data/`, `artifacts/` and `results/` elsewhere.

## Data

Download the public datasets (each has its own license terms) and place them
as follows.

| Dataset | Source | Files |
|---|---|---|
| KuaiRec 2.0 | <https://kuairec.com> | `data/kuairec/{big_matrix.csv, small_matrix.csv, item_categories.csv, user_features.csv}` |
| KuaiRand-Pure | <https://kuairand.com> | `data/KuaiRand-Pure/data/{log_standard_4_08_to_4_21_pure.csv, log_standard_4_22_to_5_08_pure.csv, log_random_4_22_to_5_08_pure.csv, video_features_basic_pure.csv, user_features_pure.csv}` |
| MIND-small | <https://msnews.github.io> | `data/mind/MINDsmall_train/{behaviors.tsv, news.tsv}` |
| RetailRocket | Kaggle `retailrocket/ecommerce-dataset` | `data/retailrocket/{events.csv, item_properties_part1.csv, item_properties_part2.csv, category_tree.csv}` |
| Santander Product Recommendation | Kaggle competition `santander-product-recommendation` | `data/santander/train_ver2.csv` (downloaded by the script below) |

```bash
python -m prep.kuairec_kuairand     # data/kuairec/prepared, data/kuairand/prepared
python -m prep.mind                 # data/mind/prepared (temporal 70/15/15)
python -m prep.retailrocket         # data/retailrocket/prepared (active visitors, temporal 70/15/15)
bash prep/download_santander.sh     # Kaggle download, then python -m prep.santander
```

The Santander download needs Kaggle credentials in `~/.kaggle/kaggle.json` or
in `KAGGLE_USERNAME` / `KAGGLE_KEY`.

### KuaiRec training conditions

```bash
python -m conditions.sparse_positives    # pos05, pos02, pos01 (about 5%, 2%, 1% clean positives)
python -m conditions.missing_positives   # pos05_mnar75 (5% + MNAR-75)
```

Both builders are deterministic. `missing_positives` hides training positives
only and writes the true labels to `labels_clean.parquet`; no sampler or rule
reads that file, only `analysis.negative_precision`.

## Pipeline for one cell

A cell is a dataset (`kuairec`, `mind`, `retailrocket`) with an optional
KuaiRec condition (`pos05`, `pos02`, `pos01`, `pos05_mnar75`).

```bash
export NEGCAND_DATASET=kuairec NEGCAND_CONDITION=pos05_mnar75   # omit the condition for natural data

python -m experiments.negative_rules       # artifacts/<cell>/rules.csv, results/<cell>/candidate_quality.csv
python -m experiments.positive_rules       # artifacts/<cell>/positive_levels.parquet
for SD in 42 1 2; do
  NEGCAND_SEED=$SD python -m experiments.train_recommender   # results/<cell>/downstream_s$SD.csv
done
```

Rules are discovered on the training split only. `train_recommender` runs all
samplers by default; `NEGCAND_SAMPLERS=random,bns,...` selects a subset. Seeds
can run in parallel because each seed writes its own files.

The protocol is the default: K = 4 negatives per positive, binary
cross-entropy, Adam, batch size 4,096, at most 40 epochs, early stopping on
validation PR-AUC with patience 15, seeds 42, 1 and 2. LightGCN (64
dimensions, 2 layers, learning rate 5e-3) is used on KuaiRec and DeepFM (32
dimensions, MLP 256-128, dropout 0.2, learning rate 5e-4) on MIND and
RetailRocket. Every sampler draws K distinct negatives per positive from the
same pool (the catalogue minus the user's observed training positives).

### Samplers

| Paper name | Sampler name |
|---|---|
| Random | `random` |
| BNS | `bns` |
| BNS-S (smoothed BNS) | `bns_smooth` |
| DNS (top K of 20 by the current model) | `dns` |
| Negative Symbolic (Symbolic, Symbolic Alone) | `negative_symbolic` |
| Positive Symbolic | `positive_symbolic` |
| Random + Symbolic | `symbolic_random` |
| BNS + Symbolic | `symbolic_bns` |
| BNS-S + Symbolic | `symbolic_bns_smooth` |
| DNS + Symbolic | `symbolic_dns` |

Negative ladder. Each (user, category) receives the best level of any rule
that nominates it: 0 = S+E+H+R, 1 = S+E+H, 2 = S+E, 3 = E >= 0.35,
4 = E >= 0.20, 5 = E >= 0, 6 = no rule. `negative_symbolic` fills the budget
from level 0 downward, uniformly within a level. The `symbolic_*` samplers
keep the base sampler's rule and draw first from levels 0 to 2, then from
levels 3 to 4, then from the rest of the pool; `symbolic_dns` searches levels
0 to 2 when they hold at least 20 items, otherwise levels 0 to 4, otherwise
the full pool.

Positive tiers. E_pos mirrors E with the contrast reversed. Each
(user, category) gets a tier: 0 none, 1 weak (E_pos < 0.20), 2 medium
(< 0.45), 3 strong (>= 0.45). `positive_symbolic` samples tier 0 first and
reaches strongly supported positive regions only when nothing else is left.
Positive rules never add positives.

## Reproducing the paper tables

Run the pipeline above for the natural cells `kuairec`, `mind`,
`retailrocket` and the KuaiRec conditions `pos05`, `pos02`, `pos01`,
`pos05_mnar75`, then:

```bash
python -m analysis.summarize_natural          # results/summary/NATURAL.md, natural_*.csv
python -m analysis.summarize_sparse_mnar      # results/summary/SPARSE_MNAR.md, sparse_mnar_*.csv
python -m analysis.negative_precision         # results/summary/candidate_quality*.csv, protected_set.csv
python -m analysis.negative_precision_report  # results/summary/NEGATIVE_PRECISION.md, dns_hardness_bands.csv
```

| Paper table or figure | Where the numbers come from |
|---|---|
| Characteristics of the public datasets | printed by `prep.kuairec_kuairand`, `prep.mind`, `prep.retailrocket` |
| Negative and positive symbolic sampling on natural datasets | `natural_summary.csv` (random, bns, dns, negative_symbolic, positive_symbolic) |
| Quality of the strongest positive protection region | `protected_set.csv`, rows `strong (P=3)` |
| Symbolic candidate selection as a sampling layer | `natural_summary.csv` (random, symbolic_random, symbolic_bns_smooth, symbolic_dns, negative_symbolic; bns_smooth for the text) |
| Quality of selected negatives on KuaiRec MNAR-75 | `candidate_quality_mean.csv`, `negative_precision_labeled` for `5% + MNAR-75`; hardness bands in `dns_hardness_bands.csv` |
| Sparse and incomplete positive feedback (table and figure) | `sparse_mnar_summary.csv`; figure data in `sparse_mnar_figure.csv` |
| Appendix: candidate quality on KuaiRec and KuaiRand | `results/kuairec/candidate_quality.csv`, `results/kuairand/candidate_quality.csv` |
| Appendix: candidate quality on Santander | `results/santander/candidate_quality_full.csv` |

All files above are under `results/summary/` unless a path is given. No
plotting script is included.

Appendix candidate quality on KuaiRec (rules from the big matrix, evaluated on
the small matrix) and KuaiRand (rules from the standard log, evaluated on the
random-exposure log); every comparator receives the symbolic candidate budget:

```bash
NEGCAND_DATASET=kuairec  python -m experiments.negative_rules
NEGCAND_DATASET=kuairand python -m experiments.negative_rules
```

Santander (rules from a 150,000-case training sample, applied to the held-out
test transition):

```bash
python -m experiments.santander_candidate_quality   # Random, Bayesian, Symbolic ladder
python -m experiments.santander_baselines           # final table with Popularity and Product-Matched Random
```

## LLM domain review (optional)

`experiments.llm_domain_review` sends each statistically admitted
(rule, category) pair of a KuaiRec or KuaiRand cell to an OpenAI chat model
(default `gpt-4o-mini`, temperature 0). The request contains the rule in
plain language, the category's published name and feature definitions; a
banned-token check prevents any statistic from entering it. The verdict is
`BUSINESS_RELEVANT`, `NOT_RELEVANT` or `UNRESOLVED`.

```bash
export OPENAI_API_KEY=...
NEGCAND_CATEGORY_NAMES=/path/to/category_names.json \
  python -m experiments.llm_domain_review --dataset kuairec   # artifacts/kuairec/domain_review.csv
```

The category-name file maps each KuaiRec category id to its published
first-level category name (from `kuairec_caption_category.csv`) and is not
distributed with the code. None of the public-data results in the paper use
this review.

## Environment variables

| Variable | Used by | Purpose |
|---|---|---|
| `NEGCAND_ROOT` | all | root of `data/`, `artifacts/`, `results/` (default: repository root) |
| `NEGCAND_DATASET` | experiments | `kuairec`, `kuairand`, `mind`, `retailrocket` |
| `NEGCAND_CONDITION` | experiments | KuaiRec condition (`pos05`, `pos02`, `pos01`, `pos05_mnar75`) |
| `NEGCAND_SEED` | `train_recommender` | seed (default 42) |
| `NEGCAND_SAMPLERS` | `train_recommender` | comma-separated samplers (default: all) |
| `NEGCAND_REPLICATES` | `negative_rules` | category-matched replicates (default 200) |
| `NEGCAND_CATEGORY_NAMES` | `llm_domain_review` | category-name JSON |
| `OPENAI_API_KEY` | `llm_domain_review` | API key |

Keep credentials in the shell or in an untracked `.env` file. Run outputs
(`data/`, `artifacts/`, `results/`) are not tracked and can be regenerated
with the commands above. Rule discovery, the conditions and the pre-selected
negatives are deterministic given the seed; model scores (and therefore DNS
picks and downstream metrics) can differ slightly across hardware because of
floating-point non-determinism.
