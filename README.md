# Implicit Negative Candidate Discovery with Symbolic Evidence

An unobserved user-item interaction is not necessarily a negative. This code
first tests whether an unobserved interaction is a defensible negative, using
symbolic rules over user features and four separate criteria: Scope, Evidence
Strength, Hierarchical Contribution and Product Relevance. Validated
candidates fill a tiered negative budget, and samplers such as Random, BNS
and DNS can operate inside the admitted set. Positive Symbolic applies the
same evidence in reverse to protect likely positives.

The industrial dataset in the paper is proprietary and not included. This
repository covers the public experiments on KuaiRec, KuaiRand, MIND,
RetailRocket and Santander.

## Layout

| Folder | Contents |
|---|---|
| `symbolic/` | rule evaluation, the four criteria, rule discovery, negative ladder |
| `prep/` | data preparation for the five datasets |
| `conditions/` | KuaiRec sparse and MNAR training conditions |
| `models/` | LightGCN, DeepFM, DNS |
| `experiments/` | rule discovery, samplers, recommender training, LLM review, Santander |
| `analysis/` | result tables |

## Setup

```bash
pip install -r requirements.txt
```

Place the raw data under `data/`:

| Dataset | Source | Location |
|---|---|---|
| KuaiRec 2.0 | kuairec.com | `data/kuairec/` |
| KuaiRand-Pure | kuairand.com | `data/KuaiRand-Pure/data/` |
| MIND-small | msnews.github.io | `data/mind/MINDsmall_train/` |
| RetailRocket | Kaggle `retailrocket/ecommerce-dataset` | `data/retailrocket/` |
| Santander | Kaggle `santander-product-recommendation` | downloaded by `prep/download_santander.sh` |

```bash
python -m prep.kuairec_kuairand
python -m prep.mind
python -m prep.retailrocket
bash prep/download_santander.sh
python -m conditions.sparse_positives     # KuaiRec 5%, 2%, 1% positives
python -m conditions.missing_positives    # KuaiRec 5% + MNAR-75
```

## Running

Each run uses a dataset (`kuairec`, `mind`, `retailrocket`) and, for KuaiRec,
an optional condition (`pos05`, `pos02`, `pos01`, `pos05_mnar75`).

```bash
export NEGCAND_DATASET=kuairec NEGCAND_CONDITION=pos05_mnar75
python -m experiments.negative_rules
python -m experiments.positive_rules
for SD in 42 1 2; do NEGCAND_SEED=$SD python -m experiments.train_recommender; done
```

After all runs, build the result tables:

```bash
python -m analysis.summarize_natural
python -m analysis.summarize_sparse_mnar
python -m analysis.negative_precision
python -m analysis.negative_precision_report
```

Appendix candidate quality:

```bash
NEGCAND_DATASET=kuairec  python -m experiments.negative_rules
NEGCAND_DATASET=kuairand python -m experiments.negative_rules
python -m experiments.santander_candidate_quality
python -m experiments.santander_baselines
```

The optional LLM domain review (`experiments.llm_domain_review`) needs
`OPENAI_API_KEY` and a category-name file set in `NEGCAND_CATEGORY_NAMES`.

Samplers: `random`, `bns`, `bns_smooth`, `dns`, `negative_symbolic`,
`positive_symbolic`, `symbolic_random`, `symbolic_bns`,
`symbolic_bns_smooth`, `symbolic_dns`. Outputs go to `artifacts/` and
`results/`.
