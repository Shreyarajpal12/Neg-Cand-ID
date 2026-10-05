"""File locations shared by every script.

All scripts are run as modules from the repository root, for example
``python -m experiments.negative_rules``. Data, artifacts and results are
placed under the repository root unless ``NEGCAND_ROOT`` points elsewhere.

    data/        raw downloads and prepared tables
    artifacts/   discovered rules, sampled candidates, DNS picks
    results/     candidate-quality tables, per-seed downstream results, logs

A *cell* is a dataset, optionally combined with a constructed training
condition (for example ``kuairec_pos01``). Each cell has its own artifact and
result directory.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
ROOT = str(Path(os.environ.get("NEGCAND_ROOT") or REPO))
DATA = f"{ROOT}/data"
ARTIFACTS = f"{ROOT}/artifacts"
RESULTS = f"{ROOT}/results"
SUMMARY = f"{RESULTS}/summary"

# Datasets prepared into the common (user_id, video_id, label, category) schema
# with temporal train / val / test splits and a per-user feature table.
SPLIT_DATASETS = ("mind", "retailrocket")


def cell(dataset: str, condition: str = "") -> str:
    """Directory name of a cell: the dataset, plus the condition if any."""
    return f"{dataset}_{condition}" if condition else dataset


def cell_from_env() -> tuple[str, str]:
    """(dataset, condition) from NEGCAND_DATASET and NEGCAND_CONDITION."""
    return (os.environ.get("NEGCAND_DATASET", "kuairec"),
            os.environ.get("NEGCAND_CONDITION", ""))


def kuairec_table(name: str) -> str:
    """Prepared KuaiRec interactions; name is 'big' or 'small'."""
    return f"{DATA}/kuairec/prepared/{name}.parquet"


def kuairand_table(name: str) -> str:
    """Prepared KuaiRand-Pure log; name is 'standard_train', 'standard_val' or 'random'."""
    return f"{DATA}/kuairand/prepared/{name}.parquet"


def split_table(dataset: str, split: str) -> str:
    """Prepared MIND or RetailRocket split; split is 'train', 'val' or 'test'."""
    return f"{DATA}/{dataset}/prepared/{split}.parquet"


def condition_dir(dataset: str, condition: str) -> str:
    """Directory holding train/val/test parquet files of a constructed condition."""
    return f"{DATA}/conditions/{dataset}/{condition}"


def load_users(dataset: str):
    """Per-user feature table of a dataset."""
    import pandas as pd
    if dataset in SPLIT_DATASETS:
        return pd.read_pickle(f"{DATA}/{dataset}/prepared/users.pkl")
    if dataset == "kuairec":
        return pd.read_csv(f"{DATA}/kuairec/user_features.csv")
    if dataset == "kuairand":
        return pd.read_csv(f"{DATA}/KuaiRand-Pure/data/user_features_pure.csv")
    raise ValueError(f"unknown dataset {dataset!r}")
