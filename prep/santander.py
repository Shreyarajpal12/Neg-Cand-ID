"""Santander Product Recommendation: month-transition cases.

    python -m prep.santander            # full file
    python -m prep.santander 1000000    # first N rows only (quick check)

Input:  data/santander/train_ver2.csv
Output: data/santander/cache/{manifest.npy, trans_XX.npz, meta_XX.parquet}

A case is one (customer, month t) pair. Its eligible candidates are the 24
products the customer does not own at t; the label of each is whether it was
newly acquired at t + 1. Products already owned are ineligible, not negative.

Features are read at month t only; the t + 1 frame is used only to form the
label. Transitions are ordered by month: the last one is the test transition,
the one before it the validation transition, and all earlier ones are
training transitions.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

from paths import DATA

SRC = f"{DATA}/santander/train_ver2.csv"
CACHE = f"{DATA}/santander/cache"
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


PROD = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
        'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
        'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
        'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
        'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
        'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
CAT = ["sexo", "segmento", "canal_entrada", "nomprov", "ind_empleado", "pais_residencia",
       "indresi", "indext", "tiprel_1mes", "indrel_1mes"]
NUM = ["age", "antiguedad", "renta", "ind_nuevo", "indrel", "ind_actividad_cliente"]
KEY = ["fecha_dato", "ncodpers"]


def load(path=SRC, nrows=None):
    log(f"reading {path}")
    df = pd.read_csv(path, usecols=KEY + CAT + NUM + PROD, nrows=nrows,
                     dtype={c: "float32" for c in PROD}, low_memory=False)
    for c in PROD:
        df[c] = df[c].fillna(0).astype(np.int8)
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    for c in CAT:
        df[c] = df[c].astype("string").fillna("NA")
    df["month"] = pd.to_datetime(df["fecha_dato"]).dt.to_period("M").astype(str)
    df = df.drop(columns=["fecha_dato"])
    log(f"rows={len(df):,}  customers={df.ncodpers.nunique():,}  months={df.month.nunique()}")
    return df


def build_transitions(df):
    """Case table per transition: features at t, ownership at t, acquisitions at t + 1."""
    months = sorted(df.month.unique())
    log(f"months: {months[0]} .. {months[-1]}  ({len(months)})")
    own = df.set_index(["month", "ncodpers"])[PROD]
    parts = []
    for a, b in zip(months[:-1], months[1:]):
        ta = df[df.month == a]
        nb = own.loc[b] if b in own.index.get_level_values(0) else None
        if nb is None:
            continue
        nxt = nb.reindex(ta.ncodpers)
        keep = nxt.notna().all(axis=1).to_numpy()
        if keep.sum() == 0:
            continue
        cur = ta.loc[keep]
        nx = nxt[keep].to_numpy(dtype=np.int8)
        cu = cur[PROD].to_numpy(dtype=np.int8)
        parts.append({
            "t": a, "t1": b,
            "meta": cur[["ncodpers"] + CAT + NUM].reset_index(drop=True),
            "own_t": cu,
            "acq": ((cu == 0) & (nx == 1)).astype(np.int8),
            "eligible": (cu == 0).astype(np.int8),
        })
        log(f"  {a} -> {b}: cases={keep.sum():,}  "
            f"eligible pairs={int((cu==0).sum()):,}  "
            f"acquisitions={int(((cu==0)&(nx==1)).sum()):,}")
    return parts


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    nrows = int(sys.argv[1]) if len(sys.argv) > 1 else None
    df = load(nrows=nrows)
    parts = build_transitions(df)
    months = [p["t"] for p in parts]
    split = {"train": months[:-2], "validation": months[-2], "test": months[-1]}
    log(f"SPLIT  train t = {split['train'][0]} .. {split['train'][-1]}")
    log(f"       validation transition = {split['validation']} -> {parts[-2]['t1']}")
    log(f"       test transition       = {split['test']} -> {parts[-1]['t1']}")
    np.save(f"{CACHE}/manifest.npy",
            np.array(json.dumps({"months": months, "split": split,
                                 "products": PROD}), dtype=object))
    for i, p in enumerate(parts):
        np.savez_compressed(f"{CACHE}/trans_{i:02d}.npz",
                            own_t=p["own_t"], acq=p["acq"], eligible=p["eligible"])
        p["meta"].to_parquet(f"{CACHE}/meta_{i:02d}.parquet", index=False)
    log(f"cached {len(parts)} transitions")
