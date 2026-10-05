"""Downstream backbones, the DNS selector and evaluation metrics.

    LightGCN  KuaiRec (natural and constructed conditions)
    DeepFM    MIND and RetailRocket

The backbone is fixed for every sampler on a dataset: architecture, optimizer,
learning rate, batch size, epochs, early stopping, seed and evaluation are
identical, and only the rows that enter the loss with label 0 change.

DNS is implemented here rather than with the pre-selecting samplers because
it needs the current model's scores: at each refresh it draws M = 20
candidates per positive from its pool, scores them with the current model and
keeps the K highest-scoring ones.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

SEED = 42


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pr_auc(y, s):
    """Average precision."""
    o = np.argsort(-s, kind="stable"); y = np.asarray(y)[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1); P = tp[-1]
    return float((prec * y).sum() / P) if P > 0 else float("nan")


def roc_auc(y, s):
    y = np.asarray(y).astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = np.empty(len(s)); r[np.argsort(s, kind="stable")] = np.arange(1, len(s) + 1)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def log_loss(y, s):
    p = 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def top1_hit(users, y, s):
    """User-level Top-1: is the highest-scored row of each user a positive?

    Only users with at least one positive in the slice are counted. The
    candidates are exactly the rows of the evaluation slice. Ties are broken
    by row order, which is the same for every sampler.

    Returns (hit_rate, n_users_evaluated, mean_positives_per_evaluated_user).
    """
    users = np.asarray(users); y = np.asarray(y)
    order = np.lexsort((-s, users))
    u_s, y_s = users[order], y[order]
    bounds = np.flatnonzero(np.r_[True, u_s[1:] != u_s[:-1], True])
    starts = bounds[:-1]
    npos = np.add.reduceat(y_s, starts)
    ok = npos >= 1
    if not ok.any():
        return float("nan"), 0, float("nan")
    hits = y_s[starts][ok]
    return (float(hits.mean()), int(ok.sum()), float(npos[ok].mean()))


def rank_metrics(users, y, s, k=10):
    """NDCG@k and Recall@k per user, averaged over users with at least one positive."""
    order = np.lexsort((-s, users)); u_s, y_s = users[order], np.asarray(y)[order]
    bounds = np.flatnonzero(np.r_[True, u_s[1:] != u_s[:-1], True])
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    nd, rc = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        rel = y_s[a:b]; tot = int(rel.sum())
        if tot == 0:
            continue
        top = rel[:k]
        nd.append((top * disc[:len(top)]).sum() / disc[:min(k, tot)].sum())
        rc.append(top.sum() / tot)
    return (float(np.mean(nd)) if nd else float("nan"),
            float(np.mean(rc)) if rc else float("nan"))


# ---------------------------------------------------------------------------
# LightGCN
# ---------------------------------------------------------------------------

def build_norm_adj(pos_u, pos_i, n_users, n_items, device):
    """Symmetrically normalized user-item adjacency built from training positives."""
    e = np.unique(np.stack([pos_u, pos_i]), axis=1)
    u, i = e[0], e[1]; n = n_users + n_items
    rows = np.concatenate([u, i + n_users]); cols = np.concatenate([i + n_users, u])
    deg = np.bincount(rows, minlength=n).astype(np.float32); deg[deg == 0] = 1.0
    vals = (1.0 / np.sqrt(deg[rows])) * (1.0 / np.sqrt(deg[cols]))
    idx = torch.as_tensor(np.stack([rows, cols]), dtype=torch.long)
    A = torch.sparse_coo_tensor(idx, torch.as_tensor(vals, dtype=torch.float32),
                                (n, n)).coalesce().to(device)
    return A, len(u)


class LightGCN(nn.Module):
    """LightGCN: no feature transform, no non-linearity, mean over layer outputs."""
    def __init__(self, n_users, n_items, A, dim=64, layers=2):
        super().__init__()
        self.n_users, self.A, self.layers = n_users, A, layers
        self.emb = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.emb.weight, std=0.1)

    def propagate(self):
        x = self.emb.weight; out = x
        for _ in range(self.layers):
            x = torch.sparse.mm(self.A, x); out = out + x
        return out / (self.layers + 1)

    def forward(self, u, i, E=None):
        E = self.propagate() if E is None else E
        return (E[u] * E[self.n_users + i]).sum(-1)


# ---------------------------------------------------------------------------
# DeepFM
# ---------------------------------------------------------------------------

class DeepFM(nn.Module):
    """DeepFM over categorical fields; field_dims is the vocabulary size per field."""
    def __init__(self, field_dims, dim=32, hidden=(256, 128), dropout=0.2):
        super().__init__()
        self.offsets = np.r_[0, np.cumsum(field_dims)[:-1]].astype(np.int64)
        total = int(sum(field_dims))
        self.emb = nn.Embedding(total, dim)
        self.lin = nn.Embedding(total, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.emb.weight); nn.init.zeros_(self.lin.weight)
        layers, d = [], len(field_dims) * dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*layers)
        self.register_buffer("_off", torch.as_tensor(self.offsets, dtype=torch.long))

    def forward(self, x):
        x = x + self._off
        e = self.emb(x)                                    # B x F x D
        first = self.lin(x).sum(dim=(1, 2)) + self.bias
        s = e.sum(1)
        second = 0.5 * ((s * s) - (e * e).sum(1)).sum(1)    # FM interaction
        deep = self.mlp(e.flatten(1)).squeeze(-1)
        return first + second + deep


# ---------------------------------------------------------------------------
# DNS
# ---------------------------------------------------------------------------

class DNSSelector:
    """Draw M candidates per positive from `pool_fn`, keep the top K by the model's score."""
    def __init__(self, anchor_u, pool_fn, K, M=20, seed=SEED, refresh_every=1):
        self.au, self.pool_fn, self.K, self.M = anchor_u, pool_fn, K, M
        self.rng = np.random.default_rng(seed); self.refresh_every = refresh_every
        self.cache = None; self.epoch = -1

    def select(self, epoch, score_fn):
        if self.cache is not None and (epoch - self.epoch) < self.refresh_every:
            return self.cache
        n = len(self.au)
        cand = self.pool_fn(self.au, self.M, self.rng)        # n x M item indices
        flat_u = np.repeat(self.au, self.M)
        s = score_fn(flat_u, cand.reshape(-1)).reshape(n, self.M)
        top = np.argpartition(-s, min(self.K, self.M - 1) - 1, axis=1)[:, :self.K]
        picked = np.take_along_axis(cand, top, axis=1)
        self.cache, self.epoch = (np.repeat(self.au, self.K), picked.reshape(-1)), epoch
        return self.cache
