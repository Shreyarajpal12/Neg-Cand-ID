"""Logistic matrix factorization.

    score(u, i) = <p_u, q_i> + b_u + b_i

Used as the base scorer of the Bayesian (BNS) comparator in the
candidate-quality evaluation (experiments.negative_rules). It is trained on
the discovery log only.
"""
from __future__ import annotations

import torch.nn as nn


class LogisticMF(nn.Module):
    def __init__(self, n_users, n_items, dim=32):
        super().__init__()
        self.pu = nn.Embedding(n_users, dim)
        self.qi = nn.Embedding(n_items, dim)
        self.bu = nn.Embedding(n_users, 1)
        self.bi = nn.Embedding(n_items, 1)
        nn.init.normal_(self.pu.weight, std=0.01)
        nn.init.normal_(self.qi.weight, std=0.01)
        nn.init.zeros_(self.bu.weight)
        nn.init.zeros_(self.bi.weight)

    def forward(self, u, i):
        return ((self.pu(u) * self.qi(i)).sum(-1)
                + self.bu(u).squeeze(-1) + self.bi(i).squeeze(-1))
