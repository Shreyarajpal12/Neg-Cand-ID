"""Evidence Strength: how much lower the positive rate is under a rule.

For a rule r and a product (category) p, let c_r of n_r observed outcomes be
positive among users satisfying r, and c0_total of n0_total among all users
for whom r is observable. The observed effect is

    D_r = 1 - q_r / q_0

and Evidence Strength E_r is a conservative lower bound on D_r under a
Jeffreys Beta(0.5, 0.5) model, taken as the 10th percentile of 10,000
posterior draws and clipped to [0, 1].

Why the baseline is reconstructed. The rule-satisfying users are a subset of
the observable baseline population, so q_0 already contains the rule arm.
Drawing q_0 and q_r as two independent Beta posteriors would count the same
observations twice. The baseline is instead split into two disjoint arms,

    rule      : c_r,                  n_r
    non-rule  : c_notr = c0 - c_r,    n_notr = n0 - n_r

q_r and q_notr are drawn independently (the arms are disjoint), and q_0 is
reconstructed per draw as their support-weighted mixture:

    q_0^(b) = (n_r * q_r^(b) + n_notr * q_notr^(b)) / n0_total
    D^(b)   = 1 - q_r^(b) / q_0^(b)

When a rule covers a large share of the baseline, q_0 moves toward q_r in
proportion to that share, which is the intended behavior.

The function also reports the diagnostic contrast against non-rule users,
1 - q_r / q_notr. It is never used in E_r.

`positive_evidence` is the mirror statistic used by Positive Symbolic. It keeps
the prior, the number of draws, the reconstructed baseline and the
10th-percentile lower bound, and reverses only the contrast:
D_pos = q_r / q_0 - 1, so E_pos > 0 means the rule rate is above baseline.
"""
from __future__ import annotations

import numpy as np

DEFAULT_ALPHA = 0.5
DEFAULT_BETA = 0.5
DEFAULT_DRAWS = 10_000
DEFAULT_PERCENTILE = 10
DEFAULT_SEED = 20260811


def beta_binomial_effect(c_r: int, n_r: int, c0_total: int, n0_total: int,
                         alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA,
                         draws: int = DEFAULT_DRAWS, percentile: float = DEFAULT_PERCENTILE,
                         seed: int = DEFAULT_SEED) -> dict:
    """D_r and E_r with the reconstructed baseline.

    Returns a dict with D_p10, D_median, D_p90, E, estimable, reason and seed,
    plus the diagnostic rule-versus-non-rule contrast.
    """
    if n0_total <= 0:
        return _not_estimable(seed, "no observable baseline rows (n0_total == 0)")
    if n_r <= 0:
        return _not_estimable(seed, "no rule-satisfying rows (n_r == 0)")
    if n_r > n0_total:
        return _not_estimable(seed, "rule support exceeds baseline support (n_r > n0_total)")
    if c_r > n_r or c0_total > n0_total:
        return _not_estimable(seed, "conversions exceed support (schema violation)")

    n_notr = n0_total - n_r
    c_notr = c0_total - c_r

    rng = np.random.default_rng(seed)
    q_r = rng.beta(c_r + alpha, (n_r - c_r) + beta, size=draws)

    if n_notr == 0:
        # The rule covers the whole observable population, so q_0 equals q_r
        # by construction and D_r is exactly 0.
        return {
            "D_p10": 0.0, "D_median": 0.0, "D_p90": 0.0, "E": 0.0,
            "estimable": True, "reason": None, "seed": seed,
            "diagnostic_D_rule_vs_nonrule": None,
            "diagnostic_estimable": False,
            "diagnostic_reason": "no non-rule arm exists (n_notr == 0)",
        }

    q_notr = rng.beta(c_notr + alpha, (n_notr - c_notr) + beta, size=draws)
    q_0 = (n_r * q_r + n_notr * q_notr) / n0_total
    D = 1 - q_r / q_0

    d_p10 = float(np.percentile(D, percentile))
    d_median = float(np.percentile(D, 50))
    d_p90 = float(np.percentile(D, 90))
    e_r = float(min(1.0, max(0.0, d_p10)))

    D_rule_vs_nonrule = float(np.median(1 - q_r / q_notr))

    return {
        "D_p10": d_p10, "D_median": d_median, "D_p90": d_p90, "E": e_r,
        "estimable": True, "reason": None, "seed": seed,
        "diagnostic_D_rule_vs_nonrule": D_rule_vs_nonrule,
        "diagnostic_estimable": True,
        "diagnostic_reason": None,
    }


def _not_estimable(seed: int, reason: str) -> dict:
    return {
        "D_p10": None, "D_median": None, "D_p90": None, "E": None,
        "estimable": False, "reason": reason, "seed": seed,
        "diagnostic_D_rule_vs_nonrule": None,
        "diagnostic_estimable": False,
        "diagnostic_reason": reason,
    }


def positive_evidence(c_r: int, n_r: int, c0: int, n0: int, seed: int,
                      alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA,
                      draws: int = DEFAULT_DRAWS,
                      percentile: float = DEFAULT_PERCENTILE):
    """E_pos: `beta_binomial_effect` with the contrast reversed.

    Returns a float in [0, 1], or None when the counts are not estimable.
    """
    if n0 <= 0 or n_r <= 0 or n_r > n0 or c_r > n_r or c0 > n0:
        return None
    n_notr = n0 - n_r
    if n_notr == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    q_r = rng.beta(c_r + alpha, (n_r - c_r) + beta, size=draws)
    q_notr = rng.beta((c0 - c_r) + alpha, (n_notr - (c0 - c_r)) + beta, size=draws)
    q_0 = (n_r * q_r + n_notr * q_notr) / n0
    with np.errstate(divide="ignore", invalid="ignore"):
        d_pos = q_r / q_0 - 1.0
    return float(min(1.0, max(0.0, float(np.percentile(d_pos, percentile)))))
