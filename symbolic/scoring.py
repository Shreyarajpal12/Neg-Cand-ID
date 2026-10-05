"""Hierarchical Contribution (H) and Product Relevance (R).

E, H and R answer different questions and are never averaged into one score:

  E  is the rule strong negative evidence for this product in absolute terms?
  H  does the rule isolate a more negative group than the part of its simpler
     parent rule that it leaves out?
  R  is that refinement specific to this product, or does it describe low
     engagement with every product?

A rule can have high E and H near zero (it is negative, but no more so than its
parent), or high H and R near zero (it refines its parent in the same way for
every product, so it measures general propensity rather than product fit).

H is an edge quantity over disjoint arms. For a child rule C of parent P:

    A = users satisfying C
    B = users satisfying P but not C        (the parent remainder)
    H_draw = 1 - q_C / q_remainder,          H = 10th percentile of H_draw

Comparing C with P itself would compare a set with its own superset. H is not
clipped: a negative H means the child is less negative than the remainder, so
the refinement goes the wrong way. When the remainder rate is near zero the
magnitude of a negative H is unreliable, but its sign is still meaningful; such
edges are flagged.

A child with several parents receives the minimum H over its parents (the
critical parent), so a child that adds little beyond any one simpler rule is
not rescued by a more favorable parent.

R evaluates the same critical edge on every other product where both arms
have at least MIN_ARM outcomes, and takes the 10th percentile of
H_target - median(H_other) per draw. R is not clipped. When no other product
can evaluate the edge, R is NOT_ESTIMABLE, which is different from R = 0.
"""
from __future__ import annotations

import numpy as np

from symbolic.evidence import (
    DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_DRAWS, DEFAULT_PERCENTILE, DEFAULT_SEED,
)

NOT_ESTIMABLE = "NOT_ESTIMABLE"
MIN_ARM = 5          # minimum outcomes in each arm of an H comparison


def hierarchical_gain(c_child, n_child, c_remainder, n_remainder,
                      alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA,
                      draws=DEFAULT_DRAWS, percentile=DEFAULT_PERCENTILE,
                      seed=DEFAULT_SEED):
    """H for one edge, over disjoint child and parent-remainder arms.

    Returns a posterior summary; `_draws` keeps the H draws in memory for R
    and is not serialized.
    """
    if n_child < MIN_ARM or n_remainder < MIN_ARM:
        return {"H": None, "estimable": False,
                "reason": f"arm too small (child={n_child}, remainder={n_remainder}, min={MIN_ARM})"}
    rng = np.random.default_rng(seed)
    q_c = rng.beta(c_child + alpha, (n_child - c_child) + beta, size=draws)
    q_rem = rng.beta(c_remainder + alpha, (n_remainder - c_remainder) + beta, size=draws)
    # The Jeffreys prior excludes exact zeros; the guard covers degenerate input.
    with np.errstate(divide="ignore", invalid="ignore"):
        H_draw = 1 - q_c / q_rem
    H_draw = H_draw[np.isfinite(H_draw)]
    if H_draw.size == 0:
        return {"H": None, "estimable": False, "reason": "no finite draws"}
    extreme = bool(np.percentile(H_draw, percentile) < -10)
    return {
        "H": float(np.percentile(H_draw, percentile)),
        "H_magnitude_unreliable": extreme,
        "H_magnitude_note": ("remainder arm has near-zero conversion rate; the "
                             "sign is meaningful, the magnitude is not"
                             ) if extreme else None,
        "H_median": float(np.percentile(H_draw, 50)),
        "H_q05": float(np.percentile(H_draw, 5)),
        "H_q10": float(np.percentile(H_draw, 10)),
        "H_q50": float(np.percentile(H_draw, 50)),
        "H_q90": float(np.percentile(H_draw, 90)),
        "H_q95": float(np.percentile(H_draw, 95)),
        "estimable": True, "reason": None, "seed": seed,
        "_draws": H_draw,
    }


def product_relevance(H_target_draws, other_draws, percentile=DEFAULT_PERCENTILE):
    """R for one edge: is the hierarchical gain specific to this product?

    `other_draws` holds one H draw array per other product for the same edge.
    For each draw the median across other products is subtracted from the
    target draw, so R carries the posterior uncertainty of both sides.
    """
    usable = [d for d in other_draws if d is not None and len(d)]
    if H_target_draws is None or not len(H_target_draws):
        return {"R": None, "estimable": False,
                "reason": "target edge H not estimable"}
    if not usable:
        return {"R": None, "estimable": False, "status": NOT_ESTIMABLE,
                "reason": "edge not estimable on any other product",
                "n_other_products": 0}
    m = min(min(len(d) for d in usable), len(H_target_draws))
    stack = np.vstack([d[:m] for d in usable])
    g_other = np.median(stack, axis=0)
    R_draw = H_target_draws[:m] - g_other
    return {
        "R": float(np.percentile(R_draw, percentile)),
        "R_median": float(np.percentile(R_draw, 50)),
        "estimable": True, "reason": None,
        "n_other_products": len(usable),
    }


def node_H_from_edges(edge_H):
    """Rule-level H: the H of the critical (minimum-H) parent edge.

    Returns (critical_parent_id, H, {H_min, H_median, H_max}).
    """
    vals = [(pid, v["H"]) for pid, v in edge_H.items()
            if v.get("estimable") and v.get("H") is not None]
    if not vals:
        return None, None, {"H_min": None, "H_median": None, "H_max": None}
    vals.sort(key=lambda x: x[1])
    critical_parent, h_node = vals[0]
    hs = [v for _, v in vals]
    return critical_parent, h_node, {
        "H_min": float(min(hs)),
        "H_median": float(np.median(hs)),
        "H_max": float(max(hs)),
    }
