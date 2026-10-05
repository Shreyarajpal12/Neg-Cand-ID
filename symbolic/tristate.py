"""Tri-state predicate evaluation and packed rule signatures.

Every predicate and every rule evaluates to TRUE, FALSE or UNKNOWN for a user.
UNKNOWN is never converted to FALSE: a user whose feature value is missing is
not a user with a zero value, and a rule built on absent data must not be
treated as a rule built on observed behavior.

A rule is therefore represented by two boolean masks over users:

    observable[i]   every feature the rule needs is present for user i
    true[i]         observable[i] and the rule holds for user i

The empirical identity of a rule is the pair (observable, true). Two rules
that hold for the same users are still different if they can be evaluated on
different populations, so rules are merged only when both masks are equal.
Masks are packed with numpy.packbits so that equality tests are a single bytes
comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def split_predicate(p: str) -> tuple:
    """'register_days<=30' -> ('register_days', '<=', 30.0).

    Only the operators '<=' and '==' occur in the predicate vocabulary.
    """
    if "<=" in p:
        col, thr = p.split("<=", 1)
        return col.strip(), "<=", float(thr)
    if "==" in p:
        col, thr = p.split("==", 1)
        return col.strip(), "==", float(thr)
    raise ValueError(f"unparseable predicate: {p!r}")


def canonical_rule_key(predicates) -> str:
    """Order-independent identity of a conjunction ('A AND B' == 'B AND A')."""
    return " AND ".join(sorted(predicates))


def eval_predicate_tristate(df: pd.DataFrame, pred: str) -> tuple:
    """Evaluate one atomic predicate over a DataFrame.

    Returns (observable, fires), two boolean arrays:
      observable[i]  the column is present and non-missing for row i
      fires[i]       observable[i] and the comparison holds

    A missing value gives observable=False and fires=False. Callers must read
    `fires` together with `observable`. An absent column is unknown for every
    row, so any rule that uses it cannot be evaluated.
    """
    col, op, thr = split_predicate(pred)
    if col not in df.columns:
        n = len(df)
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    observable = ~np.isnan(v)
    fires = np.zeros(len(v), dtype=bool)
    if op == "==":
        np.equal(v, thr, out=fires, where=observable)
    else:
        np.less_equal(v, thr, out=fires, where=observable)
    fires &= observable
    return observable, fires


def eval_rule_tristate(df: pd.DataFrame, predicates) -> tuple:
    """Evaluate a conjunction of predicates over a DataFrame.

    A conjunction is observable for a row only if every conjunct is
    observable for that row, even when another conjunct is already false.
    This keeps every comparison arm (rule, non-rule, baseline) restricted to
    users whose status is actually known.
    """
    observable = np.ones(len(df), dtype=bool)
    fires = np.ones(len(df), dtype=bool)
    for p in predicates:
        o, f = eval_predicate_tristate(df, p)
        observable &= o
        fires &= f
    fires &= observable
    return observable, fires


@dataclass(frozen=True)
class TriStateSignature:
    """Packed (observable, true) masks: the empirical identity of a rule."""
    n_rows: int
    observable: bytes
    true: bytes

    @staticmethod
    def from_masks(observable: np.ndarray, true_mask: np.ndarray) -> "TriStateSignature":
        assert observable.dtype == bool and true_mask.dtype == bool
        assert len(observable) == len(true_mask)
        # A rule can never hold where it cannot be evaluated.
        assert not np.any(true_mask & ~observable), "true outside observable"
        return TriStateSignature(
            n_rows=len(observable),
            observable=np.packbits(observable).tobytes(),
            true=np.packbits(true_mask).tobytes(),
        )

    def unpack_observable(self) -> np.ndarray:
        return np.unpackbits(np.frombuffer(self.observable, dtype=np.uint8),
                             count=self.n_rows).astype(bool)

    def unpack_true(self) -> np.ndarray:
        return np.unpackbits(np.frombuffer(self.true, dtype=np.uint8),
                             count=self.n_rows).astype(bool)

    @property
    def support(self) -> int:
        return int(self.unpack_true().sum())

    @property
    def n_observable(self) -> int:
        return int(self.unpack_observable().sum())

    @property
    def unknown_fraction(self) -> float:
        return 1.0 - (self.n_observable / self.n_rows) if self.n_rows else 0.0

    def key(self) -> tuple:
        """Hashable identity: same evaluable population and same satisfying rows."""
        return (self.n_rows, self.observable, self.true)
