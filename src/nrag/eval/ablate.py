"""Ablations: re-run retrieval evaluation with one component switched off at a time.

Models and connections are loaded once and shared by every variant; each variant only
changes the config the Retriever reads. Deltas vs the base run come with bootstrap 95%
confidence intervals over questions.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from nrag.eval import retrieval_metrics as rm


def apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    c = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        node = c
        *head, last = dotted.split(".")
        for h in head:
            if h not in node:
                raise KeyError(f"unknown config path {dotted!r}")
            node = node[h]
        if last not in node:
            raise KeyError(f"unknown config path {dotted!r}")
        node[last] = value
    return c


def run_variants(base_rows: list[dict[str, Any]], variants: dict[str, Callable[[], list[dict[str, Any]]]],
                 metrics: list[str]) -> dict[str, Any]:
    """variants: {name: fn returning per-question rows}. Returns per-variant means and deltas vs base."""
    base = {r["question_id"]: r for r in base_rows}
    out = {"base": {m: rm.mean([r.get(m) for r in base_rows]) for m in metrics}, "variants": {}}
    for name, fn in variants.items():
        rows = fn()
        common = [r for r in rows if r["question_id"] in base]
        out["variants"][name] = {
            "means": {m: rm.mean([r.get(m) for r in rows]) for m in metrics},
            "delta_vs_base": {m: rm.paired_delta([r.get(m) for r in common],
                                                 [base[r["question_id"]].get(m) for r in common]) for m in metrics},
        }
    return out
