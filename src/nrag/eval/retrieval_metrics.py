"""Retrieval metrics over ranked document ids (documents, not chunks: a paper is relevant or not).

recall@k, precision@k, MRR (first relevant), nDCG@k with graded relevance (gain 2^g - 1),
aggregated overall and per group, with bootstrap confidence intervals over questions.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    return len(set(ranked[:k]) & relevant) / k


def mrr(ranked: list[str], relevant: set[str]) -> float | None:
    if not relevant:
        return None
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], grades: dict[str, int], k: int) -> float | None:
    if not any(g > 0 for g in grades.values()):
        return None
    dcg = sum((2 ** grades.get(d, 0) - 1) / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1))
    ideal = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg else None


def per_question(ranked: list[str], grades: dict[str, int], relevant_min: int = 2,
                 ks: Iterable[int] = (5, 10, 20)) -> dict[str, float | None]:
    rel = {d for d, g in grades.items() if g >= relevant_min}
    out: dict[str, float | None] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked, rel, k)
        out[f"precision@{k}"] = precision_at_k(ranked, rel, k)
    out["mrr"] = mrr(ranked, rel)
    out["ndcg@10"] = ndcg_at_k(ranked, grades, 10)
    return out


def mean(vals: list[float | None]) -> float | None:
    v = [x for x in vals if x is not None]
    return round(sum(v) / len(v), 4) if v else None


def bootstrap_ci(vals: list[float | None], iters: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return None
    rng = random.Random(seed)
    stats = sorted(sum(rng.choice(v) for _ in v) / len(v) for _ in range(iters))
    return round(stats[int(0.025 * iters)], 4), round(stats[int(0.975 * iters) - 1], 4)


def aggregate(rows: list[dict[str, Any]], metric_keys: list[str], group_key: str | None = None,
              iters: int = 2000) -> dict[str, Any]:
    """rows: per-question dicts with metric values (+ group fields). Returns means (+ CIs overall)."""
    def summarise(rs):
        return {m: mean([r.get(m) for r in rs]) for m in metric_keys} | {"n": len(rs)}
    out = {"overall": summarise(rows)}
    out["overall_ci"] = {m: bootstrap_ci([r.get(m) for r in rows], iters) for m in metric_keys}
    if group_key:
        groups = defaultdict(list)
        for r in rows:
            groups[r[group_key]].append(r)
        out[f"by_{group_key}"] = {g: summarise(rs) for g, rs in sorted(groups.items())}
    return out


def paired_delta(a: list[float | None], b: list[float | None], iters: int = 2000,
                 seed: int = 0) -> dict[str, Any] | None:
    """Mean of (a - b) over questions where both exist, with a bootstrap 95% CI (questions resampled)."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    diffs = [x - y for x, y in pairs]
    rng = random.Random(seed)
    stats = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(iters))
    return {"delta": round(sum(diffs) / len(diffs), 4), "ci95": [round(stats[int(0.025 * iters)], 4),
            round(stats[int(0.975 * iters) - 1], 4)], "n": len(diffs)}
