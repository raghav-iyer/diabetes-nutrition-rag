"""Rank fusion: reciprocal rank fusion (default) and plain concatenation (ablation baseline)."""

from __future__ import annotations


def rrf(ranked_lists: dict[str, list[str]], k: int = 60) -> list[tuple[str, float, dict[str, int]]]:
    """Fuse ranked id lists: score(id) = sum over lists of 1 / (k + rank), rank starting at 1.

    Returns [(id, score, {list_name: rank})] sorted by score (ties: best single rank, then id)."""
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, ids in ranked_lists.items():
        seen = set()
        for r, cid in enumerate(ids, 1):
            if cid in seen:          # an id listed twice in one list counts once (its best rank)
                continue
            seen.add(cid)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r)
            ranks.setdefault(cid, {})[name] = r
    return sorted(((cid, s, ranks[cid]) for cid, s in scores.items()),
                  key=lambda x: (-x[1], min(x[2].values()), x[0]))


def concat(ranked_lists: dict[str, list[str]]) -> list[tuple[str, float, dict[str, int]]]:
    """Ablation: lists one after another (in the given order), duplicates dropped; score = 1/position."""
    out, seen = [], set()
    for name, ids in ranked_lists.items():
        for r, cid in enumerate(ids, 1):
            if cid not in seen:
                seen.add(cid)
                out.append((cid, 1.0 / (len(out) + 1), {name: r}))
    return out
