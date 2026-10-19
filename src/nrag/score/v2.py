"""Quality score v2 and supersession (Phase 3).

v2 = v1 (design x size x recency, computed at ingest) x extraction-derived multipliers:
risk-of-bias signals, industry funding and extraction confidence. Un-extracted documents keep v1.
Supersession links an older guideline / review to a newer one on the same question; nothing is removed,
the older document gets `superseded_by` and a down-weight in ranking is left to retrieval.
"""

from __future__ import annotations

import json
import re
from typing import Any

from nrag.ingest.pubmed import size_factor

_STOP = set("""a an and the of in on for to with by from at as is are was were be been or vs versus among between
effect effects impact role association associations systematic review meta analysis metaanalysis randomized
randomised controlled trial trials study studies evidence guideline guidelines patients adults women people
health disease risk using use based new update updated diabetes diabetic mellitus gestational gdm pregnancy pregnant
supplementation supplement supplements intake dietary diet type prevention preventing prevent management outcomes
glycemic glycaemic control women""".split())


def tokens(text: str | None) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2 and t not in _STOP}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def load_extractions(path) -> dict[str, dict[str, Any]]:
    """doc_id -> validated extraction dict (latest 'ok' record wins)."""
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("extraction"):
                out[r["doc_id"]] = r["extraction"]
    return out


def rob_multiplier(cfg2: dict[str, Any], design: str, ex: dict[str, Any]) -> tuple[float, dict[str, float]]:
    signals = ex["risk_of_bias_signals"]
    parts: dict[str, float] = {}
    for group in cfg2["rob"].values():
        if design not in group["designs"]:
            continue
        for sig, table in group.items():
            if sig == "designs":
                continue
            parts[sig] = table[signals[sig]]
    lo, hi = cfg2["rob_clip"]
    prod = 1.0
    for v in parts.values():
        prod *= v
    return min(max(prod, lo), hi), parts


def score_doc(cfg: dict[str, Any], doc: dict[str, Any], ex: dict[str, Any] | None) -> dict[str, Any]:
    cfg2 = cfg["score_v2"]
    base = doc["quality_score"]
    row: dict[str, Any] = {"doc_id": doc["doc_id"], "quality_score_v1": base, "extracted": ex is not None,
                           "flags": list(doc.get("flags", []))}
    if ex is None:
        row.update(quality_score_v2=base, components={})
        return row
    comps: dict[str, Any] = {}
    q = base
    if cfg2["use_extracted_sample_size"] and doc.get("sample_size") is None and ex.get("sample_size"):
        sc = cfg["scoring"]
        old = size_factor(sc["size"], doc["design"], None)
        new = size_factor(sc["size"], doc["design"], ex["sample_size"])
        if old:
            q *= new / old
        comps["size_from_extraction"] = {"n": ex["sample_size"], "factor_ratio": round(new / old, 3) if old else None}
    rob, parts = rob_multiplier(cfg2, doc["design"], ex)
    if parts:
        q *= rob
        comps["rob"] = {"multiplier": round(rob, 3), **parts}
    ind = cfg2["industry_funding"][ex["funding_industry"]]
    if ind != 1.0:
        q *= ind
        comps["industry_funding"] = ind
        if ex["funding_industry"] == "yes":
            row["flags"].append("industry_funded")
    if ex["confidence"] == "low":
        q *= cfg2["low_confidence_multiplier"]
        row["flags"].append("low_extraction_confidence")
    row.update(quality_score_v2=round(q, 4), components=comps)
    return row


def _signature(doc: dict[str, Any], ex: dict[str, Any] | None) -> set[str]:
    return tokens(doc.get("title")) | (tokens(ex.get("intervention")) if ex else set())


def _pop_tags(doc: dict[str, Any], ex: dict[str, Any] | None) -> set[str]:
    return set(ex["population"]["tags"]) if ex else set()


def find_supersession(cfg: dict[str, Any], docs: list[dict[str, Any]],
                      exts: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """older doc_id -> {superseded_by, title_similarity, population_overlap}. Newest matching doc wins."""
    s = cfg["score_v2"]["supersession"]
    cands = [d for d in docs if d["design"] in s["designs"] and d.get("year") and not d.get("retracted")]
    sig = {d["doc_id"]: _signature(d, exts.get(d["doc_id"])) for d in cands}
    pop = {d["doc_id"]: _pop_tags(d, exts.get(d["doc_id"])) for d in cands}
    out: dict[str, dict[str, Any]] = {}
    for old in cands:
        best = None
        for new in cands:
            if new["year"] - old["year"] < s["min_year_gap"]:
                continue
            # a guideline can supersede a review, but a review never supersedes a guideline
            if old["design"] == "guideline" and new["design"] != "guideline":
                continue
            if s["require_shared_topic"] and not set(old["topics"]) & set(new["topics"]):
                continue
            ts = jaccard(sig[old["doc_id"]], sig[new["doc_id"]])
            po = jaccard(pop[old["doc_id"]], pop[new["doc_id"]])
            if ts < s["min_title_similarity"] or po < s["min_population_overlap"]:
                continue
            key = (new["year"], ts)
            if best is None or key > best[0]:
                best = (key, new["doc_id"], ts, po)
        if best:
            out[old["doc_id"]] = {"superseded_by": best[1], "title_similarity": round(best[2], 3),
                                  "population_overlap": round(best[3], 3)}
    return out


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")
