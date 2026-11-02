"""Golden set: schema, dev/test split, candidate pooling and relevance labelling.

Relevance is judged per document (0 = not relevant, 1 = marginal, 2 = relevant, 3 = directly answers).
Candidates are pooled from several systems (BM25, dense, the full pipeline, the LlamaIndex
baseline) so that no single system defines what "relevant" means. Labels come in tiers:
Gemma drafts a grade for each pair (tier 1, "silver"); a second, independent reviewer
re-grades it (`your_grade` in the label sheet). The reviewer is a second LLM by default
(`eval.label_review`, source "llm_review"), or a human (source "human", "gold"). The reviewer's
grade is the reference; the Gemma draft is kept in `draft_relevance` so every report can state
the agreement between the two tiers (a measure of label noise) next to the label counts.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nrag.extract.pico import json_payload
from nrag.extract.schemas import PopulationTag

QType = Literal["factual", "comparison", "safety", "conflicting", "unanswerable", "india_specific"]
Topic = Literal["diabetes_prevention", "weight_management", "maternal_nutrition", "gestational_diabetes"]
LabelSource = Literal["human", "llm_review", "llm"]


class GoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    question: str
    topic: Topic
    population: list[PopulationTag]
    type: QType
    relevant_doc_ids: list[str] = Field(default_factory=list)
    graded_relevance: dict[str, int] = Field(default_factory=dict)
    relevance_source: dict[str, LabelSource] = Field(default_factory=dict)
    draft_relevance: dict[str, int] = Field(default_factory=dict)       # tier-1 (Gemma) grades, for agreement
    reference_answer: str
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    should_abstain: bool = False
    expected_safety_mode: Literal["normal", "restricted", "supportive_only"] | None = None
    expert_verified: bool = False
    notes: str = ""


def load(path: Path) -> list[GoldenItem]:
    items = [GoldenItem.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [i.id for i in items]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        raise ValueError(f"duplicate golden ids: {dup}")
    return items


def save(path: Path, items: list[GoldenItem]) -> None:
    path.write_text("".join(i.model_dump_json(exclude_none=True) + "\n" for i in items))


def split(item_id: str, dev_fraction: float = 0.5, seed: str = "nrag") -> str:
    """Deterministic dev/test assignment by hashing the id (stable as items are added)."""
    h = int(hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest(), 16) % 1000
    return "dev" if h < dev_fraction * 1000 else "test"


def coverage(items: list[GoldenItem], relevant_min: int = 2, reviewer: str | None = None) -> dict[str, Any]:
    src = Counter(s for i in items for s in i.relevance_source.values())
    return {"items": len(items), "by_topic": dict(Counter(i.topic for i in items)),
            "by_type": dict(Counter(i.type for i in items)),
            "labelled": sum(1 for i in items if i.graded_relevance),
            "gold_labels": src["human"], "review_labels": src["llm_review"], "silver_labels": src["llm"],
            "reviewer": reviewer, "tier_agreement": label_agreement(items, relevant_min),
            "expert_verified": sum(i.expert_verified for i in items)}


def _kappa(pairs: list[tuple[int, int]], weighted: bool = False, k: int = 4) -> float | None:
    """Cohen's kappa over grades 0..k-1; weighted=True uses quadratic weights (ordinal grades)."""
    n = len(pairs)
    if not n:
        return None
    w = (lambda a, b: (a - b) ** 2 / (k - 1) ** 2) if weighted else (lambda a, b: float(a != b))
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    observed = sum(w(a, b) for a, b in pairs) / n
    expected = sum(ca[a] * cb[b] * w(a, b) for a in range(k) for b in range(k)) / (n * n)
    return round(1 - observed / expected, 4) if expected else 1.0


def label_agreement(items: list[GoldenItem], relevant_min: int = 2) -> dict[str, Any] | None:
    """Tier-1 draft vs reviewer grade on every reviewed pair: exact and binary agreement, kappas, flips."""
    pairs = [(i.draft_relevance[d], g) for i in items for d, g in i.graded_relevance.items()
             if i.relevance_source.get(d) in ("human", "llm_review") and d in i.draft_relevance]
    if not pairs:
        return None
    n = len(pairs)
    binary = [(int(a >= relevant_min), int(b >= relevant_min)) for a, b in pairs]
    return {"n": n, "exact": round(sum(a == b for a, b in pairs) / n, 4),
            "within_one": round(sum(abs(a - b) <= 1 for a, b in pairs) / n, 4),
            "kappa": _kappa(pairs), "weighted_kappa": _kappa(pairs, weighted=True),
            "binary_agreement": round(sum(a == b for a, b in binary) / n, 4), "binary_kappa": _kappa(binary, k=2),
            "draft_mean": round(sum(a for a, _ in pairs) / n, 3), "review_mean": round(sum(b for _, b in pairs) / n, 3),
            "became_relevant": sum(1 for a, b in binary if (a, b) == (0, 1)),
            "became_irrelevant": sum(1 for a, b in binary if (a, b) == (1, 0)),
            "confusion": {f"{a}->{b}": c for (a, b), c in sorted(Counter(pairs).items())}}


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------
def unique_docs(ranked: list[tuple[str, str]], depth: int) -> list[str]:
    """[(chunk_id, doc_id)] in rank order -> first `depth` distinct doc_ids."""
    out = []
    for _, doc in ranked:
        if doc not in out:
            out.append(doc)
        if len(out) >= depth:
            break
    return out


def pool(items: list[GoldenItem], systems: dict[str, Callable[[str], list[tuple[str, str]]]],
         depth: int) -> dict[str, dict[str, list[str]]]:
    """{question_id: {doc_id: [systems that retrieved it in their top `depth` docs]}}."""
    out: dict[str, dict[str, list[str]]] = {}
    for it in items:
        if it.should_abstain:
            continue                                  # nothing to label for abstention questions
        found: dict[str, list[str]] = {}
        for name, fn in systems.items():
            for doc in unique_docs(fn(it.question), depth):
                found.setdefault(doc, []).append(name)
        out[it.id] = found
    return out


# ---------------------------------------------------------------------------
# LLM draft grades ("silver")
# ---------------------------------------------------------------------------
GRADE_PROMPT = """You judge search results for a nutrition-research question. Grade how useful the paper is for
answering the question, using its title and abstract only:
3 = directly answers it (right population, intervention/exposure and outcome)
2 = relevant (answers part of it, or a close population/outcome)
1 = marginal (same topic, but would not help answer it)
0 = not relevant
Return JSON only: {"grade": <0-3>, "reason": "<one short sentence>"}"""


class _Grade(BaseModel):
    model_config = ConfigDict(extra="ignore")
    grade: int = Field(ge=0, le=3)
    reason: str = ""


def llm_grades(pairs: list[tuple[str, str, str]], chat: Callable[[list[dict]], str], cache_dir: Path,
               workers: int = 2) -> dict[tuple[str, str], dict[str, Any]]:
    """pairs: [(question, doc_id, doc_text)] -> {(question, doc_id): {"grade", "reason"}}; cached per pair."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    def one(q, doc, text):
        key = hashlib.sha256(json.dumps([GRADE_PROMPT, q, doc, text]).encode()).hexdigest()
        path = cache_dir / f"{key}.json"
        if path.exists():
            return (q, doc), json.loads(path.read_text())
        msgs = [{"role": "system", "content": GRADE_PROMPT}, {"role": "user", "content": f"Question: {q}\n\nPaper:\n{text}"}]
        for _ in range(3):
            try:
                g = _Grade.model_validate_json(json_payload(chat(msgs)))
                path.write_text(json.dumps(g.model_dump()))
                return (q, doc), g.model_dump()
            except (ValidationError, ValueError):
                continue
        return (q, doc), {"grade": None, "reason": "grading failed"}

    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(one, *p) for p in pairs]):
            k, v = fut.result()
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Labelling spreadsheet
# ---------------------------------------------------------------------------
LABEL_COLUMNS = ["question_id", "question", "doc_id", "title", "year", "design", "retrieved_by", "llm_grade",
                 "llm_reason", "your_grade", "notes", "abstract"]


def write_label_csv(path: Path, items: list[GoldenItem], pooled: dict[str, dict[str, list[str]]],
                    docs: dict[str, dict[str, Any]], grades: dict[tuple[str, str], dict[str, Any]]) -> int:
    by_id = {i.id: i for i in items}
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        w.writeheader()
        for qid, found in pooled.items():
            q = by_id[qid].question
            # Most systems first, then LLM grade: the likely-relevant papers come first.
            order = sorted(found, key=lambda d: (-len(found[d]), -((grades.get((q, d)) or {}).get("grade") or 0), d))
            for doc in order:
                d = docs.get(doc, {})
                g = grades.get((q, doc)) or {}
                w.writerow({"question_id": qid, "question": q, "doc_id": doc, "title": d.get("title", ""),
                            "year": d.get("year", ""), "design": d.get("design", ""),
                            "retrieved_by": ";".join(found[doc]), "llm_grade": "" if g.get("grade") is None else g["grade"],
                            "llm_reason": g.get("reason", ""), "your_grade": "", "notes": "",
                            "abstract": (d.get("abstract") or "")[:1500]})
                n += 1
    return n


def import_labels(items: list[GoldenItem], rows: list[dict[str, str]], relevant_min: int = 2,
                  review_source: LabelSource = "human") -> dict[str, int]:
    """The reviewer's grade (your_grade, recorded as `review_source`) wins over the tier-1 LLM draft, which is
    kept in draft_relevance; relevant_doc_ids = docs graded >= relevant_min."""
    by_id = {i.id: i for i in items}
    stats = Counter()
    for r in rows:
        it = by_id.get(r["question_id"])
        if it is None:
            stats["unknown_question"] += 1
            continue
        review, llm = str(r.get("your_grade") or "").strip(), str(r.get("llm_grade") or "").strip()
        if llm in {"0", "1", "2", "3"}:
            it.draft_relevance[r["doc_id"]] = int(llm)
        if review in {"0", "1", "2", "3"}:
            it.graded_relevance[r["doc_id"]], it.relevance_source[r["doc_id"]] = int(review), review_source
            stats[review_source] += 1
        elif llm in {"0", "1", "2", "3"}:
            it.graded_relevance[r["doc_id"]], it.relevance_source[r["doc_id"]] = int(llm), "llm"
            stats["llm"] += 1
        else:
            stats["unlabelled"] += 1
    for it in items:
        it.relevant_doc_ids = sorted(d for d, g in it.graded_relevance.items() if g >= relevant_min)
    return dict(stats)
