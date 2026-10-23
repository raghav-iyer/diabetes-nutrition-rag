"""Failure analysis: classify each failed question's root cause from the stage logs.

Buckets (first match wins, in this order):
  safety_miss       safety rule not applied (mode wrong, forbidden content, amounts leaked)
  wrong_abstention  abstained when it shouldn't, or answered when it should have abstained
  query_misread     query understanding got the population or answerability wrong
  not_in_corpus     no relevant document exists in the labels
  not_retrieved     relevant docs never reached the fused candidate lists (top 100s)
  filtered_out      relevant docs were removed by a metadata filter
  ranked_low        retrieved but lost in fusion / rerank (not in the final 10)
  ignored_by_llm    in the generator's context but not cited
  unfaithful        claims not supported by their cited source
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

BUCKETS = ["safety_miss", "wrong_abstention", "query_misread", "not_in_corpus", "not_retrieved", "filtered_out",
           "ranked_low", "ignored_by_llm", "unfaithful"]


def is_failure(row: dict[str, Any], item) -> bool:
    if row.get("abstention_correct") in (0.0, False) or row.get("safety_compliant") in (0.0, False):
        return True
    if item.should_abstain:
        return False
    if row.get("judge_score") is not None and row["judge_score"] == 0:
        return True
    if row.get("faithfulness") is not None and row["faithfulness"] < 0.8:
        return True
    return bool(item.graded_relevance) and row.get("recall@10") == 0.0


def classify(row: dict[str, Any], item) -> str:
    if row.get("safety_compliant") in (0.0, False):
        return "safety_miss"
    if row.get("abstention_correct") in (0.0, False):
        return "wrong_abstention"
    qa = row.get("analysis") or {}
    gold_pop = set(item.population) - {"general_adults"}
    if (qa.get("answerable") is False and not item.should_abstain) or \
       (gold_pop and qa.get("population_tags") is not None and not gold_pop & set(qa["population_tags"])):
        return "query_misread"
    relevant = set(item.relevant_doc_ids)
    if not relevant:
        return "not_in_corpus"
    st = row.get("stages") or {}
    fused = {x[1] for x in st.get("fused", []) if len(x) == 3}
    filtered = {x[1] for x in st.get("filtered_out", []) if len(x) == 3}
    final = {x[1] for x in st.get("final", [])}
    if not relevant & (fused | filtered):
        return "not_retrieved"
    if relevant & filtered and not relevant & final:
        return "filtered_out"
    if not relevant & final:
        return "ranked_low"
    cited = set(row.get("cited") or [])
    if row.get("faithfulness") is not None and row["faithfulness"] < 0.8:
        return "unfaithful"
    if relevant & set(row.get("sources") or []) and not relevant & cited:
        return "ignored_by_llm"
    return "unfaithful"


def analyse(rows: list[dict[str, Any]], items_by_id: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        it = items_by_id[r["question_id"]]
        if is_failure(r, it):
            out.append({"question_id": r["question_id"], "question": r["question"], "type": r["type"],
                        "topic": r["topic"], "auto_bucket": classify(r, it), "status": r.get("answer_status")
                        or r.get("retrieval_status"), "recall@10": r.get("recall@10"), "judge_score": r.get("judge_score"),
                        "faithfulness": r.get("faithfulness")})
    return out


def counts(failures: list[dict[str, Any]]) -> dict[str, int]:
    c = Counter(f["auto_bucket"] for f in failures)
    return {b: c[b] for b in BUCKETS if c[b]}


COLUMNS = ["question_id", "question", "type", "topic", "status", "recall@10", "judge_score", "faithfulness",
           "auto_bucket", "your_bucket", "notes"]


def write_csv(path: Path, failures: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for x in failures:
            w.writerow({**{k: x.get(k, "") for k in COLUMNS}, "your_bucket": "", "notes": ""})


def corrected_counts(path: Path) -> dict[str, int]:
    """Counts after the human's corrections (your_bucket overrides auto_bucket)."""
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    c = Counter((r.get("your_bucket") or "").strip() or r["auto_bucket"] for r in rows)
    return dict(c.most_common())
