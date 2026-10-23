import csv
import json
import math
from pathlib import Path

import pytest

from nrag.eval import golden, retrieval_metrics as rm

ROOT = Path(__file__).resolve().parents[1]


def test_seed_golden_set_meets_brief():
    items = golden.load(ROOT / "eval" / "golden_set.jsonl")
    cov = golden.coverage(items)
    assert cov["items"] == 60 and set(cov["by_topic"].values()) == {15}
    assert cov["by_type"]["unanswerable"] >= 8 and cov["by_type"]["conflicting"] >= 8
    assert set(cov["by_type"]) == {"factual", "comparison", "safety", "conflicting", "unanswerable", "india_specific"}
    assert all(not i.expert_verified for i in items)
    assert all(i.should_abstain for i in items if i.type == "unanswerable")


def test_schema_rejects_bad_items(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({"id": "x", "question": "q", "topic": "astrology", "population": [], "type": "factual",
                             "reference_answer": ""}) + "\n")
    with pytest.raises(Exception):
        golden.load(p)


def test_split_is_deterministic_and_balanced():
    ids = [f"q{i}" for i in range(400)]
    s = [golden.split(i) for i in ids]
    assert s == [golden.split(i) for i in ids]
    assert 150 < s.count("dev") < 250


def test_pool_merges_systems_and_skips_abstention_items():
    items = [golden.GoldenItem(id="a", question="q1", topic="weight_management", population=[], type="factual",
                               reference_answer=""),
             golden.GoldenItem(id="b", question="q2", topic="weight_management", population=[], type="unanswerable",
                               reference_answer="", should_abstain=True)]
    systems = {"bm25": lambda q: [("c1", "D1"), ("c2", "D1"), ("c3", "D2")],
               "dense": lambda q: [("c9", "D3"), ("c1", "D1")]}
    pooled = golden.pool(items, systems, depth=2)
    assert set(pooled) == {"a"}
    assert pooled["a"] == {"D1": ["bm25", "dense"], "D2": ["bm25"], "D3": ["dense"]}


def test_label_csv_roundtrip_human_overrides_llm(tmp_path):
    items = [golden.GoldenItem(id="a", question="q1", topic="weight_management", population=[], type="factual",
                               reference_answer="")]
    pooled = {"a": {"D1": ["bm25", "dense"], "D2": ["bm25"]}}
    grades = {("q1", "D1"): {"grade": 3, "reason": "direct"}, ("q1", "D2"): {"grade": 1, "reason": "topic"}}
    path = tmp_path / "labels.csv"
    assert golden.write_label_csv(path, items, pooled, {"D1": {"title": "T1"}, "D2": {"title": "T2"}}, grades) == 2
    rows = list(csv.DictReader(path.open()))
    assert rows[0]["doc_id"] == "D1" and rows[0]["retrieved_by"] == "bm25;dense"
    rows[1]["your_grade"] = "2"                                      # human upgrades D2
    stats = golden.import_labels(items, rows)
    assert stats == {"llm": 1, "human": 1}
    assert items[0].graded_relevance == {"D1": 3, "D2": 2}
    assert items[0].relevance_source == {"D1": "llm", "D2": "human"}
    assert items[0].relevant_doc_ids == ["D1", "D2"]


def test_llm_grades_cached_and_validated(tmp_path):
    calls = []

    def chat(msgs):
        calls.append(1)
        return ['{"grade": 7}', '{"grade": 2, "reason": "ok"}'][len(calls) - 1] if len(calls) <= 2 else '{"grade": 2}'

    out = golden.llm_grades([("q", "D1", "text")], chat, tmp_path, workers=1)
    assert out[("q", "D1")]["grade"] == 2 and len(calls) == 2           # invalid grade retried
    golden.llm_grades([("q", "D1", "text")], chat, tmp_path, workers=1)
    assert len(calls) == 2                                               # cached


# --- retrieval metrics (hand-computed) ---------------------------------------------------------

RANKED = ["A", "X", "B", "Y", "C"]
GRADES = {"A": 3, "B": 2, "C": 1, "D": 3}


def test_recall_precision_mrr():
    rel = {d for d, g in GRADES.items() if g >= 2}                      # A, B, D
    assert rm.recall_at_k(RANKED, rel, 5) == pytest.approx(2 / 3)
    assert rm.precision_at_k(RANKED, rel, 5) == pytest.approx(2 / 5)
    assert rm.mrr(RANKED, rel) == 1.0
    assert rm.mrr(["X", "B"], rel) == 0.5
    assert rm.mrr(["X"], rel) == 0.0
    assert rm.recall_at_k(RANKED, set(), 5) is None


def test_ndcg():
    dcg = (2 ** 3 - 1) / math.log2(2) + (2 ** 2 - 1) / math.log2(4) + (2 ** 1 - 1) / math.log2(6)
    idcg = (7 / math.log2(2)) + (7 / math.log2(3)) + (3 / math.log2(4)) + (1 / math.log2(5))
    assert rm.ndcg_at_k(RANKED, GRADES, 10) == pytest.approx(dcg / idcg)
    assert rm.ndcg_at_k(["A", "D", "B", "C"], GRADES, 10) == pytest.approx(1.0)
    assert rm.ndcg_at_k(RANKED, {"A": 0}, 10) is None


def test_aggregate_and_paired_delta():
    rows = [{"type": "factual", "mrr": 1.0}, {"type": "factual", "mrr": 0.5}, {"type": "safety", "mrr": None}]
    agg = rm.aggregate(rows, ["mrr"], "type", iters=200)
    assert agg["overall"]["mrr"] == 0.75 and agg["by_type"]["safety"]["mrr"] is None
    d = rm.paired_delta([1.0, 0.5, 1.0], [0.5, 0.5, None], iters=200)
    assert d["delta"] == 0.25 and d["n"] == 2
    assert rm.paired_delta([1.0], [0.5]) is None
