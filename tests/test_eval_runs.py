import csv

import pytest

from nrag.eval import ablate, answer_metrics as am, failures as fm, golden, judge, report, run as run_mod
from nrag.retrieve.pipeline import RetrievalResult


def item(**kw):
    base = dict(id="q1", question="Does X help?", topic="weight_management", population=["overweight_obese"],
                type="factual", reference_answer="X modestly helps weight loss.", must_include=["weight"],
                graded_relevance={"D1": 3, "D2": 2, "D3": 0}, relevant_doc_ids=["D1", "D2"])
    base.update(kw)
    return golden.GoldenItem(**base)


# --- answer metrics ---------------------------------------------------------------------------

def test_rouge_l():
    assert am.rouge_l("the cat sat", "the cat sat") == 1.0
    assert am.rouge_l("a b c d", "a c") == pytest.approx(2 * (2 / 4) * 1 / ((2 / 4) + 1), abs=1e-4)
    assert am.rouge_l("", "x") is None


def _out(**kw):
    o = {"status": "answered", "markdown": "X helps weight loss [D1].", "safety": {"mode": "normal"},
         "answer": {"direct_answer": "X modestly helps weight loss.", "evidence_strength": {"label": "moderate"},
                    "claims": [{"text": "X helps", "citations": ["D1", "D9"]}]},
         "verification": [{"claim": 1, "supported": "yes"}, {"claim": 2, "supported": "partial"}],
         "sources": {"D1": {}, "D2": {}, "D9": {}}}
    o.update(kw)
    return o


def test_score_answer_answered(cfg):
    m = am.score_answer(item(), _out(), cfg["safety"]["dose_pattern"], cfg["safety"]["dose_redaction"])
    assert m["abstention_correct"] and m["faithfulness"] == 0.75
    assert m["citation_precision"] == 0.5 and m["citation_recall"] == 0.5     # cited D1 of relevant {D1, D2} in context
    assert m["must_include_coverage"] == 1.0 and m["rouge_l"] == 1.0
    assert "safety_compliant" not in m


def test_score_answer_abstention_and_safety(cfg):
    sc = cfg["safety"]
    unans = item(type="unanswerable", should_abstain=True, graded_relevance={}, relevant_doc_ids=[])
    m = am.score_answer(unans, {"status": "not_answerable", "markdown": "can't", "safety": {"mode": "normal"}},
                        sc["dose_pattern"], sc["dose_redaction"])
    assert m["abstained"] and m["abstention_correct"]
    m = am.score_answer(unans, _out(), sc["dose_pattern"], sc["dose_redaction"])
    assert not m["abstention_correct"]
    saf = item(type="safety", expected_safety_mode="restricted", must_not_include=["you can stop"])
    leaked = _out(markdown="WHO says 60 mg daily. You can stop your tablets.", safety={"mode": "restricted"})
    m = am.score_answer(saf, leaked, sc["dose_pattern"], sc["dose_redaction"])
    assert m["safety_compliant"] is False and m["must_not_violations"] == ["you can stop"]
    ok = _out(markdown="WHO recommends iron [amount withheld].", safety={"mode": "restricted"})
    assert am.score_answer(saf, ok, sc["dose_pattern"], sc["dose_redaction"])["safety_compliant"] is True
    wrong_mode = item(type="safety", expected_safety_mode="normal")
    assert am.score_answer(wrong_mode, {"status": "supportive_only", "markdown": "", "safety": {"mode": "supportive_only"}},
                           sc["dose_pattern"], sc["dose_redaction"])["safety_compliant"] is False


# --- judge -----------------------------------------------------------------------------------

def test_judge_cached_and_retries(tmp_path):
    replies = iter(['{"score": 5}', '{"score": 2, "reason": "matches"}'])
    v = judge.judge("q", "ref", "ans", lambda m: next(replies), tmp_path)
    assert v["score"] == 2
    assert judge.judge("q", "ref", "ans", lambda m: 1 / 0, tmp_path)["score"] == 2        # cache hit


def test_judge_validation_export_and_agreement(tmp_path):
    rows = [{"question_id": f"q{i}", "question": "q", "reference_answer": "r", "system_answer": "a",
             "judge_score": i % 3, "judge_reason": ""} for i in range(9)]
    p = tmp_path / "jv.csv"
    assert judge.export_validation(p, "run1", rows, n=6) == 6
    out = list(csv.DictReader(p.open()))
    assert sorted(int(r["judge_score"]) for r in out) == [0, 0, 1, 1, 2, 2]               # stratified by score
    for r in out:
        r["your_score"] = r["judge_score"]
    assert judge.agreement(out) == {"n": 6, "accuracy": 1.0, "cohen_kappa": 1.0, "judge_mean": 1.0, "human_mean": 1.0}
    assert judge.agreement([{"judge_score": "1", "your_score": ""}]) is None


# --- runner ----------------------------------------------------------------------------------

STAGES = {"final": [("c1", "D2", 0.9), ("c2", "D1", 0.8)],
          "reranked": [["c1", "D2", 0.9, 0.5, "primary"], ["c2", "D1", 0.6, 0.95, "review"], ["c3", "D3", 0.95, 0.2, "primary"]],
          "after_filters": [("c4", "D4", 0.01)]}


def test_ranked_docs_order():
    assert run_mod.ranked_docs(STAGES) == ["D2", "D1", "D3", "D4"]


def test_offline_alpha_tuning_changes_order():
    assert run_mod.rerank_offline(STAGES["reranked"], 1.0, 0.0, 0.3, 2, 10) == ["D3", "D2", "D1"]
    assert run_mod.rerank_offline(STAGES["reranked"], 0.0, 0.0, 0.3, 2, 10) == ["D1", "D2", "D3"]


def test_tune_alpha_and_threshold(cfg):
    it = item()
    rows = [{"question_id": "q1", "split": "dev", "stages": STAGES, "retrieval_status": "ok"},
            {"question_id": "q1", "split": "test", "stages": STAGES, "retrieval_status": "ok"}]
    t = run_mod.tune_alpha(rows, {"q1": it}, [0.0, 1.0], cfg["retrieval"])
    assert t["best_alpha_dev"] == 0.0                         # quality-heavy ranking puts D1 (grade 3) first
    unans = item(id="q2", type="unanswerable", should_abstain=True)
    low = {"reranked": [["c", "D9", 0.1, 0.5, "primary"]]}
    rows2 = [{"question_id": "q1", "split": "dev", "stages": STAGES, "retrieval_status": "ok"},
             {"question_id": "q2", "split": "dev", "stages": low, "retrieval_status": "ok"}]
    th = run_mod.tune_threshold(rows2, {"q1": it, "q2": unans}, [0.05, 0.5])
    assert th["best_threshold_dev"] == 0.5 and th["by_split"]["dev"][0.5] == 1.0


def test_evaluate_item_with_fake_retriever(cfg):
    class R:
        def retrieve(self, q):
            return RetrievalResult(q, "ok", {"population_tags": ["overweight_obese"]}, [], False, STAGES, {"search": 0.1})

    row = run_mod.evaluate_item(item(), R())
    assert row["ranked_docs"][:2] == ["D2", "D1"] and row["recall@5"] == 1.0 and row["mrr"] == 1.0


def test_compare_runs():
    a = [{"question_id": "q1", "mrr": 1.0}, {"question_id": "q2", "mrr": 0.5}]
    b = [{"question_id": "q1", "mrr": 0.5}, {"question_id": "q2", "mrr": 0.5}]
    assert run_mod.compare_runs(a, b, ["mrr"])["mrr"]["delta"] == 0.25


# --- failures ------------------------------------------------------------------------------------

@pytest.mark.parametrize("row, kw, bucket", [
    ({"safety_compliant": 0.0}, {}, "safety_miss"),
    ({"abstention_correct": 0.0}, {}, "wrong_abstention"),
    ({"analysis": {"population_tags": ["pregnant"], "answerable": True}}, {}, "query_misread"),
    ({"analysis": {"population_tags": ["overweight_obese"]}}, {"relevant_doc_ids": []}, "not_in_corpus"),
    ({"analysis": {}, "stages": {"fused": [("c", "D7", 0.1)]}}, {}, "not_retrieved"),
    ({"analysis": {}, "stages": {"fused": [], "filtered_out": [("c", "D1", "superseded")]}}, {}, "filtered_out"),
    ({"analysis": {}, "stages": {"fused": [("c", "D1", 0.1)], "final": [("x", "D7", 0.5)]}}, {}, "ranked_low"),
    ({"analysis": {}, "stages": {"fused": [("c", "D1", 0.1)], "final": [("c", "D1", 0.5)]},
      "sources": ["D1"], "cited": ["D9"]}, {}, "ignored_by_llm"),
    ({"analysis": {}, "faithfulness": 0.5, "stages": {"fused": [("c", "D1", 0.1)], "final": [("c", "D1", 0.5)]}},
     {}, "unfaithful"),
])
def test_failure_buckets(row, kw, bucket):
    assert fm.classify(row, item(**kw)) == bucket


def test_failure_detection_and_corrected_counts(tmp_path):
    it = item()
    rows = [{"question_id": "q1", "question": "q", "type": "factual", "topic": "t", "recall@10": 0.0, "analysis": {},
             "stages": {}}]
    fails = fm.analyse(rows, {"q1": it})
    assert len(fails) == 1 and fails[0]["auto_bucket"] == "not_retrieved"
    p = tmp_path / "f.csv"
    fm.write_csv(p, fails)
    data = list(csv.DictReader(p.open()))
    data[0]["your_bucket"] = "not_in_corpus"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fm.COLUMNS)
        w.writeheader()
        w.writerows(data)
    assert fm.corrected_counts(p) == {"not_in_corpus": 1}
    assert not fm.is_failure({"recall@10": 0.6, "judge_score": 1.0}, it)


# --- ablations and report --------------------------------------------------------------------------

def test_apply_overrides(cfg):
    c = ablate.apply_overrides(cfg, {"retrieval.bm25.enabled": False, "retrieval.final.alpha": 1.0})
    assert c["retrieval"]["bm25"]["enabled"] is False and cfg["retrieval"]["bm25"]["enabled"] is True
    with pytest.raises(KeyError):
        ablate.apply_overrides(cfg, {"retrieval.nope": 1})


def test_run_variants_and_report():
    base = [{"question_id": "q1", "mrr": 1.0}, {"question_id": "q2", "mrr": 0.5}]
    res = ablate.run_variants(base, {"worse": lambda: [{"question_id": "q1", "mrr": 0.5}, {"question_id": "q2", "mrr": 0.5}]},
                              ["mrr"])
    assert res["base"]["mrr"] == 0.75 and res["variants"]["worse"]["delta_vs_base"]["mrr"]["delta"] == -0.25
    md = report.render("r1", {"n_questions": 2, "n_retrieval_labelled": 2, "label_sources": {"silver_labels": 4}},
                       ablation=res, failures=[], failure_counts={})
    assert "| worse | -0.250" in md and "Caveat" in md



def test_ragas_samples_skip_abstained_and_contextless_rows():
    from nrag.eval.ragas_eval import samples_from_rows
    rows = [{"question_id": "a", "question": "q", "direct_answer": "ans", "contexts": ["c"], "abstained": False},
            {"question_id": "b", "question": "q", "direct_answer": "ans", "contexts": [], "abstained": False},
            {"question_id": "c", "question": "q", "direct_answer": None, "contexts": ["c"], "abstained": True}]
    samples, ids = samples_from_rows(rows, {"a": "ref"})
    assert ids == ["a"] and samples[0].reference == "ref" and samples[0].retrieved_contexts == ["c"]
