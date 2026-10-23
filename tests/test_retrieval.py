import copy
import json

import pytest

from nrag.llm import LLMError, LLMResponse
from nrag.retrieve.pipeline import Retriever
from nrag.retrieve.query_understanding import QueryAnalysis, QueryUnderstanding
from nrag.retrieve.rerank import Reranker
from nrag.retrieve.rrf import concat, rrf

# --- fusion -------------------------------------------------------------------------

def test_rrf_scores_and_order():
    out = rrf({"a": ["x", "y", "z"], "b": ["y", "x", "w"]}, k=60)
    ids = [o[0] for o in out]
    assert ids[:2] == ["x", "y"]                       # tie broken by best single rank, then id
    assert out[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert dict((o[0], o[2]) for o in out)["w"] == {"b": 3}


def test_rrf_counts_duplicates_in_one_list_once():
    out = dict((o[0], o[1]) for o in rrf({"a": ["x", "x", "y"]}, k=60))
    assert out["x"] == pytest.approx(1 / 61)


def test_concat_keeps_first_occurrence_in_list_order():
    assert [o[0] for o in concat({"dense": ["a", "b"], "bm25": ["b", "c"]})] == ["a", "b", "c"]


# --- query understanding --------------------------------------------------------------

class FakeLLM:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = 0

    def chat(self, endpoint, model, messages, **kw):
        self.calls += 1
        c = self.contents.pop(0)
        if isinstance(c, Exception):
            raise c
        return LLMResponse(content=c, latency_s=0.5)


QA_JSON = json.dumps({"population_tags": ["pregnant"], "topics": ["gestational_diabetes"], "intent": "what_works",
                      "answerable": True, "answerable_reason": "trials exist",
                      "rewritten_query": "low glycaemic index diet gestational diabetes incidence", "keywords": ["low GI", "GDM"]})


def test_query_understanding_validates_retries_and_caches(tmp_path):
    llm = FakeLLM(['{"intent": "magic"}', QA_JSON])
    qu = QueryUnderstanding(llm, "http://x", "m", tmp_path)
    qa, info = qu.analyse("Does low GI diet prevent GDM?")
    assert qa.population_tags == ["pregnant"] and info["source"] == "llm" and llm.calls == 2
    qa2, info2 = QueryUnderstanding(FakeLLM([]), "http://x", "m", tmp_path).analyse("Does low GI diet prevent GDM?")
    assert info2["source"] == "cache" and qa2 == qa


def test_query_understanding_falls_back_to_original_question(tmp_path):
    qa, info = QueryUnderstanding(FakeLLM([LLMError("down")]), "http://x", "m", tmp_path).analyse("Q?")
    assert qa.rewritten_query == "Q?" and qa.answerable and info["source"] == "fallback"
    qa, info = QueryUnderstanding(None, None, "m", tmp_path, enabled=False).analyse("Q?")
    assert info["source"] == "disabled"


# --- pipeline ---------------------------------------------------------------------------

CHUNKS = {
    "c1": dict(doc_id="D1", text="Low GI diet reduced GDM in a meta-analysis", quality_score=0.9, source_type="review",
               population_tags=["pregnant"], superseded=False),
    "c2": dict(doc_id="D1", text="Methods of the meta-analysis", quality_score=0.9, source_type="review",
               population_tags=["pregnant"], superseded=False),
    "c3": dict(doc_id="D1", text="Results again", quality_score=0.9, source_type="review",
               population_tags=["pregnant"], superseded=False),
    "c4": dict(doc_id="D2", text="Old version of the guideline", quality_score=1.0, source_type="guideline",
               population_tags=["pregnant"], superseded=True),
    "c5": dict(doc_id="D3", text="Guideline: low GI diets recommended in pregnancy", quality_score=0.95,
               source_type="guideline", population_tags=["pregnant"], superseded=False),
    "c6": dict(doc_id="D4", text="Low GI diet and weight in men", quality_score=0.5, source_type="primary",
               population_tags=["general_adults"], superseded=False),
}
for cid, c in CHUNKS.items():
    c.update(chunk_id=cid, parent_id=f"{c['doc_id']}#s0", kind="section", section="results", title=f"Title {c['doc_id']}",
             url=None, year=2020, design="rct", topics=[], india_focused=False)


class FakeSearcher:
    def __init__(self, dense, bm25):
        self.d, self.b = dense, bm25
        self.queries = []

    def dense(self, q, k):
        self.queries.append(("dense", q))
        return self.d[:k]

    def bm25(self, q, k):
        self.queries.append(("bm25", q))
        return self.b[:k]

    def meta(self, ids):
        return {i: copy.deepcopy(CHUNKS[i]) for i in ids if i in CHUNKS}

    def parents(self, pids):
        return {p: {"section": "results", "title": "", "text": f"parent of {p}"} for p in pids}


def fake_rerank(scores):
    return Reranker("fake", score_fn=lambda q, ps: [scores.get(p.split("\n", 1)[1], 0.0) for p in ps])


RERANK = {CHUNKS["c1"]["text"]: 0.8, CHUNKS["c2"]["text"]: 0.7, CHUNKS["c3"]["text"]: 0.6,
          CHUNKS["c4"]["text"]: 0.9, CHUNKS["c5"]["text"]: 0.75, CHUNKS["c6"]["text"]: 0.4}


class FixedQU:
    def __init__(self, qa=None):
        self.qa = qa or QueryAnalysis.model_validate_json(QA_JSON)

    def analyse(self, q):
        return self.qa, {"source": "fixed"}


def _retriever(cfg, rerank_scores=RERANK, qa=None, **overrides):
    c = copy.deepcopy(cfg)
    for path, value in overrides.items():
        node = c["retrieval"]
        *head, last = path.split("__")
        for h in head:
            node = node[h]
        node[last] = value
    searcher = FakeSearcher(["c1", "c4", "c5", "c2", "c3", "c6"], ["c5", "c6", "c1"])
    return Retriever(c, FixedQU(qa), searcher, fake_rerank(rerank_scores)), searcher


def test_full_pipeline_order_filters_cap_and_guideline_first(cfg):
    r, searcher = _retriever(cfg)
    res = r.retrieve("Does low GI diet prevent GDM?")
    ids = [x["chunk_id"] for x in res.results]
    assert res.status == "ok" and not res.retried
    assert "c4" not in ids                                        # superseded guideline filtered
    assert ids[0] == "c5"                                         # relevant guideline (bonus) first
    assert sum(1 for x in res.results if x["doc_id"] == "D1") == 2 # per-document cap
    assert res.results[0]["parent_text"] == "parent of D3#s0"
    assert ("dense", "low glycaemic index diet gestational diabetes incidence") in searcher.queries
    assert res.stages["filtered_out"] == [("c4", "superseded")]
    a = cfg["retrieval"]["final"]["alpha"]
    assert res.results[0]["final_score"] == pytest.approx(a * 0.75 + (1 - a) * 0.95 + 0.05, abs=1e-4)


def test_population_boost_affects_fusion_order(cfg):
    def order(boost):
        r, _ = _retriever(cfg, rerank__enabled=False, filters__population_boost=boost)
        return [cid for cid, _ in r.retrieve("q").stages["after_filters"]]
    plain, boosted = order(1.0), order(3.0)
    assert plain.index("c6") < plain.index("c3")                  # c6 is in both lists: fusion favours it
    assert boosted.index("c6") > boosted.index("c3")              # a population match outweighs that


def test_switches_disable_stages(cfg):
    r, searcher = _retriever(cfg, bm25__enabled=False, use_rewrite=False)
    r.retrieve("q")
    assert all(kind == "dense" for kind, _ in searcher.queries) and len(searcher.queries) == 1
    r, _ = _retriever(cfg, filters__enabled=False)
    assert "c4" in [x["chunk_id"] for x in r.retrieve("q").results]
    r, _ = _retriever(cfg, final__enabled=False)
    res = r.retrieve("q")
    assert [x["final_score"] for x in res.results] == sorted([x["final_score"] for x in res.results], reverse=True)
    assert res.results[0]["final_score"] == res.results[0]["rerank_score"]


def test_low_confidence_retries_once_then_abstains(cfg):
    low = {k: 0.05 for k in RERANK}
    r, searcher = _retriever(cfg, rerank_scores=low)
    res = r.retrieve("q")
    assert res.retried and res.status == "insufficient_evidence"
    assert ("dense", "low GI GDM") in searcher.queries             # retry uses the keywords
    assert "retry_final" in res.stages


def test_unanswerable_question_short_circuits(cfg):
    qa = QueryAnalysis(answerable=False, answerable_reason="personal dosing", rewritten_query="x")
    r, searcher = _retriever(cfg, qa=qa)
    res = r.retrieve("How much metformin should I take?")
    assert res.status == "not_answerable" and res.results == [] and searcher.queries == []


def test_logs_written(cfg, tmp_path):
    r, _ = _retriever(cfg)
    r.log_dir = tmp_path
    r.retrieve("q")
    log = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert {"lists", "fused", "reranked", "final", "query_understanding"} <= set(log["stages"])
