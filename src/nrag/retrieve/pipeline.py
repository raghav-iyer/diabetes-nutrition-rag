"""Phase 5 retrieval pipeline.

question -> query understanding -> BM25 + dense over original and rewritten queries
-> reciprocal rank fusion -> filters / population boost -> cross-encoder rerank
-> evidence-aware final ranking (alpha * rerank + (1 - alpha) * quality, guideline bonus)
-> per-document cap -> confidence check (one rewrite-and-retry, else "insufficient evidence").

Every stage can be switched off in config `retrieval` (for Phase 7 ablations), and every
stage's candidates and scores are logged per query (for failure analysis).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nrag.retrieve.query_understanding import QueryAnalysis, QueryUnderstanding
from nrag.retrieve.rerank import Reranker
from nrag.retrieve.rrf import concat, rrf

LOG_TOP = 100   # candidates kept per stage in the log


@dataclass
class RetrievalResult:
    question: str
    status: str                      # ok | insufficient_evidence | not_answerable
    analysis: dict[str, Any]
    results: list[dict[str, Any]]
    retried: bool = False
    stages: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__


class Retriever:
    def __init__(self, cfg: dict[str, Any], qu: QueryUnderstanding, searcher, reranker: Reranker | None,
                 log_dir: Path | None = None):
        self.cfg = cfg
        self.rc = cfg["retrieval"]
        self.qu, self.searcher, self.reranker, self.log_dir = qu, searcher, reranker, log_dir

    # -- one pass ---------------------------------------------------------------
    def _search(self, queries: dict[str, str], analysis: QueryAnalysis, stages: dict, timings: dict, tag: str):
        rc = self.rc
        lists: dict[str, list[str]] = {}
        t = time.monotonic()
        for name, text in queries.items():
            if rc["dense"]["enabled"]:
                lists[f"dense_{name}"] = self.searcher.dense(text, rc["dense"]["k"])
            if rc["bm25"]["enabled"]:
                lists[f"bm25_{name}"] = self.searcher.bm25(text, rc["bm25"]["k"])
        timings[f"{tag}search"] = round(time.monotonic() - t, 2)
        stages[f"{tag}lists"] = {k: v[:LOG_TOP] for k, v in lists.items()}

        fused = rrf(lists, rc["rrf_k"]) if rc["fusion"] == "rrf" else concat(lists)
        meta = self.searcher.meta([cid for cid, _, _ in fused])
        cands = []
        for cid, score, ranks in fused:
            m = meta.get(cid)
            if m is None:
                continue
            cands.append({**m, "fused_score": score, "ranks": ranks})
        stages[f"{tag}fused"] = [(c["chunk_id"], c["doc_id"], round(c["fused_score"], 5)) for c in cands[:LOG_TOP]]

        # Filters and population boost.
        f = rc["filters"]
        removed = []
        if f["enabled"]:
            qpop = set(analysis.population_tags) - {"general_adults"}
            kept = []
            for c in cands:
                if f["exclude_superseded"] and c.get("superseded"):
                    removed.append((c["chunk_id"], c["doc_id"], "superseded"))
                    continue
                c["population_match"] = bool(qpop & set(c.get("population_tags") or []))
                if c["population_match"]:
                    c["fused_score"] *= f["population_boost"]
                kept.append(c)
            cands = sorted(kept, key=lambda c: -c["fused_score"])
        stages[f"{tag}filtered_out"] = removed
        stages[f"{tag}after_filters"] = [(c["chunk_id"], c["doc_id"], round(c["fused_score"], 5)) for c in cands[:LOG_TOP]]

        # Cross-encoder rerank of the top_n.
        rr = rc["rerank"]
        pool = cands[:rr["top_n"]] if rr["enabled"] and self.reranker else cands
        t = time.monotonic()
        if rr["enabled"] and self.reranker and pool:
            scores = self.reranker.score(queries["original"], [f"{c['title']}\n{c['text']}" for c in pool])
            for c, s in zip(pool, scores):
                c["rerank_score"] = s
        else:
            top = pool[0]["fused_score"] if pool else 1.0
            for c in pool:
                c["rerank_score"] = c["fused_score"] / top if top else 0.0      # rank proxy in [0, 1]
        timings[f"{tag}rerank"] = round(time.monotonic() - t, 2)
        # Full candidate detail, so alpha / threshold / guideline bonus can be tuned offline from logs.
        stages[f"{tag}reranked"] = sorted(([c["chunk_id"], c["doc_id"], round(c["rerank_score"], 4),
                                            c.get("quality_score"), c.get("source_type")] for c in pool),
                                          key=lambda x: -x[2])

        # Evidence-aware final ranking.
        fc = rc["final"]
        threshold = rc["confidence"]["min_top_rerank"]
        for c in pool:
            q = c.get("quality_score") or 0.0
            if fc["enabled"]:
                c["final_score"] = fc["alpha"] * c["rerank_score"] + (1 - fc["alpha"]) * q
                if c.get("source_type") == "guideline" and c["rerank_score"] >= threshold:
                    c["final_score"] += fc["guideline_bonus"]
            else:
                c["final_score"] = c["rerank_score"]
        pool.sort(key=lambda c: -c["final_score"])

        per_doc: dict[str, int] = {}
        final = []
        for c in pool:
            if per_doc.get(c["doc_id"], 0) >= rc["max_chunks_per_doc"]:
                continue
            per_doc[c["doc_id"]] = per_doc.get(c["doc_id"], 0) + 1
            final.append(c)
            if len(final) >= rr["keep"]:
                break
        stages[f"{tag}final"] = [(c["chunk_id"], c["doc_id"], round(c["final_score"], 4)) for c in final]
        return final

    # -- public ---------------------------------------------------------------
    def retrieve(self, question: str) -> RetrievalResult:
        timings: dict[str, float] = {}
        stages: dict[str, Any] = {}
        t = time.monotonic()
        analysis, qu_info = self.qu.analyse(question)
        timings["query_understanding"] = round(time.monotonic() - t, 2)
        stages["query_understanding"] = {**analysis.model_dump(), **qu_info}
        if not analysis.answerable:
            return self._finish(RetrievalResult(question, "not_answerable", analysis.model_dump(), [], False, stages,
                                                timings))
        queries = {"original": question}
        if self.rc["use_rewrite"] and analysis.rewritten_query.strip() and analysis.rewritten_query != question:
            queries["rewritten"] = analysis.rewritten_query
        final = self._search(queries, analysis, stages, timings, "")
        status, retried = "ok", False
        conf = self.rc["confidence"]
        if conf["enabled"] and self._weak(final):
            if conf["retry"]:
                retried = True
                retry_q = " ".join(analysis.keywords) or analysis.rewritten_query
                final2 = self._search({"original": question, "rewritten": retry_q}, analysis, stages, timings, "retry_")
                if not self._weak(final2):
                    final = final2
            if self._weak(final):
                status = "insufficient_evidence"
        parents = self.searcher.parents(sorted({c["parent_id"] for c in final if c.get("parent_id")}))
        results = []
        for rank, c in enumerate(final, 1):
            p = parents.get(c.get("parent_id"), {})
            results.append({"rank": rank, "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "title": c["title"],
                            "url": c.get("url"), "kind": c["kind"], "section": c["section"], "text": c["text"],
                            "parent_text": p.get("text"), "year": c.get("year"), "design": c.get("design"),
                            "source_type": c.get("source_type"), "population_tags": c.get("population_tags"),
                            "quality_score": c.get("quality_score"), "rerank_score": round(c["rerank_score"], 4),
                            "final_score": round(c["final_score"], 4), "population_match": c.get("population_match"),
                            "ranks": c.get("ranks")})
        return self._finish(RetrievalResult(question, status, analysis.model_dump(), results, retried, stages, timings))

    def _weak(self, final: list[dict[str, Any]]) -> bool:
        if not final:
            return True
        if not (self.rc["rerank"]["enabled"] and self.reranker):
            return False                     # no calibrated score to judge confidence by
        return max(c["rerank_score"] for c in final) < self.rc["confidence"]["min_top_rerank"]

    def _finish(self, res: RetrievalResult) -> RetrievalResult:
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            name = hashlib.sha1(res.question.encode()).hexdigest()[:12]
            (self.log_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}.json").write_text(
                json.dumps(res.as_dict(), ensure_ascii=False, default=str, indent=1))
        return res


def retriever_from_config(cfg: dict[str, Any]) -> Retriever:
    from nrag.config import resolve_path
    from nrag.extract import runner
    from nrag.index import store
    from nrag.index.bm25 import BM25Index
    from nrag.index.embed import embedder_from_config
    from nrag.llm import client_from_config
    from nrag.retrieve.hybrid import HybridSearcher
    from nrag.retrieve.rerank import reranker_from_config

    rc = cfg["retrieval"]
    model = runner.active_model(cfg)
    provider = runner.provider_for(cfg, model)
    qu_cfg = rc["query_understanding"]
    client = client_from_config(cfg["llm"], provider) if qu_cfg["enabled"] else None
    endpoint = runner.endpoints(cfg, provider)[0] if qu_cfg["enabled"] else None
    qu = QueryUnderstanding(client, endpoint, model, resolve_path(cfg["paths"]["cache_dir"]) / "llm" / "query_understanding",
                            qu_cfg["max_attempts"], qu_cfg["enabled"])
    bm25_dir = resolve_path(cfg["index"]["bm25"]["dir"]) / rc["strategy"]
    searcher = HybridSearcher(store.connect(cfg), BM25Index(bm25_dir) if rc["bm25"]["enabled"] else None,
                              embedder_from_config(cfg, resolve_path), rc["strategy"], rc["dense"]["ef_search"])
    reranker = reranker_from_config(rc["rerank"]) if rc["rerank"]["enabled"] else None
    return Retriever(cfg, qu, searcher, reranker, resolve_path(rc["logs_dir"]))
