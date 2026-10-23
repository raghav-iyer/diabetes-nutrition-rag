"""CLI-facing orchestration for Phase 7 (pooling, labels, eval runs, ablations, comparisons)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nrag.config import resolve_path
from nrag.eval import ablate, failures as fail_mod, golden, judge as judge_mod, report, run as run_mod


class Components:
    """Models and connections loaded once and shared across eval runs and ablation variants."""

    def __init__(self, cfg: dict[str, Any]):
        from nrag.extract import runner
        from nrag.index import store
        from nrag.index.embed import embedder_from_config
        from nrag.llm import client_from_config
        from nrag.retrieve.query_understanding import QueryUnderstanding
        from nrag.retrieve.rerank import reranker_from_config

        self.cfg = cfg
        self.model = runner.active_model(cfg)
        provider = runner.provider_for(cfg, self.model)
        self.client = client_from_config(cfg["llm"], provider)
        self.endpoint = runner.endpoints(cfg, provider)[0]
        self.conn = store.connect(cfg)
        self.embedder = embedder_from_config(cfg, resolve_path)
        self.reranker = reranker_from_config(cfg["retrieval"]["rerank"])
        self.qu = QueryUnderstanding(self.client, self.endpoint, self.model,
                                     resolve_path(cfg["paths"]["cache_dir"]) / "llm" / "query_understanding",
                                     cfg["retrieval"]["query_understanding"]["max_attempts"])
        self._searchers: dict[str, Any] = {}

    def chat(self, messages, max_tokens: int = 600) -> str:
        return self.client.chat(self.endpoint, self.model, messages, max_tokens=max_tokens).content

    def searcher(self, strategy: str):
        from nrag.index.bm25 import BM25Index
        from nrag.retrieve.hybrid import HybridSearcher

        if strategy not in self._searchers:
            bm25_dir = resolve_path(self.cfg["index"]["bm25"]["dir"]) / strategy
            self._searchers[strategy] = HybridSearcher(self.conn, BM25Index(bm25_dir) if bm25_dir.exists() else None,
                                                       self.embedder, strategy, self.cfg["retrieval"]["dense"]["ef_search"])
        return self._searchers[strategy]

    def retriever(self, cfg: dict[str, Any]):
        from nrag.retrieve.pipeline import Retriever

        rc = cfg["retrieval"]
        self.qu.enabled = rc["query_understanding"]["enabled"]
        return Retriever(cfg, self.qu, self.searcher(rc["strategy"]), self.reranker if rc["rerank"]["enabled"] else None,
                         resolve_path(rc["logs_dir"]))

    def answerer(self, cfg: dict[str, Any], retriever):
        from nrag.generate.answer import Answerer
        return Answerer(cfg, retriever, self.client, self.endpoint, self.model, resolve_path(cfg["generation"]["cache_dir"]))


def golden_items(cfg) -> list[golden.GoldenItem]:
    return golden.load(resolve_path(cfg["eval"]["golden"]))


def _documents(cfg) -> dict[str, dict[str, Any]]:
    with resolve_path(cfg["paths"]["documents"]).open() as f:
        return {d["doc_id"]: d for d in map(json.loads, f)}


# ---------------------------------------------------------------------------
def pool_and_grade(cfg, comp: Components, workers: int = 2) -> dict[str, Any]:
    ec = cfg["eval"]
    items = golden_items(cfg)
    docs = _documents(cfg)
    strategy = cfg["retrieval"]["strategy"]
    s = comp.searcher(strategy)
    k = ec["pool_depth"] * 5                                        # chunks, to get `pool_depth` distinct docs
    retr = comp.retriever(cfg)

    def chunk_doc(cids):
        meta = s.meta(cids)
        return [(c, meta[c]["doc_id"]) for c in cids if c in meta]

    systems = {"bm25": lambda q: chunk_doc(s.bm25(q, k)), "dense": lambda q: chunk_doc(s.dense(q, k)),
               "pipeline": lambda q: [(r["chunk_id"], r["doc_id"]) for r in retr.retrieve(q).results]}
    try:
        from nrag.retrieve.llamaindex_baseline import LlamaIndexBaseline
        li = LlamaIndexBaseline(cfg)
        systems["llamaindex"] = lambda q: [(r["chunk_id"], r["doc_id"]) for r in li.retrieve(q)]
    except Exception as e:                                          # baseline not built yet
        print(f"LlamaIndex baseline not used in the pool: {type(e).__name__}")
    pooled = golden.pool(items, systems, ec["pool_depth"])
    by_id = {i.id: i for i in items}
    pairs = [(by_id[qid].question, d, f"{docs.get(d, {}).get('title', '')}\n{docs.get(d, {}).get('abstract', '')}")
             for qid, found in pooled.items() for d in found]
    grades = golden.llm_grades(pairs, lambda m: comp.chat(m, 200), resolve_path(cfg["paths"]["cache_dir"]) / "llm" /
                               "relevance", workers=workers)
    n = golden.write_label_csv(resolve_path(ec["labels_csv"]), items, pooled, docs, grades)
    return {"questions": len(pooled), "pairs": n, "systems": list(systems),
            "llm_grade_counts": {g: sum(1 for v in grades.values() if v.get("grade") == g) for g in (0, 1, 2, 3, None)}}


def import_labels(cfg) -> dict[str, Any]:
    import csv
    ec = cfg["eval"]
    path = resolve_path(ec["labels_csv"])
    xlsx = path.with_suffix(".xlsx")
    if xlsx.exists() and xlsx.stat().st_mtime > path.stat().st_mtime:
        import openpyxl
        ws = openpyxl.load_workbook(xlsx, read_only=True).active
        rows_ = list(ws.iter_rows(values_only=True))
        hdr = [str(h) for h in rows_[0]]
        rows = [{k: ("" if v is None else str(v)) for k, v in zip(hdr, r)} for r in rows_[1:]]
    else:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    items = golden_items(cfg)
    stats = golden.import_labels(items, rows, ec["relevant_min"])
    golden.save(resolve_path(ec["golden"]), items)
    return {"import": stats, "coverage": golden.coverage(items)}


def evaluate(cfg, comp: Components, split: str = "all", answers: bool = True, label: str = "base",
             judge: bool = True) -> tuple[Path, dict[str, Any]]:
    ec = cfg["eval"]
    items = [i for i in golden_items(cfg) if split == "all" or golden.split(i.id) == split]
    by_id = {i.id: i for i in golden_items(cfg)}
    retriever = comp.retriever(cfg)
    answerer = comp.answerer(cfg, retriever) if answers else None
    embed = comp.embedder.encode
    rows = []
    for n, it in enumerate(items, 1):
        row = run_mod.evaluate_item(it, retriever, answerer,
                                    judge_chat=(lambda m: comp.chat(m, 200)) if (answers and judge) else None,
                                    judge_cache=resolve_path(cfg["paths"]["cache_dir"]) / "llm" / "judge",
                                    embed=embed, safety_cfg=cfg["safety"])
        rows.append(row)
        print(f"  [{n}/{len(items)}] {it.id} {row.get('answer_status') or row['retrieval_status']} "
              f"ndcg@10={row.get('ndcg@10')} ({row['seconds']}s)", flush=True)
    metrics = run_mod.summarise(rows, by_id)
    metrics["tuning"] = {"alpha": run_mod.tune_alpha(rows, by_id, ec["alphas"], cfg["retrieval"]),
                         "threshold": run_mod.tune_threshold(rows, by_id, ec["thresholds"])}
    fails = fail_mod.analyse(rows, by_id)
    metrics["failures"] = {"n": len(fails), "counts": fail_mod.counts(fails)}
    run_dir = run_mod.save_run(rows, metrics, resolve_path(ec["runs_dir"]), label,
                               {"retrieval": cfg["retrieval"], "generation": cfg["generation"]})
    fail_mod.write_csv(run_dir / "failures.csv", fails)
    agreement = None
    jv = resolve_path(ec["judge_validation"])
    if answers and judge:
        if jv.exists():
            import csv
            with jv.open(newline="") as f:
                agreement = judge_mod.agreement(list(csv.DictReader(f)))
        else:
            judge_mod.export_validation(jv, run_dir.name, [
                {"question_id": r["question_id"], "question": r["question"],
                 "reference_answer": by_id[r["question_id"]].reference_answer,
                 "system_answer": r.get("markdown") or "", "judge_score": int(r["judge_score"] * 2)
                 if r.get("judge_score") is not None else None, "judge_reason": r.get("judge_reason", "")}
                for r in rows if r.get("judge_score") is not None])
    examples = {}
    for f_ in fails:
        examples.setdefault(f_["auto_bucket"], []).append(f_)
    md = report.render(run_dir.name, metrics, tuning=metrics["tuning"], failures=fails,
                       failure_counts=metrics["failures"]["counts"], judge_agreement=agreement, examples=examples)
    (run_dir / "report.md").write_text(md)
    rp = resolve_path(ec["reports_dir"]) / f"run_{run_dir.name}.md"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(md)
    return run_dir, metrics


def ablations(cfg, comp: Components, split: str = "all") -> tuple[Path, dict[str, Any]]:
    ec = cfg["eval"]
    items = [i for i in golden_items(cfg) if (split == "all" or golden.split(i.id) == split)
             and i.graded_relevance and not i.should_abstain]
    by_id = {i.id: i for i in golden_items(cfg)}

    def retrieval_rows(c):
        r = comp.retriever(c)
        return [run_mod.evaluate_item(it, r) for it in items]

    base_rows = retrieval_rows(cfg)
    variants = {name: (lambda o=over: retrieval_rows(ablate.apply_overrides(cfg, o)))
                for name, over in ec["ablations"].items()}
    have = {r[0] for r in comp.conn.execute("SELECT DISTINCT strategy FROM chunks").fetchall()}
    for strat in ec["strategy_ablations"]:
        if strat in have:
            variants[f"strategy={strat}"] = lambda s=strat: retrieval_rows(ablate.apply_overrides(cfg, {"retrieval.strategy": s}))
    try:
        from nrag.eval import retrieval_metrics as rm
        from nrag.retrieve.llamaindex_baseline import LlamaIndexBaseline
        li = LlamaIndexBaseline(cfg)

        def li_rows():
            out = []
            for it in items:
                docs = []
                for r in li.retrieve(it.question):
                    if r["doc_id"] not in docs:
                        docs.append(r["doc_id"])
                out.append({"question_id": it.id, **rm.per_question(docs, it.graded_relevance)})
            return out
        variants["llamaindex_baseline"] = li_rows
    except Exception as e:
        print(f"LlamaIndex baseline skipped: {type(e).__name__}")
    result = ablate.run_variants(base_rows, variants, ec["ablation_metrics"])
    metrics = {"n_questions": len(items), "n_retrieval_labelled": len(items), "ablation": result,
               "label_sources": golden.coverage(list(by_id.values()))}
    run_dir = run_mod.save_run(base_rows, metrics, resolve_path(ec["runs_dir"]), "ablation", {"eval": ec})
    md = report.render(run_dir.name, {**metrics, "retrieval": None, "answers": None}, ablation=result)
    (run_dir / "report.md").write_text(md)
    (resolve_path(ec["reports_dir"]) / f"ablation_{run_dir.name}.md").write_text(md)
    return run_dir, result
