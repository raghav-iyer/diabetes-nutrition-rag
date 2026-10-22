"""Phase 4 build: documents -> chunks (per strategy) -> embeddings -> pgvector + BM25.

Inputs: data/corpus/documents.jsonl, the active model's extractions, scores_v2.jsonl,
and full-text / guideline JSON. Outputs: the `chunks`/`parents`/`documents` tables,
a partial ANN index per strategy, a BM25 index per strategy, a manifest and a report.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from nrag.config import resolve_path
from nrag.index import bm25, chunk as chunking, store
from nrag.index.embed import Embedder, embedder_from_config


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_inputs(cfg: dict[str, Any], only_extracted: bool = False) -> dict[str, Any]:
    from nrag.extract import pico, runner

    statuses = set(cfg["index"]["include_statuses"])
    docs = [d for d in _jsonl(resolve_path(cfg["paths"]["documents"])) if d.get("status") in statuses]
    model = runner.active_model(cfg)
    slug = pico.model_slug(model, cfg["extract"]["think"].get(model))
    extractions = {r["doc_id"]: r["extraction"] for r in runner.load_results(cfg, slug) if r["status"] == "ok"}
    scores = {r["doc_id"]: r for r in _jsonl(resolve_path(cfg["score_v2"]["output"]))}
    if only_extracted:
        docs = [d for d in docs if d["doc_id"] in extractions or d.get("source_type") == "guideline"]
    fulltext = {}
    for d in docs:
        p = d.get("fulltext_path")
        if p and resolve_path(p).exists():
            fulltext[d["doc_id"]] = json.loads(resolve_path(p).read_text())
    table_summaries = {}
    if cfg["index"]["tables"]["summary"] == "llm":
        from nrag.index.tables import load_summaries
        table_summaries = load_summaries(cfg, model)
    return {"docs": docs, "extractions": extractions, "scores": scores, "fulltext": fulltext, "model": model,
            "table_summaries": table_summaries}


def chunk_corpus(cfg, inputs, strategy: str, tok, embed=None) -> tuple[list, list]:
    chunks, parents = [], []
    for d in inputs["docs"]:
        cs, ps = chunking.build_doc_chunks(d, strategy, tok, cfg, extraction=inputs["extractions"].get(d["doc_id"]),
                                           score=inputs["scores"].get(d["doc_id"]),
                                           fulltext=inputs["fulltext"].get(d["doc_id"]), embed=embed,
                                           table_summaries=inputs.get("table_summaries"))
        chunks += cs
        parents += ps
    return chunks, parents


def _token_stats(chunks) -> dict[str, Any]:
    out = {}
    for kind in sorted({c.kind for c in chunks}):
        n = sorted(c.n_tokens for c in chunks if c.kind == kind)
        out[kind] = {"count": len(n), "mean": round(statistics.mean(n), 1), "p50": n[len(n) // 2],
                     "p95": n[min(len(n) - 1, int(len(n) * 0.95))], "max": n[-1]}
    return out


def run(cfg: dict[str, Any], strategies: list[str] | None = None, index_type: str | None = None,
        only_extracted: bool = False, tok=None, embedder: Embedder | None = None, conn=None,
        write_report: bool = True) -> dict[str, Any]:
    ic = cfg["index"]
    strategies = strategies or [ic["default_strategy"]]
    index_type = index_type or ic["vector_index"]["default"]
    bad = set(strategies) - set(chunking.STRATEGIES)
    if bad:
        raise ValueError(f"unknown strategies: {sorted(bad)}")
    t0 = time.monotonic()
    inputs = load_inputs(cfg, only_extracted)
    if not inputs["docs"]:
        raise SystemExit("no documents to index; run `nrag ingest` first")
    tok = tok or chunking.HFTokenizer(ic["embedding_model"])
    embedder = embedder or embedder_from_config(cfg, resolve_path)
    conn = conn or store.connect(cfg)
    store.create_schema(conn, ic["embedding_dim"])

    metas = {d["doc_id"]: chunking.doc_meta(d, inputs["extractions"].get(d["doc_id"]), inputs["scores"].get(d["doc_id"]))
             for d in inputs["docs"]}
    # Documents no longer in the corpus are removed (cascades to chunks and parents).
    keep = list(metas)
    conn.execute("DELETE FROM documents WHERE NOT (doc_id = ANY(%s))", (keep,))
    store.upsert_documents(conn, [{"doc_id": k, "meta": v} for k, v in metas.items()])

    report: dict[str, Any] = {"documents": len(inputs["docs"]), "with_extraction": sum(d["doc_id"] in inputs["extractions"]
                              for d in inputs["docs"]), "with_fulltext": len(inputs["fulltext"]),
                              "extraction_model": inputs["model"], "index_type": index_type,
                              "table_summaries": len(inputs.get("table_summaries") or {}), "strategies": {}}
    parents_done = False
    for strategy in strategies:
        s0 = time.monotonic()
        before = dict(embedder.stats)
        chunks, parents = chunk_corpus(cfg, inputs, strategy, tok, embed=embedder.encode)
        t_chunk = time.monotonic() - s0
        if not parents_done:
            store.upsert_parents(conn, parents)
            parents_done = True
        e0 = time.monotonic()
        vectors = embedder.encode([c.embed_text for c in chunks])
        t_embed = time.monotonic() - e0
        l0 = time.monotonic()
        store.replace_chunks(conn, strategy, chunks, vectors)
        t_load = time.monotonic() - l0
        i0 = time.monotonic()
        idx = store.create_vector_index(conn, strategy, index_type, ic["vector_index"][index_type])
        t_index = time.monotonic() - i0
        b0 = time.monotonic()
        bm25.build(chunks, resolve_path(ic["bm25"]["dir"]) / strategy, ic["bm25"]["stemmer"])
        t_bm25 = time.monotonic() - b0
        report["strategies"][strategy] = {
            "chunks": len(chunks), "by_kind": dict(Counter(c.kind for c in chunks)), "tokens": _token_stats(chunks),
            "embeddings": {"computed": embedder.stats["computed"] - before["computed"],
                           "cached": embedder.stats["cached"] - before["cached"]},
            "seconds": {"chunk": round(t_chunk, 1), "embed": round(t_embed, 1), "load": round(t_load, 1),
                        "vector_index": round(t_index, 1), "bm25": round(t_bm25, 1)},
            "vector_index": idx,
        }
        print(f"[{strategy}] {len(chunks)} chunks, embedded {report['strategies'][strategy]['embeddings']}, "
              f"index {idx} in {t_index:.1f}s")
    report["table_sizes"] = store.table_sizes(conn)
    report["total_seconds"] = round(time.monotonic() - t0, 1)
    if write_report:
        manifest = resolve_path(ic["manifest"])
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(report, indent=1))
        rp = resolve_path(ic["report"])
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(render_report(report))
    return report


def render_report(r: dict[str, Any]) -> str:
    lines = ["# Index statistics", "",
             f"Documents indexed: {r['documents']} (with extraction: {r['with_extraction']}, with full text: "
             f"{r['with_fulltext']}). Extraction model: `{r['extraction_model']}`. Vector index: {r['index_type']}.", "",
             "| Strategy | Chunks | Section chunks | Tokens/section chunk (mean, p50, p95, max) | Embed s | Index s |",
             "|---|---|---|---|---|---|"]
    for s, m in r["strategies"].items():
        sec = m["tokens"].get("section")
        tok = f"{sec['mean']}, {sec['p50']}, {sec['p95']}, {sec['max']}" if sec else "-"
        lines.append(f"| {s} | {m['chunks']} | {m['by_kind'].get('section', 0)} | {tok} | "
                     f"{m['seconds']['embed']} | {m['seconds']['vector_index']} |")
    lines += ["", "Chunks by kind: " + "; ".join(f"{s}: {m['by_kind']}" for s, m in r["strategies"].items()),
              "", "Table sizes: " + ", ".join(f"{k} {v}" for k, v in r.get("table_sizes", {}).items()),
              "", f"Total build time: {r['total_seconds']} s"]
    return "\n".join(lines) + "\n"
