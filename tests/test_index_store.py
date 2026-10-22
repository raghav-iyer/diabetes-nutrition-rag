"""pgvector store, BM25 and the end-to-end build, against a throwaway `nrag_test` database.

Skipped automatically when the local AlloyDB Omni container (POSTGRES_PASSWORD) isn't available.
"""

import json

import numpy as np
import pytest

from nrag.config import env
from nrag.index import bm25, build, chunk as ch, store
from nrag.index.embed import EmbeddingCache, Embedder
from test_index_chunk import DOC, EXTRACTION, FULLTEXT, TOK, para

DIM = 8


def _db_cfg(cfg, dbname="nrag_test"):
    return {**cfg, "index": {**cfg["index"], "db": {**cfg["index"]["db"], "dbname": dbname}, "embedding_dim": DIM}}


@pytest.fixture(scope="module")
def conn(cfg):
    try:
        import psycopg
        if not env(cfg["index"]["db"]["password_env"]):
            pytest.skip("no database password")
        admin = store.connect(cfg)
    except Exception as e:      # container not running
        pytest.skip(f"database unavailable: {type(e).__name__}")
    admin.execute("DROP DATABASE IF EXISTS nrag_test WITH (FORCE)")
    admin.execute("CREATE DATABASE nrag_test")
    admin.close()
    c = store.connect(_db_cfg(cfg))
    yield c
    c.close()


def fake_embed(texts):
    """Deterministic unit vectors: texts sharing words get similar vectors."""
    out = []
    for t in texts:
        v = np.zeros(DIM, dtype=np.float32)
        for w in t.lower().split():
            v[hash(w.strip(".,")) % DIM] += 1
        out.append(v / (np.linalg.norm(v) or 1))
    return np.stack(out)


def test_schema_upsert_search_and_all_index_types(conn, cfg):
    conn.execute("DROP TABLE IF EXISTS chunks, parents, documents CASCADE")
    store.create_schema(conn, DIM)
    chunks, parents = ch.build_doc_chunks(DOC, "token_aware", TOK, {"index": {**cfg["index"], "chunk": {
        "target_tokens": 40, "max_tokens": 50, "min_tokens": 8, "overlap_tokens": 5, "overlap_sentences": 1,
        "hierarchical": {"child_tokens": 15, "child_overlap": 3}, "semantic": {"breakpoint_percentile": 20}}}},
        extraction=EXTRACTION, fulltext=FULLTEXT)
    store.upsert_documents(conn, [{"doc_id": DOC["doc_id"], "meta": chunks[0].meta}])
    store.upsert_parents(conn, parents)
    vecs = fake_embed([c.embed_text for c in chunks])
    assert store.replace_chunks(conn, "token_aware", chunks, vecs) == len(chunks)
    row = conn.execute("SELECT population_tags, india_focused, quality_score IS NULL FROM chunks LIMIT 1").fetchone()
    assert row[0] == ["indian", "prediabetes"] and row[1] is True

    q = fake_embed(["Refer adults with prediabetes to lifestyle programs."])[0]
    exact = store.vector_search(conn, "token_aware", q, k=3, exact=True)
    kinds = [conn.execute("SELECT kind FROM chunks WHERE chunk_id = %s", (cid,)).fetchone()[0] for cid, _, _ in exact]
    assert exact[0][1] == "PMID:1" and "recommendation" in kinds and exact[0][2] > 0.5
    for kind in ("hnsw", "ivfflat", "scann"):
        name = store.create_vector_index(conn, "token_aware", kind, cfg["index"]["vector_index"][kind])
        assert conn.execute("SELECT count(*) FROM pg_indexes WHERE indexname = %s", (name,)).fetchone()[0] == 1
    filtered = store.vector_search(conn, "token_aware", q, k=5, where="kind = %s", params=("table",))
    assert len(filtered) == 1
    assert conn.execute("SELECT count(*) FROM chunks WHERE tsv @@ plainto_tsquery('english', 'lifestyle')").fetchone()[0] >= 1
    # Replacing a strategy's chunks doesn't touch another strategy.
    with pytest.raises(ValueError, match="passed as strategy"):
        store.replace_chunks(conn, "recursive", chunks[:2], vecs[:2])
    from dataclasses import replace
    store.replace_chunks(conn, "recursive", [replace(c, strategy="recursive") for c in chunks[:2]], vecs[:2])
    store.replace_chunks(conn, "token_aware", chunks[:1], vecs[:1])
    counts = dict(conn.execute("SELECT strategy, count(*) FROM chunks GROUP BY 1").fetchall())
    assert counts == {"token_aware": 1, "recursive": 2}


def test_bm25_build_and_search(tmp_path):
    chunks, _ = ch.build_doc_chunks(DOC, "token_aware", TOK, {"index": {"chunk": {
        "target_tokens": 40, "max_tokens": 50, "min_tokens": 8, "overlap_tokens": 5, "overlap_sentences": 1,
        "hierarchical": {"child_tokens": 15, "child_overlap": 3}, "semantic": {"breakpoint_percentile": 20}},
        "tables": {"summary": "caption", "max_markdown_chars": 4000}}}, fulltext=FULLTEXT)
    bm25.build(chunks, tmp_path / "bm", "english")
    hits = bm25.BM25Index(tmp_path / "bm").search("lifestyle programmes for prediabetes", k=3)
    assert hits and next(c for c in chunks if c.chunk_id == hits[0][0]).kind == "recommendation"


def test_build_end_to_end(conn, cfg, tmp_path, monkeypatch):
    docs = [DOC, {**DOC, "doc_id": "PMID:2", "title": "Sleep and weight", "abstract": para(3, tag="sleep"),
                  "fulltext_path": None, "status": "provisional"},
            {**DOC, "doc_id": "PMID:3", "status": "excluded"}]
    ft_path = tmp_path / "ft.json"
    ft_path.write_text(json.dumps(FULLTEXT))
    docs[0] = {**DOC, "fulltext_path": str(ft_path)}
    (tmp_path / "documents.jsonl").write_text("\n".join(json.dumps(d) for d in docs))
    (tmp_path / "scores.jsonl").write_text(json.dumps({"doc_id": "PMID:1", "quality_score_v2": 0.8, "superseded_by": None}))
    c2 = _db_cfg(cfg)
    c2 = {**c2, "paths": {**c2["paths"], "documents": str(tmp_path / "documents.jsonl")},
          "score_v2": {**c2["score_v2"], "output": str(tmp_path / "scores.jsonl")},
          "index": {**c2["index"], "bm25": {"dir": str(tmp_path / "bm25"), "stemmer": "english"},
                    "manifest": str(tmp_path / "manifest.json"), "report": str(tmp_path / "report.md"),
                    "chunk": {**c2["index"]["chunk"], "target_tokens": 40, "max_tokens": 50, "min_tokens": 8,
                              "hierarchical": {"child_tokens": 15, "child_overlap": 3}}}}
    monkeypatch.setattr("nrag.extract.runner.load_results", lambda cfg, slug: [
        {"doc_id": "PMID:1", "status": "ok", "extraction": EXTRACTION}])
    conn.execute("DROP TABLE IF EXISTS chunks, parents, documents CASCADE")
    emb = Embedder("fake", EmbeddingCache(tmp_path / "e.sqlite"), encode_fn=fake_embed)
    report = build.run(c2, ["token_aware", "hierarchical"], "hnsw", tok=TOK, embedder=emb, conn=conn)

    assert report["documents"] == 2 and report["with_extraction"] == 1 and report["with_fulltext"] == 1
    s = report["strategies"]
    assert s["hierarchical"]["by_kind"]["section"] > s["token_aware"]["by_kind"]["section"]
    assert s["hierarchical"]["embeddings"]["cached"] >= 2       # abstract chunks reused across strategies
    assert (tmp_path / "bm25" / "token_aware" / "chunk_ids.json").exists()
    assert "Index statistics" in (tmp_path / "report.md").read_text()
    n_docs = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    assert n_docs == 2                                          # excluded doc not indexed
    # Re-running after a document leaves the corpus removes its chunks.
    (tmp_path / "documents.jsonl").write_text(json.dumps(docs[0]))
    build.run(c2, ["token_aware"], "hnsw", tok=TOK, embedder=emb, conn=conn)
    assert conn.execute("SELECT count(*) FROM chunks WHERE doc_id = 'PMID:2'").fetchone()[0] == 0
