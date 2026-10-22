"""pgvector store on AlloyDB Omni (any PostgreSQL with pgvector works; ScaNN needs AlloyDB).

Tables:
  documents  one row per indexed document, with filter columns
  parents    sections / tables / recommendations returned by parent-document retrieval
  chunks     (strategy, chunk_id) -> text, embedding, denormalised filter columns, tsvector

All strategies live in one `chunks` table; vector indexes are partial per strategy
(`WHERE strategy = ...`), so each chunking strategy is searched independently.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable

import numpy as np

from nrag.config import env

VECTOR_INDEX_TYPES = ("hnsw", "ivfflat", "scann")


def connect(cfg: dict[str, Any]):
    import psycopg
    from pgvector.psycopg import register_vector

    db = cfg["index"]["db"]
    password = env(db["password_env"], required=True)
    conn = psycopg.connect(host=db["host"], port=db["port"], dbname=db["dbname"], user=db["user"],
                           password=password, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def create_schema(conn, dim: int) -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id text PRIMARY KEY, title text, year int, design text, source_type text,
            topics text[], population_tags text[], outcome_types text[], status text,
            quality_score real, superseded boolean, india_focused boolean, url text, meta jsonb);
        CREATE TABLE IF NOT EXISTS parents (
            parent_id text PRIMARY KEY, doc_id text REFERENCES documents(doc_id) ON DELETE CASCADE,
            section text, title text, text text);
        CREATE TABLE IF NOT EXISTS chunks (
            strategy text NOT NULL, chunk_id text NOT NULL,
            doc_id text REFERENCES documents(doc_id) ON DELETE CASCADE,
            parent_id text, kind text, section text, position int, n_tokens int,
            text text, embed_text text, embedding vector({dim}),
            year int, design text, source_type text, topics text[], population_tags text[],
            outcome_types text[], quality_score real, superseded boolean, india_focused boolean,
            tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED,
            PRIMARY KEY (strategy, chunk_id));
        CREATE INDEX IF NOT EXISTS chunks_doc ON chunks (doc_id);
        CREATE INDEX IF NOT EXISTS chunks_filters ON chunks (strategy, design, year);
        CREATE INDEX IF NOT EXISTS chunks_topics ON chunks USING gin (topics);
        CREATE INDEX IF NOT EXISTS chunks_pop ON chunks USING gin (population_tags);
        CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING gin (tsv);
    """)


def upsert_documents(conn, docs: Iterable[dict[str, Any]]) -> int:
    rows = [(d["doc_id"], d["meta"]["title"], d["meta"]["year"], d["meta"]["design"], d["meta"]["source_type"],
             d["meta"]["topics"], d["meta"]["population_tags"], d["meta"]["outcome_types"], d["meta"]["status"],
             d["meta"]["quality_score"], d["meta"]["superseded"], d["meta"]["india_focused"], d["meta"]["url"],
             json.dumps({k: d["meta"][k] for k in ("effect_direction", "key_finding")}))
            for d in docs]
    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO documents VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (doc_id) DO UPDATE SET title=EXCLUDED.title, year=EXCLUDED.year, design=EXCLUDED.design,
              source_type=EXCLUDED.source_type, topics=EXCLUDED.topics, population_tags=EXCLUDED.population_tags,
              outcome_types=EXCLUDED.outcome_types, status=EXCLUDED.status, quality_score=EXCLUDED.quality_score,
              superseded=EXCLUDED.superseded, india_focused=EXCLUDED.india_focused, url=EXCLUDED.url,
              meta=EXCLUDED.meta""", rows)
    return len(rows)


def upsert_parents(conn, parents: Iterable[Any]) -> int:
    rows = [(p.parent_id, p.doc_id, p.section, p.title, p.text) for p in parents]
    with conn.cursor() as cur:
        cur.executemany("""INSERT INTO parents VALUES (%s,%s,%s,%s,%s) ON CONFLICT (parent_id) DO UPDATE
                           SET section=EXCLUDED.section, title=EXCLUDED.title, text=EXCLUDED.text""", rows)
    return len(rows)


def replace_chunks(conn, strategy: str, chunks: list[Any], vectors: np.ndarray) -> int:
    """Replace all chunks of one strategy (a rebuild is cheap; embeddings are cached)."""
    wrong = {c.strategy for c in chunks} - {strategy}
    if wrong:
        raise ValueError(f"chunks built with {sorted(wrong)} passed as strategy {strategy!r}")
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    conn.execute("DELETE FROM chunks WHERE strategy = %s", (strategy,))
    rows = []
    for ch, v in zip(chunks, vectors):
        m = ch.meta
        rows.append((ch.strategy, ch.chunk_id, ch.doc_id, ch.parent_id, ch.kind, ch.section, ch.position,
                     ch.n_tokens, ch.text, ch.embed_text, v, m["year"], m["design"], m["source_type"], m["topics"],
                     m["population_tags"], m["outcome_types"], m["quality_score"], m["superseded"],
                     m["india_focused"]))
    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO chunks (strategy, chunk_id, doc_id, parent_id, kind, section, position, n_tokens, text,
                                embed_text, embedding, year, design, source_type, topics, population_tags,
                                outcome_types, quality_score, superseded, india_focused)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", rows)
    return len(rows)


def _safe(name: str) -> str:
    if not re.fullmatch(r"[a-z_]+", name):
        raise ValueError(f"unsafe identifier {name!r}")
    return name


def index_name(strategy: str, kind: str) -> str:
    return f"chunks_{_safe(kind)}_{_safe(strategy)}"


def create_vector_index(conn, strategy: str, kind: str, params: dict[str, Any]) -> str:
    """Partial ANN index over one strategy's chunks. Returns the index name."""
    name = index_name(strategy, kind)
    rows = conn.execute("SELECT count(*) FROM chunks WHERE strategy = %s", (strategy,)).fetchone()[0]
    conn.execute(f"DROP INDEX IF EXISTS {name}")
    where = f"WHERE strategy = '{_safe(strategy)}'"
    if kind == "hnsw":
        conn.execute(f"CREATE INDEX {name} ON chunks USING hnsw (embedding vector_cosine_ops) "
                     f"WITH (m = {int(params['m'])}, ef_construction = {int(params['ef_construction'])}) {where}")
    elif kind == "ivfflat":
        lists = max(1, int(params["lists_factor"] * math.sqrt(max(rows, 1))))
        conn.execute(f"CREATE INDEX {name} ON chunks USING ivfflat (embedding vector_cosine_ops) "
                     f"WITH (lists = {lists}) {where}")
    elif kind == "scann":
        conn.execute("CREATE EXTENSION IF NOT EXISTS alloydb_scann")
        leaves = max(1, int(params["leaves_factor"] * math.sqrt(max(rows, 1))))
        conn.execute(f"CREATE INDEX {name} ON chunks USING scann (embedding cosine) "
                     f"WITH (num_leaves = {leaves}) {where}")
    else:
        raise ValueError(f"unknown vector index type {kind!r}")
    return name


def vector_search(conn, strategy: str, query_vec: np.ndarray, k: int = 10, exact: bool = False,
                  where: str = "", params: tuple = ()) -> list[tuple[str, str, float]]:
    """Top-k chunks by cosine similarity: [(chunk_id, doc_id, similarity)]. `exact` disables index use."""
    if exact:
        conn.execute("SET enable_indexscan = off")
    try:
        sql = (f"SELECT chunk_id, doc_id, 1 - (embedding <=> %s) AS sim FROM chunks "
               f"WHERE strategy = %s {('AND ' + where) if where else ''} ORDER BY embedding <=> %s LIMIT %s")
        return [tuple(r) for r in conn.execute(sql, (query_vec, strategy, *params, query_vec, k)).fetchall()]
    finally:
        if exact:
            conn.execute("SET enable_indexscan = on")


def table_sizes(conn) -> dict[str, str]:
    q = """SELECT relname, pg_size_pretty(pg_relation_size(c.oid)) FROM pg_class c
           JOIN pg_namespace n ON n.oid = c.relnamespace
           WHERE n.nspname = 'public' AND relname LIKE 'chunks%' ORDER BY relname"""
    return {r[0]: r[1] for r in conn.execute(q).fetchall()}
