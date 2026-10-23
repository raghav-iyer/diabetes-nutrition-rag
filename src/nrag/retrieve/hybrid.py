"""Search backends over one chunking strategy: BM25 (bm25s), dense (pgvector HNSW), metadata lookup."""

from __future__ import annotations

from typing import Any

from nrag.index import store
from nrag.index.bm25 import BM25Index

META_COLS = ("chunk_id", "doc_id", "parent_id", "kind", "section", "text", "year", "design", "source_type",
             "topics", "population_tags", "quality_score", "superseded", "india_focused")


class HybridSearcher:
    def __init__(self, conn, bm25_index: BM25Index | None, embedder, strategy: str, ef_search: int = 200):
        self.conn, self.bm25_index, self.embedder, self.strategy = conn, bm25_index, embedder, strategy
        self.ef_search = ef_search

    def bm25(self, query: str, k: int) -> list[str]:
        return [cid for cid, _, _ in self.bm25_index.search(query, k)] if self.bm25_index else []

    def dense(self, query: str, k: int) -> list[str]:
        vec = self.embedder.encode([query])[0]
        # HNSW returns at most ef_search rows; it must be >= k.
        self.conn.execute(f"SET hnsw.ef_search = {max(int(self.ef_search), int(k))}")
        return [cid for cid, _, _ in store.vector_search(self.conn, self.strategy, vec, k=k)]

    def meta(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}
        rows = self.conn.execute(
            f"SELECT {', '.join('c.' + c for c in META_COLS)}, d.title, d.url FROM chunks c "
            f"JOIN documents d USING (doc_id) WHERE c.strategy = %s AND c.chunk_id = ANY(%s)",
            (self.strategy, chunk_ids)).fetchall()
        return {r[0]: dict(zip(META_COLS + ("title", "url"), r)) for r in rows}

    def parents(self, parent_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not parent_ids:
            return {}
        rows = self.conn.execute("SELECT parent_id, section, title, text FROM parents WHERE parent_id = ANY(%s)",
                                 (parent_ids,)).fetchall()
        return {r[0]: {"section": r[1], "title": r[2], "text": r[3]} for r in rows}
