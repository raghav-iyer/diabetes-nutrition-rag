"""LlamaIndex baseline retriever over the same chunks and embeddings as the custom pipeline.

Standard LlamaIndex hybrid setup: PGVectorStore (its own table in the same AlloyDB database,
filled with our chunk texts and precomputed bge-m3 vectors) + BM25Retriever, fused by
QueryFusionRetriever in reciprocal-rank mode with no query generation. No metadata
filters, reranker, quality weighting or confidence check; that's what Phase 7 compares.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nrag.config import env, resolve_path


def _vector_store(cfg: dict[str, Any], dim: int):
    from llama_index.vector_stores.postgres import PGVectorStore

    db, lc = cfg["index"]["db"], cfg["llamaindex"]
    return PGVectorStore.from_params(host=db["host"], port=str(db["port"]), database=db["dbname"], user=db["user"],
                                     password=env(db["password_env"], required=True),
                                     table_name=f"{lc['table']}_{cfg['retrieval']['strategy']}", embed_dim=dim,
                                     hnsw_kwargs={"hnsw_m": cfg["index"]["vector_index"]["hnsw"]["m"],
                                                  "hnsw_ef_construction": cfg["index"]["vector_index"]["hnsw"]["ef_construction"],
                                                  "hnsw_ef_search": cfg["retrieval"]["dense"]["ef_search"],
                                                  "hnsw_dist_method": "vector_cosine_ops"})


def load_nodes(conn, strategy: str, with_embeddings: bool = True):
    from llama_index.core.schema import TextNode

    cols = "chunk_id, doc_id, kind, section, text" + (", embedding" if with_embeddings else "")
    nodes = []
    for row in conn.execute(f"SELECT {cols} FROM chunks WHERE strategy = %s ORDER BY chunk_id", (strategy,)):
        node = TextNode(id_=row[0], text=row[4], metadata={"doc_id": row[1], "kind": row[2], "section": row[3]},
                        excluded_embed_metadata_keys=["doc_id", "kind", "section"],
                        excluded_llm_metadata_keys=["doc_id", "kind", "section"])
        if with_embeddings:
            vec = row[5].to_list() if hasattr(row[5], "to_list") else row[5]   # pgvector.Vector or array
            node.embedding = [float(x) for x in vec]
        nodes.append(node)
    return nodes


def build(cfg: dict[str, Any], conn, batch: int = 2000, vector_store=None, bm25_dir: Path | None = None) -> dict[str, int]:
    """(Re)create the LlamaIndex vector table and BM25 retriever from the custom pipeline's chunks."""
    from llama_index.retrievers.bm25 import BM25Retriever

    strategy = cfg["retrieval"]["strategy"]
    nodes = load_nodes(conn, strategy)
    vs = vector_store or _vector_store(cfg, cfg["index"]["embedding_dim"])
    try:
        vs.clear()
    except Exception:
        pass                                   # table not created yet
    for i in range(0, len(nodes), batch):
        vs.add(nodes[i:i + batch])
    bm25_dir = bm25_dir or resolve_path(cfg["llamaindex"]["bm25_dir"]) / strategy
    bm = BM25Retriever.from_defaults(nodes=[n.model_copy(update={"embedding": None}) for n in nodes],
                                     similarity_top_k=cfg["llamaindex"]["candidate_k"], language="english")
    bm25_dir.mkdir(parents=True, exist_ok=True)
    bm.persist(str(bm25_dir))
    return {"nodes": len(nodes)}


class LlamaIndexBaseline:
    def __init__(self, cfg: dict[str, Any], embed_model=None, vector_store=None, bm25_dir: Path | None = None):
        from llama_index.core import VectorStoreIndex
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.retrievers.bm25 import BM25Retriever

        lc = cfg["llamaindex"]
        if embed_model is None:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            embed_model = HuggingFaceEmbedding(model_name=cfg["index"]["embedding_model"], device=cfg["index"]["device"],
                                               normalize=True)
        vs = vector_store or _vector_store(cfg, cfg["index"]["embedding_dim"])
        index = VectorStoreIndex.from_vector_store(vs, embed_model=embed_model)
        vector = index.as_retriever(similarity_top_k=lc["candidate_k"])
        bm25 = BM25Retriever.from_persist_dir(str(bm25_dir or resolve_path(lc["bm25_dir"]) / cfg["retrieval"]["strategy"]))
        bm25.similarity_top_k = lc["candidate_k"]
        from llama_index.core.llms import MockLLM
        # num_queries=1: no query generation, so no LLM is called; MockLLM avoids LlamaIndex's OpenAI default.
        self.retriever = QueryFusionRetriever([vector, bm25], llm=MockLLM(), similarity_top_k=lc["top_k"],
                                              num_queries=1, mode="reciprocal_rerank", use_async=False, verbose=False)

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        out = []
        for rank, nws in enumerate(self.retriever.retrieve(question), 1):
            out.append({"rank": rank, "chunk_id": nws.node.node_id, "doc_id": nws.node.metadata.get("doc_id"),
                        "score": float(nws.score or 0.0), "text": nws.node.get_content()})
        return out
