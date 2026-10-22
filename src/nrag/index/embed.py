"""Embedding with an on-disk cache (SQLite), keyed by model and exact text.

Re-indexing after a config change only embeds chunks whose text changed.
Vectors are L2-normalised, so cosine similarity is a dot product.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np


class EmbeddingCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, dim INTEGER, vec BLOB)")
        self.conn.commit()

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        out = {}
        with self._lock:
            for i in range(0, len(keys), 900):
                chunk = keys[i:i + 900]
                q = f"SELECT key, dim, vec FROM emb WHERE key IN ({','.join('?' * len(chunk))})"
                for k, dim, blob in self.conn.execute(q, chunk):
                    out[k] = np.frombuffer(blob, dtype=np.float32).reshape(dim)
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        with self._lock:
            self.conn.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?, ?)",
                                  [(k, v.shape[0], v.astype(np.float32).tobytes()) for k, v in items.items()])
            self.conn.commit()


class Embedder:
    """Wraps a sentence-transformers model (or any `encode_fn`) with the cache."""

    def __init__(self, model_name: str, cache: EmbeddingCache, device: str = "cpu", batch_size: int = 16,
                 max_tokens: int | None = None, encode_fn: Callable[[list[str]], np.ndarray] | None = None):
        self.model_name = model_name
        self.cache = cache
        self.batch_size = batch_size
        self.stats = {"cached": 0, "computed": 0}
        self._encode_fn = encode_fn
        self._model = None
        self._device = device
        self._max_tokens = max_tokens

    def _model_encode(self, texts: list[str]) -> np.ndarray:
        if self._encode_fn is not None:
            return np.asarray(self._encode_fn(texts), dtype=np.float32)
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self._device)
            if self._max_tokens:
                self._model.max_seq_length = self._max_tokens
        return self._model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=len(texts) > 500).astype(np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        keys = [EmbeddingCache.key(self.model_name, t) for t in texts]
        found = self.cache.get_many(list(dict.fromkeys(keys)))
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in found))
        if missing:
            vecs = self._model_encode(missing)
            new = {EmbeddingCache.key(self.model_name, t): v for t, v in zip(missing, vecs)}
            self.cache.put_many(new)
            found.update(new)
        self.stats["computed"] += len(missing)
        self.stats["cached"] += len(texts) - len(missing)
        return np.stack([found[k] for k in keys])


def embedder_from_config(cfg: dict[str, Any], resolve) -> Embedder:
    ic = cfg["index"]
    return Embedder(ic["embedding_model"], EmbeddingCache(resolve(ic["embed_cache"])), device=ic["device"],
                    batch_size=ic["batch_size"], max_tokens=ic["max_embed_tokens"])
