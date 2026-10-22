"""BM25 lexical index (bm25s) over the same chunks as the vector index, one per strategy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _stemmer(lang: str | None):
    if not lang:
        return None
    import Stemmer
    return Stemmer.Stemmer(lang)


def build(chunks: list[Any], out_dir: Path, stemmer_lang: str | None = "english") -> int:
    import bm25s

    stem = _stemmer(stemmer_lang)
    texts = [c.text for c in chunks]
    tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stem, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(str(out_dir))
    (out_dir / "chunk_ids.json").write_text(json.dumps([[c.chunk_id, c.doc_id] for c in chunks]))
    (out_dir / "meta.json").write_text(json.dumps({"stemmer": stemmer_lang, "n": len(chunks)}))
    return len(chunks)


class BM25Index:
    def __init__(self, out_dir: Path):
        import bm25s

        self.retriever = bm25s.BM25.load(str(out_dir))
        self.ids = [tuple(x) for x in json.loads((out_dir / "chunk_ids.json").read_text())]
        self.stemmer = _stemmer(json.loads((out_dir / "meta.json").read_text())["stemmer"])

    def search(self, query: str, k: int = 10) -> list[tuple[str, str, float]]:
        import bm25s

        k = min(k, len(self.ids))
        q = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer, show_progress=False)
        docs, scores = self.retriever.retrieve(q, k=k, show_progress=False)
        return [(*self.ids[int(i)], float(s)) for i, s in zip(docs[0], scores[0]) if s > 0]
