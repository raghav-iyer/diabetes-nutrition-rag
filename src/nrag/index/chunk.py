"""Chunking: abstract chunks, full-text section chunks (5 strategies), table and
guideline-recommendation chunks.

Every chunk carries its document's metadata (for filtering) and a `parent_id`:
the section it came from. Retrieval can match a small chunk and return the
whole parent section (parent-document retrieval).

Full-text strategies (compared as an ablation in Phase 7):
  token_aware     pack whole sentences up to a token budget, overlap N sentences
  recursive       split on paragraph -> line -> sentence -> word boundaries until
                  pieces fit, then merge up to the target with token overlap
  sliding_window  fixed token windows with a fixed overlap, ignoring sentences
  semantic        split where adjacent-sentence embedding similarity drops
                  (lowest percentile), within the token budget
  hierarchical    small child chunks are indexed; the parent section is returned
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

STRATEGIES = ("token_aware", "recursive", "sliding_window", "semantic", "hierarchical")

_ABBREV = r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bet al\.)(?<!\bvs\.)(?<!\bapprox\.)(?<!\bFig\.)(?<!\bNo\.)"
SENTENCE_RE = re.compile(_ABBREV + r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class WhitespaceTokenizer:
    """Tokenizer for tests: one token per whitespace-separated word."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.inv: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        out = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab)
                self.inv[self.vocab[w]] = w
            out.append(self.vocab[w])
        return out

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.inv[i] for i in ids)


class HFTokenizer:
    """The embedding model's own tokenizer, so token budgets match what gets embedded."""

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True).strip()


def n_tokens(tok: Tokenizer, text: str) -> int:
    return len(tok.encode(text))


def sentences(text: str) -> list[str]:
    out = []
    for para in re.split(r"\n\s*\n|\n", text):
        out += [s.strip() for s in SENTENCE_RE.split(para.strip()) if s.strip()]
    return out


# ---------------------------------------------------------------------------
# Strategies (each splits one section's text into chunk texts)
# ---------------------------------------------------------------------------
def _merge_small(pieces: list[str], tok: Tokenizer, min_tokens: int) -> list[str]:
    """Fold a too-small trailing piece into the previous one."""
    if len(pieces) >= 2 and n_tokens(tok, pieces[-1]) < min_tokens:
        pieces[-2] = pieces[-2] + " " + pieces[-1]
        pieces.pop()
    return pieces


def _hard_split(text: str, tok: Tokenizer, max_tokens: int) -> list[str]:
    ids = tok.encode(text)
    return [tok.decode(ids[i:i + max_tokens]) for i in range(0, len(ids), max_tokens)] or [text]


def split_token_aware(text: str, tok: Tokenizer, c: dict[str, Any]) -> list[str]:
    sents = sentences(text)
    sizes = [n_tokens(tok, s) for s in sents]
    chunks, cur, cur_n = [], [], 0
    i = 0
    while i < len(sents):
        s, n = sents[i], sizes[i]
        if n > c["max_tokens"]:                       # one huge "sentence" (e.g. a list): hard split
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_n = [], 0
            chunks += _hard_split(s, tok, c["max_tokens"])
            i += 1
            continue
        if cur and cur_n + n > c["target_tokens"]:
            chunks.append(" ".join(cur))
            keep = cur[-c["overlap_sentences"]:] if c["overlap_sentences"] else []
            # Overlap only if it still leaves room for the next sentence.
            if sum(n_tokens(tok, k) for k in keep) + n > c["max_tokens"]:
                keep = []
            cur, cur_n = list(keep), sum(n_tokens(tok, k) for k in keep)
        cur.append(s)
        cur_n += n
        i += 1
    if cur:
        chunks.append(" ".join(cur))
    return _merge_small(chunks, tok, c["min_tokens"])


SEPARATORS = ["\n\n", "\n", SENTENCE_RE, " "]


def split_recursive(text: str, tok: Tokenizer, c: dict[str, Any]) -> list[str]:
    def pieces(t: str, level: int) -> list[str]:
        if n_tokens(tok, t) <= c["target_tokens"] or level >= len(SEPARATORS):
            return [t] if n_tokens(tok, t) <= c["max_tokens"] else _hard_split(t, tok, c["max_tokens"])
        sep = SEPARATORS[level]
        parts = sep.split(t) if isinstance(sep, re.Pattern) else t.split(sep)
        out = []
        for p in parts:
            if p.strip():
                out += pieces(p.strip(), level + 1)
        return out

    atoms = pieces(text.strip(), 0)
    chunks, cur = [], ""
    for a in atoms:
        cand = f"{cur} {a}".strip()
        if cur and n_tokens(tok, cand) > c["target_tokens"]:
            chunks.append(cur)
            tail = tok.decode(tok.encode(cur)[-c["overlap_tokens"]:]) if c["overlap_tokens"] else ""
            cur = f"{tail} {a}".strip() if n_tokens(tok, f"{tail} {a}") <= c["max_tokens"] else a
        else:
            cur = cand
    if cur:
        chunks.append(cur)
    return _merge_small(chunks, tok, c["min_tokens"])


def split_sliding_window(text: str, tok: Tokenizer, c: dict[str, Any]) -> list[str]:
    ids = tok.encode(text)
    size, step = c["target_tokens"], c["target_tokens"] - c["overlap_tokens"]
    if len(ids) <= c["max_tokens"]:
        return [text.strip()] if text.strip() else []
    chunks = []
    for start in range(0, len(ids), step):
        window = ids[start:start + size]
        if start and len(window) < c["min_tokens"]:
            break                                       # covered by the previous window's overlap
        chunks.append(tok.decode(window))
        if start + size >= len(ids):
            break
    return chunks


def split_semantic(text: str, tok: Tokenizer, c: dict[str, Any],
                   embed: Callable[[list[str]], np.ndarray] | None) -> list[str]:
    if embed is None:
        raise ValueError("semantic chunking needs an embedding function")
    sents = sentences(text)
    if len(sents) <= 2:
        return split_token_aware(text, tok, c)
    vecs = embed(sents)
    sims = np.sum(vecs[:-1] * vecs[1:], axis=1)         # normalised vectors: dot = cosine
    cut = np.percentile(sims, c["semantic"]["breakpoint_percentile"])
    top = float(np.max(sims))
    groups, cur, cur_n = [], [sents[0]], n_tokens(tok, sents[0])
    for s, sim in zip(sents[1:], sims):
        n = n_tokens(tok, s)
        # Never split between sentences as similar as the most similar pair (ties at the
        # percentile would otherwise split after every sentence).
        topic_shift = sim <= cut and sim < top and cur_n >= c["min_tokens"]
        if topic_shift or cur_n + n > c["max_tokens"]:
            groups.append(" ".join(cur))
            cur, cur_n = [], 0
        cur.append(s)
        cur_n += n
    groups.append(" ".join(cur))
    out = []
    for g in groups:                                   # a single over-long sentence group
        out += [g] if n_tokens(tok, g) <= c["max_tokens"] else split_token_aware(g, tok, c)
    return _merge_small(out, tok, c["min_tokens"])


def split_hierarchical(text: str, tok: Tokenizer, c: dict[str, Any]) -> list[str]:
    h = c["hierarchical"]
    child = {**c, "target_tokens": h["child_tokens"], "max_tokens": h["child_tokens"] + h["child_overlap"],
             "overlap_tokens": h["child_overlap"], "min_tokens": min(c["min_tokens"], h["child_tokens"] // 3),
             "overlap_sentences": 0}
    return split_token_aware(text, tok, child)


def split_text(strategy: str, text: str, tok: Tokenizer, c: dict[str, Any], embed=None) -> list[str]:
    if strategy == "token_aware":
        return split_token_aware(text, tok, c)
    if strategy == "recursive":
        return split_recursive(text, tok, c)
    if strategy == "sliding_window":
        return split_sliding_window(text, tok, c)
    if strategy == "semantic":
        return split_semantic(text, tok, c, embed)
    if strategy == "hierarchical":
        return split_hierarchical(text, tok, c)
    raise ValueError(f"unknown chunking strategy {strategy!r}")


# ---------------------------------------------------------------------------
# Chunk objects
# ---------------------------------------------------------------------------
@dataclass
class Parent:
    parent_id: str
    doc_id: str
    section: str
    title: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    strategy: str
    kind: str                # abstract | section | table | recommendation
    section: str
    parent_id: str
    position: int
    text: str                # shown to the LLM and indexed by BM25
    embed_text: str          # what gets embedded (header + text, or a table summary)
    n_tokens: int
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def doc_meta(doc: dict[str, Any], extraction: dict[str, Any] | None, score: dict[str, Any] | None) -> dict[str, Any]:
    """Filterable metadata copied onto every chunk of a document."""
    ex = extraction or {}
    outcomes = sorted({o["type"] for o in ex.get("outcomes") or []})
    tags = sorted(set((ex.get("population") or {}).get("tags") or []) | set(doc.get("population_tags") or []))
    return {
        "title": doc.get("title", ""),
        "year": doc.get("year"),
        "design": doc.get("design"),
        "source_type": doc.get("source_type"),
        "topics": doc.get("topics") or [],
        "population_tags": tags,
        "outcome_types": outcomes,
        "status": doc.get("status"),
        "quality_score": (score or {}).get("quality_score_v2", doc.get("quality_score")),
        "superseded": bool((score or {}).get("superseded_by") or doc.get("superseded_by")),
        "india_focused": bool({"indian", "south_asian"} & set(tags)),
        "effect_direction": ex.get("effect_direction"),
        "key_finding": ex.get("key_finding"),
        "url": doc.get("pubmed_url") or doc.get("url"),
    }


def _cid(*parts: Any) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def header(meta: dict[str, Any]) -> str:
    bits = [f"Design: {meta['design']}" if meta.get("design") else "",
            f"Year: {meta['year']}" if meta.get("year") else "",
            f"Population: {', '.join(meta['population_tags'])}" if meta.get("population_tags") else "",
            f"Key finding: {meta['key_finding']}" if meta.get("key_finding") else ""]
    return " | ".join(b for b in bits if b)


def table_markdown(table: dict[str, Any], max_chars: int) -> str:
    rows = table.get("rows") or []
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [list(r) + [""] * (width - len(r)) for r in rows]
    md = "\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
                   + ["| " + " | ".join(r) + " |" for r in rows[1:]])
    return md[:max_chars]


def build_doc_chunks(doc: dict[str, Any], strategy: str, tok: Tokenizer, cfg: dict[str, Any],
                     extraction: dict[str, Any] | None = None, score: dict[str, Any] | None = None,
                     fulltext: dict[str, Any] | None = None, embed=None,
                     table_summaries: dict[str, str] | None = None) -> tuple[list[Chunk], list[Parent]]:
    c = cfg["index"]["chunk"]
    tcfg = cfg["index"]["tables"]
    meta = doc_meta(doc, extraction, score)
    head = header(meta)
    doc_id = doc["doc_id"]
    chunks: list[Chunk] = []
    parents: list[Parent] = []

    def add(kind, section, parent_id, text, embed_text):
        chunks.append(Chunk(chunk_id=_cid(strategy, doc_id, kind, len(chunks)), doc_id=doc_id, strategy=strategy,
                            kind=kind, section=section, parent_id=parent_id, position=len(chunks), text=text,
                            embed_text=embed_text, n_tokens=n_tokens(tok, text), meta=meta))

    # 1. Abstract chunk (every document): header + title + abstract. Its own parent.
    if doc.get("abstract") or doc.get("title"):
        pid = f"{doc_id}#abstract"
        body = f"{doc.get('title', '')}\n{doc.get('abstract', '')}".strip()
        parents.append(Parent(pid, doc_id, "abstract", doc.get("title", ""), body))
        add("abstract", "abstract", pid, body, f"{head}\n{body}" if head else body)

    if fulltext:
        # 2. Full-text sections, split by the strategy; the section is the parent.
        for si, sec in enumerate(fulltext.get("sections") or []):
            text = (sec.get("text") or "").strip()
            if not text:
                continue
            pid = f"{doc_id}#s{si}"
            name = sec.get("name") or "other"
            parents.append(Parent(pid, doc_id, name, sec.get("title") or name, text))
            for piece in split_text(strategy, text, tok, c, embed):
                prefix = f"{meta['title']} | {sec.get('title') or name}"
                add("section", name, pid, piece, f"{prefix}\n{piece}")
        # 3. Tables: never split. Embed a summary (LLM, if available) or caption + table.
        for ti, t in enumerate(fulltext.get("tables") or []):
            md = table_markdown(t, tcfg["max_markdown_chars"])
            label = " ".join(x for x in (t.get("label"), t.get("caption")) if x)
            text = f"{label}\n{md}\n{t.get('footer') or ''}".strip()
            if not md:
                continue
            pid = f"{doc_id}#t{ti}"
            parents.append(Parent(pid, doc_id, "table", label or f"Table {ti + 1}", text))
            summary = (table_summaries or {}).get(pid)
            embed_text = f"{meta['title']} | {summary}" if summary else f"{meta['title']} | {label}\n{md}"
            add("table", "table", pid, text, embed_text)
        # 4. Guideline recommendations: one chunk each, with the grade.
        for ri, r in enumerate(fulltext.get("recommendations") or []):
            pid = f"{doc_id}#r{ri}"
            grade = f" (Grade: {r['grade']})" if r.get("grade") else ""
            text = f"{r['text']}{grade}"
            parents.append(Parent(pid, doc_id, "recommendation", r.get("section") or "recommendation", text))
            add("recommendation", r.get("section") or "recommendation", pid, text, f"{meta['title']} | {text}")
    return chunks, parents
