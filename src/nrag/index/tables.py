"""LLM one-paragraph table summaries, used as the table chunk's embedding text.

Tables embed poorly as raw markdown; a short summary of what the table reports
(arms, populations, outcomes, key numbers) gives the retriever something to match.
Summaries are cached per (model, prompt, table text) and saved per model in
data/interim/table_summaries/<model>.jsonl, keyed by the table's parent_id.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from nrag.config import resolve_path
from nrag.extract.pico import model_slug
from nrag.index.chunk import table_markdown
from nrag.llm import EndpointPool, LLMClient, LLMError

PROMPT = ("You summarise tables from nutrition research papers for a search index. In one paragraph of at most "
          "80 words, say what the table reports: the groups or arms, the population, the outcomes, and the key "
          "numbers or differences. Use only the table and its caption; do not add interpretation or outside "
          "knowledge. Return the paragraph only.")


def table_inputs(docs: list[dict[str, Any]], fulltext: dict[str, dict], max_chars: int) -> list[dict[str, str]]:
    """One item per table, with the same parent_id the chunker uses."""
    items = []
    for d in docs:
        ft = fulltext.get(d["doc_id"])
        for ti, t in enumerate((ft or {}).get("tables") or []):
            md = table_markdown(t, max_chars)
            if not md:
                continue
            label = " ".join(x for x in (t.get("label"), t.get("caption")) if x)
            items.append({"parent_id": f"{d['doc_id']}#t{ti}", "title": d.get("title", ""),
                          "text": f"Paper: {d.get('title', '')}\n{label}\n{md}\n{t.get('footer') or ''}".strip()})
    return items


def _hash(model: str, text: str) -> str:
    return hashlib.sha256(json.dumps([model, PROMPT, text]).encode()).hexdigest()


def output_path(cfg: dict[str, Any], model: str) -> Path:
    return resolve_path(cfg["index"]["tables"]["summaries_dir"]) / f"{model_slug(model)}.jsonl"


def load_summaries(cfg: dict[str, Any], model: str) -> dict[str, str]:
    p = output_path(cfg, model)
    if not p.exists():
        return {}
    with p.open() as f:
        return {r["parent_id"]: r["summary"] for r in map(json.loads, f) if r.get("summary")}


def summarize(client: LLMClient, pool: EndpointPool, model: str, items: list[dict[str, str]], out: Path,
              max_tokens: int = 300, progress_every: int = 100) -> dict[str, int]:
    done: dict[str, dict] = {}
    if out.exists():
        with out.open() as f:
            for r in map(json.loads, f):
                done[r["input_hash"]] = r
    todo = [it for it in items if _hash(model, it["text"]) not in done]
    stats = {"tables": len(items), "cached": len(items) - len(todo), "ok": 0, "failed": 0}
    out.parent.mkdir(parents=True, exist_ok=True)

    def job(it):
        with pool.acquire() as ep:
            resp = client.chat(ep, model, [{"role": "system", "content": PROMPT},
                                           {"role": "user", "content": it["text"]}], max_tokens=max_tokens)
            return it, resp

    with ThreadPoolExecutor(max_workers=pool.size) as ex, out.open("a") as f:
        futures = [ex.submit(job, it) for it in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                it, resp = fut.result()
            except LLMError:
                stats["failed"] += 1
                continue
            summary = " ".join(resp.content.split())
            if not summary:
                stats["failed"] += 1
                continue
            f.write(json.dumps({"parent_id": it["parent_id"], "input_hash": _hash(model, it["text"]),
                                "summary": summary, "latency_s": round(resp.latency_s, 2)}) + "\n")
            f.flush()
            stats["ok"] += 1
            if i % progress_every == 0:
                print(f"  {i}/{len(todo)} tables summarised", flush=True)
    return stats
