"""LLM extraction of PICO and evidence fields, with caching and validation.

Each document's result is cached under its input hash (model, think setting,
prompt version, schema, text), so re-runs only call the model for new or
changed documents. A response that fails JSON or schema validation is retried
with the validation error fed back (up to `extract.max_attempts`); documents
that still fail are logged, never silently dropped.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from nrag.config import resolve_path
from nrag.extract.schemas import Extraction, extraction_json_schema
from nrag.llm import EndpointPool, LLMClient, LLMError

PROMPT_DIR = Path(__file__).parent / "prompts"


def load_prompt(version: str) -> tuple[str, str]:
    text = (PROMPT_DIR / f"{version}.md").read_text()
    system = text.split("<!-- SYSTEM -->", 1)[1].split("<!-- USER -->", 1)[0].strip()
    user = text.split("<!-- USER -->", 1)[1].strip()
    return system, user


def model_slug(model: str, think: Any = None) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model).strip("-")
    return slug + (f"__think-{think}" if think not in (None, False) else "")


@dataclass
class ExtractConfig:
    model: str
    think: Any
    prompt_version: str
    max_attempts: int
    dropout_threshold_pct: float
    max_tokens: int | None
    cache_dir: Path
    output_mode: str = "schema"

    @classmethod
    def from_config(cls, cfg: dict[str, Any], model: str, think: Any = "default", cache_dir: Path | None = None):
        ex = cfg["extract"]
        if think == "default":
            think = ex["think"].get(model)
        modes = ex.get("output_mode", {})
        return cls(model=model, think=think, prompt_version=ex["prompt_version"], max_attempts=ex["max_attempts"],
                   dropout_threshold_pct=ex["dropout_threshold_pct"], max_tokens=ex.get("max_tokens"),
                   cache_dir=cache_dir or resolve_path(cfg["paths"]["cache_dir"]) / "llm",
                   output_mode=modes.get(model, modes.get("default", "schema")))


def doc_text(doc: dict[str, Any]) -> tuple[str, str]:
    return doc.get("title", ""), doc.get("abstract", "")


def input_hash(ec: ExtractConfig, title: str, abstract: str) -> str:
    system, user = load_prompt(ec.prompt_version)
    blob = json.dumps([ec.model, ec.think, ec.prompt_version, system, user, ec.dropout_threshold_pct,
                       extraction_json_schema(), title, abstract] + ([ec.output_mode] if ec.output_mode != "schema" else []),
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def build_messages(ec: ExtractConfig, title: str, abstract: str) -> list[dict[str, str]]:
    system, user = load_prompt(ec.prompt_version)
    system = system.replace("{dropout_threshold_pct}", f"{ec.dropout_threshold_pct:g}")
    if ec.output_mode == "prompt":
        system += ("\n\nJSON schema (follow it exactly; use only the allowed enum values):\n"
                   + json.dumps(extraction_json_schema(), separators=(",", ":"))
                   + "\n\nReturn one JSON object only: no markdown, no commentary.")
    return [{"role": "system", "content": system},
            {"role": "user", "content": user.replace("{title}", title).replace("{abstract}", abstract)}]


def json_payload(content: str) -> str:
    """Strip markdown fences or stray text around the JSON object (prompt mode)."""
    text = content.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


# ---------------------------------------------------------------------------
# Quote verification (cheap hallucination check)
# ---------------------------------------------------------------------------
DASHES = re.compile(r"[\u2010-\u2015\u2212-]")
ELLIPSIS = re.compile(r"\.\.\.|\u2026")


def _norm(s: str) -> str:
    """Lowercase; dashes, quotes and other punctuation become spaces (models often swap
    ASCII hyphens for Unicode ones or straight quotes for curly ones)."""
    s = DASHES.sub(" ", s.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w%.,=<>]+", " ", s)).strip(" .,")


def _segment_supported(q: str, t: str, min_ratio: float) -> bool:
    if q in t:
        return True
    m = SequenceMatcher(None, q, t, autojunk=False).find_longest_match(0, len(q), 0, len(t))
    return m.size / len(q) >= min_ratio


def quote_supported(quote: str | None, text: str, min_ratio: float = 0.9) -> bool | None:
    """True if the quote appears (near-)verbatim in the text; None if there is no quote.
    A quote elided with "..." is supported if every segment is."""
    if not quote:
        return None
    t = _norm(text)
    segments = [_norm(seg) for seg in ELLIPSIS.split(quote)]
    segments = [seg for seg in segments if len(seg) >= 3]
    if not segments:
        return None
    return all(_segment_supported(seg, t, min_ratio) for seg in segments)


def quote_checks(extraction: dict[str, Any], title: str, abstract: str) -> dict[str, bool | None]:
    text = f"{title}\n{abstract}"
    return {k: quote_supported(v, text) for k, v in (extraction.get("quotes") or {}).items()}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def _cache_path(ec: ExtractConfig, h: str) -> Path:
    return ec.cache_dir / model_slug(ec.model, ec.think) / h[:2] / f"{h}.json"


def extract_one(client: LLMClient, endpoint: str, ec: ExtractConfig, doc: dict[str, Any],
                use_cache: bool = True) -> dict[str, Any]:
    title, abstract = doc_text(doc)
    h = input_hash(ec, title, abstract)
    path = _cache_path(ec, h)
    if use_cache and path.exists():
        cached = json.loads(path.read_text())
        # Quote checks are cheap and deterministic: recompute so checker fixes apply to cached results.
        cached["quote_checks"] = quote_checks(cached["extraction"], title, abstract)
        return {**cached, "cached": True}

    messages = build_messages(ec, title, abstract)
    schema = extraction_json_schema() if ec.output_mode == "schema" else None
    errors: list[str] = []
    attempts, latency, prompt_tokens, completion_tokens = 0, 0.0, 0, 0
    extraction = None
    for attempt in range(ec.max_attempts):
        attempts += 1
        try:
            resp = client.chat(endpoint, ec.model, messages, schema=schema, think=ec.think, max_tokens=ec.max_tokens)
        except LLMError as e:
            errors.append(f"llm_error: {e}")
            break
        latency += resp.latency_s
        prompt_tokens += resp.prompt_tokens or 0
        completion_tokens += resp.completion_tokens or 0
        try:
            extraction = Extraction.model_validate_json(json_payload(resp.content)).model_dump()
            break
        except ValidationError as e:
            err = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:8])
            errors.append(f"validation: {err}")
            messages = messages + [
                {"role": "assistant", "content": resp.content[:4000]},
                {"role": "user", "content": f"That output failed validation: {err}. Return corrected JSON only."},
            ]

    result = {
        "doc_id": doc["doc_id"], "model": ec.model, "think": ec.think, "prompt_version": ec.prompt_version,
        "output_mode": ec.output_mode,
        "input_hash": h, "status": "ok" if extraction else "failed", "extraction": extraction,
        "quote_checks": quote_checks(extraction, title, abstract) if extraction else {},
        "attempts": attempts, "errors": errors, "latency_s": round(latency, 2),
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "endpoint": endpoint,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # Only successful results are cached, so failures are retried on the next run.
    if extraction:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False))
    return {**result, "cached": False}


def run_extraction(client: LLMClient, pool: EndpointPool, ec: ExtractConfig, docs: Iterable[dict[str, Any]],
                   use_cache: bool = True, progress: bool = True) -> list[dict[str, Any]]:
    docs = list(docs)
    results: list[dict[str, Any]] = []

    def job(doc):
        with pool.acquire() as endpoint:
            return extract_one(client, endpoint, ec, doc, use_cache=use_cache)

    with ThreadPoolExecutor(max_workers=pool.size) as ex:
        futures = {ex.submit(job, d): d["doc_id"] for d in docs}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            if progress:
                tag = "cache" if r["cached"] else f"{r['latency_s']:.1f}s"
                print(f"  [{i}/{len(docs)}] {r['doc_id']} {r['status']} ({tag})", flush=True)
    order = {d["doc_id"]: i for i, d in enumerate(docs)}
    return sorted(results, key=lambda r: order[r["doc_id"]])


def summarize_run(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    ok = [r for r in results if r["status"] == "ok"]
    fresh = [r for r in results if not r.get("cached")]
    first_try = sum(1 for r in ok if r["attempts"] == 1)
    qc = [v for r in ok for v in r["quote_checks"].values() if v is not None]
    lat = sorted(r["latency_s"] for r in fresh if r["status"] == "ok")
    return {
        "documents": n,
        "ok": len(ok),
        "failed": n - len(ok),
        "schema_failure_rate": round((n - len(ok)) / n, 4) if n else None,
        "first_attempt_valid_rate": round(first_try / n, 4) if n else None,
        "quote_support_rate": round(sum(qc) / len(qc), 4) if qc else None,
        "fresh_calls": len(fresh),
        "latency_s": ({"median": lat[len(lat) // 2], "p90": lat[int(len(lat) * 0.9) - 1] if len(lat) >= 10 else lat[-1],
                       "mean": round(sum(lat) / len(lat), 2)} if lat else None),
        "completion_tokens_mean": round(sum(r["completion_tokens"] for r in fresh) / len(fresh), 1) if fresh else None,
    }
