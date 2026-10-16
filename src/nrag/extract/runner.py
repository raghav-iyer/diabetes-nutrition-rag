"""Glue between the corpus, the extractor, the review CSV and the CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from nrag.config import env, resolve_path
from nrag.extract import pico, review
from nrag.llm import EndpointPool, client_from_config


def parse_think(value: str | None) -> Any:
    if value is None:
        return "default"
    v = value.strip().lower()
    return {"true": True, "false": False, "none": None}.get(v, v)


def active_model(cfg: dict[str, Any]) -> str:
    return env("LLM_MODEL") or cfg["extract"]["active_model"]


def endpoints(cfg: dict[str, Any]) -> list[str]:
    raw = env("LLM_ENDPOINTS")
    return [e.strip() for e in raw.split(",") if e.strip()] if raw else list(cfg["llm"]["endpoints"])


def load_documents(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    path = resolve_path(cfg["paths"]["documents"])
    if not path.exists():
        raise SystemExit(f"{path} not found; run `nrag ingest` first.")
    with path.open() as f:
        return [d for d in map(json.loads, f) if d.get("pmid")]


def select(cfg: dict[str, Any], docs: list[dict[str, Any]], sample: str, limit: int | None = None):
    s = cfg["extract"]["sample"]
    if sample == "all":
        chosen = sorted(docs, key=lambda d: d["doc_id"])
    elif sample in ("screen", "review"):
        chosen = review.stratified_sample(docs, s[sample], s["seed"])
    else:
        raise SystemExit(f"unknown sample {sample!r} (screen | review | all)")
    return chosen[:limit] if limit else chosen


def output_path(cfg: dict[str, Any], slug: str) -> Path:
    return resolve_path(cfg["extract"]["output_dir"]) / f"{slug}.jsonl"


def save_results(cfg: dict[str, Any], slug: str, results: list[dict[str, Any]]) -> Path:
    """Merge into the per-system results file (keyed by doc_id)."""
    path = output_path(cfg, slug)
    merged = {r["doc_id"]: r for r in load_results(cfg, slug)}
    merged.update({r["doc_id"]: {k: v for k, v in r.items() if k != "cached"} for r in results})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r in sorted(merged.values(), key=lambda r: r["doc_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    failures = [r for r in results if r["status"] != "ok"]
    if failures:
        with (path.parent / "failures.jsonl").open("a") as f:
            for r in failures:
                f.write(json.dumps({k: r[k] for k in ("doc_id", "model", "think", "errors", "attempts", "created")}) + "\n")
    return path


def load_results(cfg: dict[str, Any], slug: str) -> list[dict[str, Any]]:
    path = output_path(cfg, slug)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def speed_stats(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    lat = sorted(r["latency_s"] for r in results if r["status"] == "ok")
    if not lat:
        return None
    return {"n": len(lat), "median": lat[len(lat) // 2], "mean": round(sum(lat) / len(lat), 2), "max": lat[-1]}


def run(cfg: dict[str, Any], model: str, think: Any, sample: str, limit: int | None, use_cache: bool):
    ec = pico.ExtractConfig.from_config(cfg, model, think)
    client = client_from_config(cfg["llm"])
    eps = endpoints(cfg)
    for e in eps:
        available = client.list_models(e)
        if model not in available and f"{model}:latest" not in available:
            raise SystemExit(f"model {model!r} not available at {e}; pull it first (`ollama pull {model}`).")
    pool = EndpointPool(eps, cfg["llm"]["concurrency_per_endpoint"])
    docs = select(cfg, load_documents(cfg), sample, limit)
    slug = pico.model_slug(model, ec.think)
    print(f"Extracting {len(docs)} documents with {model} (think={ec.think}) on {len(eps)} endpoint(s)")
    results = pico.run_extraction(client, pool, ec, docs, use_cache=use_cache)
    path = save_results(cfg, slug, results)
    summary = pico.summarize_run(results)
    summary.update(system=slug, output=str(path))
    return summary


def compare(cfg: dict[str, Any], systems: list[str], sample: str, force: bool = False) -> Path:
    ex = cfg["extract"]
    csv_path, key_path = resolve_path(ex["review_csv"]), resolve_path(ex["review_key"])
    if csv_path.exists() and not force and any(r["verdict"].strip() for r in review.read_review(csv_path)):
        raise SystemExit(f"{csv_path} already has verdicts; pass --force to overwrite (your labels would be lost).")
    docs = select(cfg, load_documents(cfg), sample)
    extractions = {}
    for slug in systems:
        by_doc = {r["doc_id"]: r["extraction"] for r in load_results(cfg, slug) if r["status"] == "ok"}
        missing = [d["doc_id"] for d in docs if d["doc_id"] not in by_doc]
        if missing:
            raise SystemExit(f"{slug}: no successful extraction for {len(missing)} sampled docs "
                             f"(e.g. {missing[:3]}); run `nrag extract` for it first.")
        extractions[slug] = by_doc
    review.write_review_csv(csv_path, key_path, docs, extractions, seed=ex["sample"]["seed"])
    return csv_path


def accuracy(cfg: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    ex = cfg["extract"]
    csv_path, key_path = resolve_path(ex["review_csv"]), resolve_path(ex["review_key"])
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found; run `nrag extract-compare` first.")
    rows = review.read_review(csv_path)
    key = json.loads(key_path.read_text())
    docs = {d["doc_id"]: d for d in load_documents(cfg)}
    baseline = {doc_id: review.rules_baseline(docs[doc_id], ex["rules_baseline_mesh_tags"])
                for doc_id in key["assignment"] if doc_id in docs}
    speeds = {slug: speed_stats(load_results(cfg, slug)) for slug in key["systems"]}
    result = review.score_review(rows, key, baseline, speeds, ex["decision"], iters=ex["bootstrap_iters"],
                                 seed=ex["sample"]["seed"])
    report = resolve_path(ex["report"])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(review.render_report(result))
    report.with_suffix(".json").write_text(json.dumps(result, indent=1))
    return result, report
