"""Phase 1 ingestion pipeline: PubMed -> screening -> full text -> guidelines -> corpus.

The corpus is rebuilt from scratch on every run from:
  - the store of every PubMed record ever fetched (refetched, cached, at most
    every `cache.efetch_ttl_days`, so retractions/MeSH/EoC changes are seen),
  - the current config, the Retraction Watch data and the NLM Catalog.
So documents.jsonl and excluded.jsonl are deterministic for a given input
state, and documents that drop out are appended to the removals log.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from nrag.config import resolve_path
from nrag.ingest import credibility, guidelines as gl, journals, pmc_fulltext, retractions
from nrag.ingest.pubmed import (
    PubMedClient, Rules, build_query, classify_design, india_query, exclusion_reason, extract_sample_size,
    parse_articles, recency_factor, reference_year, score,
)
from nrag.net import CachedHttp

EXCLUDED_KEYS = ("doc_id", "pmid", "doi", "title", "year", "topics", "design", "exclusion_reason",
                 "duplicate_of", "retraction_notices")
SOURCE_TYPE = {"guideline": "guideline", "meta_analysis": "review", "systematic_review": "review",
               "narrative_review": "review"}


@dataclass
class Clients:
    pubmed: PubMedClient
    ncbi: CachedHttp          # rate-limited for NCBI
    web: CachedHttp           # other hosts (Retraction Watch CSV)
    ncbi_params: dict[str, str]


def _rel(path: Path) -> str:
    """Repo-relative path when inside the repo, else absolute."""
    try:
        return str(path.relative_to(resolve_path(".")))
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def search_topics(cfg, client: PubMedClient, since, max_per_topic, topics, design_filter, today):
    """Return ({pmid: topics}, {pmid: sources}, {search: total hits})."""
    pm = cfg["pubmed"]
    ib = pm["india_boost"]
    mindate, maxdate = (since, pm["date_range"][1]) if since else pm["date_range"]
    datetype = "edat" if since else "pdat"   # entry date for refreshes
    hits: dict[str, set[str]] = {}
    sources: dict[str, set[str]] = {}
    totals: dict[str, int] = {}
    wanted = set(topics or [])

    def add(pmid, topic, source):
        hits.setdefault(pmid, set()).add(topic)
        sources.setdefault(pmid, set()).add(source)

    for topic, q in pm["topics"].items():
        if wanted and topic not in wanted:
            continue
        total, ids = client.search(build_query(cfg, q, design_filter), mindate, maxdate, datetype,
                                   limit=max_per_topic, today=today)
        totals[topic] = total
        for pmid in ids:
            add(pmid, topic, "main")
        msg = f"[{topic}] {total} hits, taking {len(ids)}"
        if ib["enabled"]:
            quota = math.ceil(ib["fraction"] * max_per_topic) if max_per_topic else None
            # Over-fetch so the quota is filled with records the main search didn't return.
            itotal, iids = client.search(india_query(cfg, topic, design_filter), mindate, maxdate, datetype,
                                         limit=quota * 2 + len(ids) if quota else None, today=today)
            new = [p for p in iids if p not in set(ids)]
            new = new[:quota] if quota else new
            totals[f"{topic}:india_boost"] = itotal
            for pmid in new:
                add(pmid, topic, "india_boost")
            msg += f"; India boost {itotal} hits, adding {len(new)} new"
        print(msg)
    return hits, sources, totals


def fetch_and_merge(client: PubMedClient, store: dict[str, dict], hits: dict[str, set[str]], today: date,
                    sources: dict[str, set[str]] | None = None):
    """Fetch new hits and refetch everything already stored; merge topics and search sources."""
    sources = sources or {}
    ids = sorted(set(store) | set(hits), key=int)   # stable batches -> cache-friendly
    returned = 0
    for chunk in client.fetch(ids):
        for rec in parse_articles(chunk):
            old = store.get(rec["pmid"], {})
            rec["topics"] = sorted(set(old.get("topics", [])) | hits.get(rec["pmid"], set()))
            # Records stored before search sources were tracked came from the main searches.
            prior = set(old.get("search_sources", ["main"])) if old else set()
            rec["search_sources"] = sorted(prior | sources.get(rec["pmid"], set()) or {"main"})
            rec["first_seen"] = old.get("first_seen", today.isoformat())
            store[rec["pmid"]] = rec
            returned += 1
    return returned


def derive(rules: Rules, cfg, rec: dict, indexed: set[str], rw: retractions.RetractionIndex, ref_year: int):
    """Compute every derived field from the raw record. Returns the new document dict."""
    d = dict(rec)
    rw_hit = rw.lookup(d["doi"], d["pmid"])
    sources = [s for s, hit in (("pubmed", d["pubmed_retracted"]), ("retraction_watch", rw_hit["retracted"])) if hit]
    d.update({
        "doc_id": f"PMID:{d['pmid']}",
        "retracted": bool(sources),
        "retraction_sources": sources,
        "expression_of_concern": d["pubmed_expression_of_concern"] or rw_hit["expression_of_concern"],
        "retraction_notices": rw_hit["notices"],
        "medline_indexed": journals.is_medline(d, indexed),
    })
    design, dscore, method = classify_design(rules, d["pub_types"], d["mesh"], d["title"], d["abstract_sections"])
    d.update({
        "design": design, "design_score": dscore, "design_method": method,
        "source_type": SOURCE_TYPE.get(design, "primary"),
        "sample_size": extract_sample_size(rules, d["abstract"]),
        "superseded_by": None,   # supersession pass (Phase 3)
        "pico": None,            # LLM extraction (Phase 2)
    })
    reason = exclusion_reason(rules, d)
    if reason:
        d["status"], d["status_reason"] = "excluded", reason
    else:
        d["status"], d["status_reason"] = credibility.assess(cfg["credibility"], d)
    score(rules, d, ref_year)
    return d


def dedupe_dois(docs: list[dict]) -> None:
    """Records sharing a DOI are one paper. If any copy is retracted, all are;
    otherwise keep the earliest seen non-excluded copy (then lowest PMID)."""
    by_doi: dict[str, list[dict]] = {}
    for d in docs:
        if d.get("doi"):
            by_doi.setdefault(d["doi"], []).append(d)
    for all_copies in by_doi.values():
        if len(all_copies) < 2:
            continue
        retracted = [d for d in all_copies if d["retracted"]]
        if retracted:
            for d in all_copies:
                if not d["retracted"]:
                    d.update(status="excluded", status_reason="retracted", retracted=True,
                             retraction_sources=[f"doi_duplicate_of:{retracted[0]['doc_id']}"])
            continue
        group = sorted((d for d in all_copies if d["status"] != "excluded"),
                       key=lambda d: (d["first_seen"], int(d["pmid"])))
        if len(group) < 2:
            continue
        keep = group[0]
        for dup in group[1:]:
            keep["topics"] = sorted(set(keep["topics"]) | set(dup["topics"]))
            dup.update(status="excluded", status_reason="duplicate_doi", duplicate_of=keep["doc_id"])


def attach_fulltext(cfg, clients: Clients, docs: list[dict], out_dir: Path) -> Counter:
    candidates = {d["pmcid"]: d for d in docs if d["status"] != "excluded" and d.get("pmcid")}
    licences: Counter = Counter()
    for d in docs:
        d.setdefault("licence", None)
        d.setdefault("fulltext_path", None)
    if not candidates:
        return licences
    out_dir.mkdir(parents=True, exist_ok=True)
    order = sorted(candidates, key=lambda p: int(p[3:]) if p[3:].isdigit() else 0)
    for art in pmc_fulltext.fetch_fulltext(clients.ncbi, cfg, order, clients.ncbi_params):
        d = candidates.get(art["pmcid"])
        if d is None:
            continue
        d["licence"], d["licence_url"] = art["licence"], art["licence_url"]
        licences[art["licence"]] += 1
        if art["allowed"] and (art["sections"] or art["tables"]):
            path = out_dir / f"{art['pmcid']}.json"
            path.write_text(json.dumps({"doc_id": d["doc_id"], **art}, ensure_ascii=False, indent=1))
            d["fulltext_path"] = _rel(path)
            d["fulltext_sections"] = sorted({s["name"] for s in art["sections"]})
            d["n_tables"] = len(art["tables"])
    return licences


def guideline_docs(cfg, rules: Rules, out_dir: Path, ref_year: int) -> tuple[list[dict], list[str]]:
    entries = gl.load_registry(cfg)
    superseded_by = {e.supersedes: e.id for e in entries if e.supersedes}
    gscore = next(r[2] for r in rules.pubtype_rules if r[0] == "guideline")
    rc = rules.scoring["recency"]
    docs, missing = [], []
    for e in entries:
        parsed = gl.parse_entry(e, cfg)
        if parsed is None:
            missing.append(e.id)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"guideline_{e.id}.json"
        path.write_text(json.dumps(parsed, ensure_ascii=False, indent=1))
        rf = recency_factor(e.year, ref_year, rc["half_life_years"], rc["floor"])
        sup = superseded_by.get(e.id)
        docs.append({
            "doc_id": parsed["doc_id"], "source_type": "guideline", "design": "guideline",
            "design_score": gscore, "design_method": "registry", "title": e.title, "year": e.year,
            "issuing_body": e.issuing_body, "version": e.version, "url": e.url, "topics": e.topics,
            "population_tags": e.population, "status": "included", "status_reason": None,
            "superseded_by": f"GUIDELINE:{sup}" if sup else None,
            "licence": "user-supplied-pdf", "fulltext_path": _rel(path),
            "n_sections": len(parsed["sections"]), "n_tables": len(parsed["tables"]),
            "n_recommendations": len(parsed["recommendations"]),
            "score_components": {"design": gscore, "size": 1.0, "recency": round(rf, 3)},
            "quality_score": round(gscore * rf, 4),
            "flags": (["superseded"] if sup or e.status == "superseded" else []) + (["verify_registry"] if e.verify else []),
        })
    return docs, missing


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(cfg, clients: Clients, since=None, max_per_topic=None, topics=None, design_filter=None,
        fulltext=True, guidelines=True, today: date | None = None) -> dict[str, Any]:
    today = today or date.today()
    rules = Rules.from_config(cfg)
    ref_year = reference_year(rules, today)
    paths = {k: resolve_path(v) for k, v in cfg["paths"].items()}
    design_filter = cfg["pubmed"]["design_filter"] if design_filter is None else design_filter

    store = {r["pmid"]: r for r in read_jsonl(paths["pubmed_records"])}
    previous_docs = {d["doc_id"] for d in read_jsonl(paths["documents"])}
    print(f"Store: {len(store)} PubMed records; previous corpus: {len(previous_docs)} documents")

    hits, sources, totals = search_topics(cfg, clients.pubmed, since, max_per_topic, topics, design_filter, today)
    n_new = len(set(hits) - set(store))
    returned = fetch_and_merge(clients.pubmed, store, hits, today, sources)
    write_jsonl(paths["pubmed_records"], sorted(store.values(), key=lambda r: int(r["pmid"])))
    print(f"Fetched {returned} records ({n_new} new); store now {len(store)}")

    indexed = journals.currently_indexed(clients.ncbi, cfg, (r.get("nlm_unique_id") for r in store.values()),
                                         clients.ncbi_params)
    rw = retractions.load_retraction_index(clients.web, cfg)
    print(f"NLM Catalog: {len(indexed)} journals currently indexed; Retraction Watch keys: {len(rw)}")

    docs = [derive(rules, cfg, r, indexed, rw, ref_year) for r in store.values()]
    dedupe_dois(docs)
    licences = attach_fulltext(cfg, clients, docs, paths["fulltext_dir"]) if fulltext and cfg["pmc"]["enabled"] else Counter()

    g_docs, g_missing = guideline_docs(cfg, rules, paths["fulltext_dir"], ref_year) if guidelines else ([], [])
    kept = [d for d in docs if d["status"] != "excluded"] + g_docs
    kept.sort(key=lambda d: (-d["quality_score"], d["doc_id"]))
    excluded = sorted((d for d in docs if d["status"] == "excluded"), key=lambda d: int(d["pmid"]))
    for d in excluded:
        d["exclusion_reason"] = d["status_reason"]

    write_jsonl(paths["documents"], kept)
    write_jsonl(paths["excluded"], ({k: d[k] for k in EXCLUDED_KEYS if k in d} for d in excluded))

    reasons = {d["doc_id"]: d["status_reason"] for d in excluded}
    kept_ids = {d["doc_id"] for d in kept}
    removals = [{"doc_id": i, "date": today.isoformat(), "reason": reasons.get(i, "no_longer_in_store_or_registry")}
                for i in sorted(previous_docs - kept_ids)]
    if removals:
        paths["removals_log"].parent.mkdir(parents=True, exist_ok=True)
        with paths["removals_log"].open("a") as f:
            for r in removals:
                f.write(json.dumps(r) + "\n")

    summary = summarize(cfg, kept, excluded, totals, licences, g_missing, removals, indexed, store, today,
                        dict(since=since, max_per_topic=max_per_topic, topics=topics, design_filter=design_filter))
    paths["summary"].write_text(json.dumps(summary, indent=1))
    return summary


def _count(it) -> dict[str, int]:
    return dict(Counter(it).most_common())


def summarize(cfg, kept, excluded, totals, licences, g_missing, removals, indexed, store, today, args):
    pubmed_docs = [d for d in kept if d["source_type"] != "guideline" or d.get("pmid")]
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    return {
        "run_date": today.isoformat(), "config_hash": cfg_hash, "args": args,
        "pubmed_hits_by_topic": totals,
        "store_size": len(store),
        "documents": len(kept),
        "by_status": _count(d["status"] if not d["status_reason"] else f"{d['status']}:{d['status_reason']}" for d in kept),
        "by_topic": _count(t for d in kept for t in d["topics"]),
        "by_design": _count(d["design"] for d in kept),
        "by_source_type": _count(d["source_type"] for d in kept),
        "by_year": {str(y or "unknown"): n for y, n in sorted(Counter(d["year"] for d in kept).items(),
                                                              key=lambda kv: kv[0] or 0)},
        "design_decided_by": _count(d["design_method"] for d in kept),
        "fulltext_licences_seen": dict(licences.most_common()),
        "fulltext_stored": sum(1 for d in pubmed_docs if d.get("fulltext_path")),
        "flags": _count(f for d in kept for f in d["flags"]),
        "excluded": len(excluded),
        "exclusion_reasons": _count(d["status_reason"] for d in excluded),
        "retracted_in_store": {
            "pubmed_only": sum(1 for d in excluded if d.get("retraction_sources") == ["pubmed"]),
            "retraction_watch_only": sum(1 for d in excluded if d.get("retraction_sources") == ["retraction_watch"]),
            "both": sum(1 for d in excluded if len(d.get("retraction_sources", [])) == 2),
        },
        "without_prospective_cohorts": sum(1 for d in kept if d["design"] != "prospective_cohort"),
        "by_search_source": _count(src for d in pubmed_docs for src in d.get("search_sources", [])),
        "india_boost_only": sum(1 for d in pubmed_docs if d.get("search_sources") == ["india_boost"]),
        "journals": {"unique_nlm_ids": len({r.get("nlm_unique_id") for r in store.values() if r.get("nlm_unique_id")}),
                     "currently_indexed": len(indexed)},
        "guidelines": {"parsed": [d["doc_id"] for d in kept if d["source_type"] == "guideline" and not d.get("pmid")],
                       "awaiting_pdf": g_missing},
        "removed_this_run": removals,
    }


def print_summary(s: dict[str, Any]) -> None:
    print("\n=== Corpus summary ===")
    print(f"Documents: {s['documents']}   Excluded: {s['excluded']}   (store: {s['store_size']} PubMed records)")
    for key in ("pubmed_hits_by_topic", "by_status", "by_topic", "by_design", "by_source_type", "by_year",
                "design_decided_by", "fulltext_licences_seen", "flags", "exclusion_reasons", "retracted_in_store",
                "journals", "by_search_source"):
        print(f"{key}: {s[key]}")
    print(f"fulltext_stored: {s['fulltext_stored']}")
    print(f"without prospective cohorts: {s['without_prospective_cohorts']} documents")
    main_only = s["documents"] - s["india_boost_only"] - len(s["guidelines"]["parsed"])
    if main_only:
        print(f"India boost: {s['india_boost_only']} documents found only by India searches "
              f"(+{s['india_boost_only'] / main_only:.0%} on top of the main searches)")
    print(f"guidelines parsed: {s['guidelines']['parsed'] or 'none'}; awaiting PDF: {s['guidelines']['awaiting_pdf']}")
    if s["removed_this_run"]:
        print(f"removed this run: {len(s['removed_this_run'])} (see removals log)")
