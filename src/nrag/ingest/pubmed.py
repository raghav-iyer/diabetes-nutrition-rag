"""PubMed ingestion, ported from legacy/ingest_pubmed.py.

Fetches papers for each topic query, then:
  1. deduplicates by PMID / DOI (merging topic tags)
  2. hard-filters: humans only, not retracted, has abstract, language
  3. classifies study design from publication types (+ MeSH and text fallback)
  4. extracts a rough sample size
  5. computes a first-pass quality score (design x size x recency)
  6. writes the kept corpus and the excluded records (with reasons)

All rules, weights and thresholds come from config.yaml. Behaviour matches the
legacy script exactly (tests/test_parity.py); known issues are fixed in Phase 1.
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Iterator

from nrag.net import CachedHttp

EXCLUDED_KEYS = ("pmid", "doi", "title", "year", "topics", "exclusion_reason")


# ---------------------------------------------------------------------------
# Rules compiled from config
# ---------------------------------------------------------------------------
@dataclass
class Rules:
    pubtype_rules: list[tuple[str, set[str], float, bool]]
    mesh_rules: list[tuple[str, set[str], float]]
    text_rules: list[tuple[str, re.Pattern, float]]
    unknown_score: float
    n_patterns: list[re.Pattern]
    n_min: int
    n_max: int
    n_skip: tuple[int, int]
    exclusion: dict[str, Any]
    scoring: dict[str, Any]

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Rules":
        d, ss = cfg["design"], cfg["sample_size"]
        return cls(
            pubtype_rules=[(r["label"], set(r["pub_types"]), r["score"], r.get("defer", False))
                           for r in d["pubtype_rules"]],
            mesh_rules=[(r["label"], set(r["mesh"]), r["score"]) for r in d["mesh_rules"]],
            text_rules=[(r["label"], re.compile(r["pattern"], re.I), r["score"]) for r in d["text_rules"]],
            unknown_score=d["unknown_score"],
            n_patterns=[re.compile(p) for p in ss["patterns"]],
            n_min=ss["min"],
            n_max=ss["max"],
            n_skip=tuple(ss["skip_range"]),
            exclusion=cfg["exclusion"],
            scoring=cfg["scoring"],
        )


# ---------------------------------------------------------------------------
# Study-design classification
# ---------------------------------------------------------------------------
def classify_design(rules: Rules, pub_types, mesh_terms, title, abstract):
    """Return (design_label, base_score, method). method says how it was decided."""
    pts = set(pub_types)
    deferred = None
    for label, types, score, defer in rules.pubtype_rules:
        if pts & types:
            if defer:
                # e.g. "Review" alone: check MeSH and text for a stronger design first.
                deferred = (label, score)
                break
            return label, score, "pubtype"
    meshes = set(mesh_terms)
    for label, terms, score in rules.mesh_rules:
        if meshes & terms:
            return label, score, "mesh"
    text = f"{title} {abstract}"
    for label, pattern, score in rules.text_rules:
        if pattern.search(text):
            return label, score, "text"
    if deferred:
        return deferred[0], deferred[1], "pubtype"
    return "unknown", rules.unknown_score, "none"


# ---------------------------------------------------------------------------
# Sample size (rough; replaced by LLM extraction in Phase 2)
# ---------------------------------------------------------------------------
def extract_sample_size(rules: Rules, abstract: str):
    """Return the largest plausible participant count found, or None."""
    best = None
    lo, hi = rules.n_skip
    for pattern in rules.n_patterns:
        for m in pattern.finditer(abstract):
            try:
                n = int(m.group(1).replace(",", ""))
            except ValueError:
                continue
            if rules.n_min <= n <= rules.n_max and not (lo <= n <= hi):
                best = max(best or 0, n)
    return best


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def size_factor(size_cfg: dict[str, Any], design: str, n):
    """Small studies count for less. Not applied to guidelines/reviews."""
    if design in size_cfg["exempt_designs"]:
        return 1.0
    if n is None:
        return size_cfg["unknown_n"]
    for bound, factor in size_cfg["bands"]:
        if n < bound:
            return factor
    return size_cfg["large"]


def recency_factor(year, ref_year: int, half_life: float, floor: float):
    """Gentle tiebreaker: never drops below `floor`, halves the remainder every half_life years."""
    if not year:
        return floor
    age = max(0, ref_year - year)
    return floor + (1 - floor) * (0.5 ** (age / half_life))


def reference_year(rules: Rules, today: date) -> int:
    return rules.scoring["recency"]["reference_year"] or today.year


def score(rules: Rules, rec: dict[str, Any], ref_year: int) -> dict[str, Any]:
    sc = rules.scoring
    rc = sc["recency"]
    sf = size_factor(sc["size"], rec["design"], rec["sample_size"])
    rf = recency_factor(rec["year"], ref_year, rc["half_life_years"], rc["floor"])
    q = rec["design_score"] * sf * rf
    flags = []
    if rec["expression_of_concern"]:
        q *= sc["expression_of_concern_multiplier"]
        flags.append("expression_of_concern")
    if not rec["medline_indexed"]:
        flags.append("not_medline_indexed")
    if not rec["mesh"]:
        flags.append("not_yet_mesh_indexed")
    rec["score_components"] = {"design": rec["design_score"], "size": round(sf, 3), "recency": round(rf, 3)}
    rec["quality_score"] = round(q, 4)
    rec["flags"] = flags
    return rec


def exclusion_reason(rules: Rules, rec: dict[str, Any]):
    ex = rules.exclusion
    if rec["retracted"]:
        return "retracted"
    if set(rec["pub_types"]) & set(ex["notice_pub_types"]):
        return "notice_not_article"
    if not rec["abstract"] or len(rec["abstract"]) < ex["min_abstract_chars"]:
        return "no_or_short_abstract"
    if ex["required_mesh"] not in rec["mesh"] and rec["mesh"]:
        # Only reject when MeSH exists; brand-new records may not be indexed yet.
        return "not_human"
    if rec["languages"] and ex["language"] not in rec["languages"]:
        return "not_english"
    if rec["design"] in ex["excluded_designs"]:
        return "opinion_piece"
    return None


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------
def _text(el):
    return "".join(el.itertext()).strip() if el is not None else ""


def parse_articles(xml_bytes: bytes) -> Iterator[dict[str, Any]]:
    root = ET.fromstring(xml_bytes)
    for art in root.iter("PubmedArticle"):
        mc = art.find("MedlineCitation")
        a = mc.find("Article")
        pmid = _text(mc.find("PMID"))

        # Abstract, keeping section labels for structured abstracts.
        parts = []
        for ab in a.findall("Abstract/AbstractText"):
            label = ab.get("Label")
            txt = _text(ab)
            parts.append(f"{label}: {txt}" if label else txt)
        abstract = "\n".join(p for p in parts if p)

        # Year: prefer journal issue year, fall back to MedlineDate or ArticleDate.
        year = None
        y = a.find("Journal/JournalIssue/PubDate/Year")
        if y is not None and y.text and y.text.isdigit():
            year = int(y.text)
        else:
            md = _text(a.find("Journal/JournalIssue/PubDate/MedlineDate"))
            m = re.search(r"(19|20)\d{2}", md)
            if m:
                year = int(m.group(0))
            else:
                ad = a.find("ArticleDate/Year")
                if ad is not None and ad.text and ad.text.isdigit():
                    year = int(ad.text)

        doi = None
        for aid in art.findall("PubmedData/ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = (aid.text or "").strip().lower() or None
        if not doi:
            for eid in a.findall("ELocationID"):
                if eid.get("EIdType") == "doi":
                    doi = (eid.text or "").strip().lower() or None

        pmcid = None
        for aid in art.findall("PubmedData/ArticleIdList/ArticleId"):
            if aid.get("IdType") == "pmc":
                pmcid = (aid.text or "").strip() or None

        pub_types = [_text(p) for p in a.findall("PublicationTypeList/PublicationType")]
        mesh = [_text(d.find("DescriptorName")) for d in mc.findall("MeshHeadingList/MeshHeading")]
        mesh_major = [
            _text(d.find("DescriptorName"))
            for d in mc.findall("MeshHeadingList/MeshHeading")
            if d.find("DescriptorName").get("MajorTopicYN") == "Y"
            or any(q.get("MajorTopicYN") == "Y" for q in d.findall("QualifierName"))
        ]
        languages = [_text(lang) for lang in a.findall("Language")]

        # Retraction / correction links recorded by NLM.
        corrections = {cc.get("RefType") for cc in mc.findall("CommentsCorrectionsList/CommentsCorrections")}

        coi = _text(art.find("MedlineCitation/CoiStatement"))
        grants = [_text(g.find("Agency")) for g in a.findall("GrantList/Grant") if g.find("Agency") is not None]

        yield {
            "pmid": pmid,
            "doi": doi,
            "pmcid": pmcid,
            "title": _text(a.find("ArticleTitle")),
            "abstract": abstract,
            "journal": _text(a.find("Journal/Title")),
            "issn": _text(a.find("Journal/ISSN")) or None,
            "year": year,
            "languages": languages,
            "pub_types": pub_types,
            "mesh": mesh,
            "mesh_major": mesh_major,
            "medline_indexed": mc.get("Status") == "MEDLINE",
            "retracted": bool(set(pub_types) & {"Retracted Publication"}) or "RetractionIn" in corrections,
            "expression_of_concern": "ExpressionOfConcernIn" in corrections,
            "has_erratum": "ErratumIn" in corrections,
            "coi_statement": coi or None,
            "funders": grants,
            "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }


# ---------------------------------------------------------------------------
# NCBI client
# ---------------------------------------------------------------------------
class PubMedClient:
    """E-utilities client. esearch returns PMIDs; efetch POSTs them in batches,
    so every response is keyed by its exact inputs and cacheable."""

    def __init__(self, http: CachedHttp, ncbi_cfg: dict[str, Any], cache_cfg: dict[str, Any],
                 email: str, api_key: str | None = None):
        self.http = http
        self.cfg = ncbi_cfg
        self.cache_cfg = cache_cfg
        self.base = {"tool": ncbi_cfg["tool"], "email": email}
        if api_key:
            self.base["api_key"] = api_key

    def search(self, term: str, mindate: str, maxdate: str, datetype: str = "pdat",
               limit: int | None = None) -> tuple[int, list[str]]:
        cap = self.cfg["esearch_max_ids"]
        retmax = min(limit, cap) if limit else cap
        content = self.http.get(f"{self.cfg['eutils_base']}/esearch.fcgi", params={
            **self.base, "db": "pubmed", "term": term, "retmode": "json",
            "datetype": datetype, "mindate": mindate, "maxdate": maxdate, "retmax": retmax,
        }, ttl_days=self.cache_cfg["esearch_ttl_days"])
        res = json.loads(content)["esearchresult"]
        count = int(res["count"])
        if count > cap and (limit is None or limit > cap):
            print(f"  WARNING: {count} hits but PubMed returns at most {cap} per search; "
                  f"only the first {cap} are fetched.", file=sys.stderr)
        return count, res.get("idlist", [])

    def fetch(self, pmids: list[str]) -> Iterator[bytes]:
        batch = self.cfg["efetch_batch"]
        for start in range(0, len(pmids), batch):
            yield self.http.post(f"{self.cfg['eutils_base']}/efetch.fcgi", data={
                **self.base, "db": "pubmed", "retmode": "xml",
                "id": ",".join(pmids[start:start + batch]),
            }, ttl_days=self.cache_cfg["efetch_ttl_days"])


def build_query(cfg: dict[str, Any], topic_query: str, strong_only: bool = False) -> str:
    pm = cfg["pubmed"]
    query = f"({topic_query}) AND {pm['base_filters']}"
    if strong_only:
        query = f"({query}) AND {pm['strong_designs']}"
    return query


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def ingest(
    cfg: dict[str, Any],
    client: PubMedClient,
    corpus_path: Path,
    excluded_path: Path,
    since: str | None = None,
    max_per_topic: int | None = None,
    topics: Iterable[str] | None = None,
    strong_only: bool = False,
    today: date | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    today = today or date.today()
    rules = Rules.from_config(cfg)
    ref_year = reference_year(rules, today)

    # Load existing corpus so refreshes merge rather than overwrite.
    by_pmid = {r["pmid"]: r for r in read_jsonl(corpus_path)}
    if by_pmid:
        print(f"Loaded {len(by_pmid)} existing records")

    mindate = since or cfg["pubmed"]["date_range"][0]
    maxdate = cfg["pubmed"]["date_range"][1]
    datetype = "edat" if since else "pdat"  # entry date for refreshes

    excluded: list[dict[str, Any]] = []
    seen_dois = {r["doi"]: r["pmid"] for r in by_pmid.values() if r.get("doi")}
    wanted = set(topics or [])
    selected = {k: v for k, v in cfg["pubmed"]["topics"].items() if not wanted or k in wanted}

    for topic, q in selected.items():
        count, pmids = client.search(build_query(cfg, q, strong_only), mindate, maxdate, datetype,
                                     limit=max_per_topic)
        total = min(count, max_per_topic) if max_per_topic else count
        print(f"[{topic}] {count} hits, fetching {min(total, len(pmids))}")

        n_new = 0
        for chunk in client.fetch(pmids[:total]):
            for rec in parse_articles(chunk):
                pmid = rec["pmid"]
                # Duplicate by PMID: just add the topic tag.
                if pmid in by_pmid:
                    by_pmid[pmid]["topics"] = sorted(set(by_pmid[pmid]["topics"]) | {topic})
                    # Refresh retraction status on re-fetch.
                    if rec["retracted"] and not by_pmid[pmid]["retracted"]:
                        excluded.append({**by_pmid.pop(pmid), "exclusion_reason": "retracted_since_ingest"})
                    continue
                # Duplicate by DOI under a different PMID (rare, but happens).
                if rec["doi"] and rec["doi"] in seen_dois:
                    other = seen_dois[rec["doi"]]
                    if other in by_pmid:
                        by_pmid[other]["topics"] = sorted(set(by_pmid[other]["topics"]) | {topic})
                    continue

                design, dscore, method = classify_design(
                    rules, rec["pub_types"], rec["mesh"], rec["title"], rec["abstract"])
                rec.update({
                    "topics": [topic],
                    "design": design,
                    "design_score": dscore,
                    "design_method": method,
                    "sample_size": extract_sample_size(rules, rec["abstract"]),
                    "superseded_by": None,  # filled by the supersession pass (Phase 3)
                    "pico": None,           # filled by LLM extraction (Phase 2)
                    "ingested": today.isoformat(),
                })
                reason = exclusion_reason(rules, rec)
                if reason:
                    excluded.append({**rec, "exclusion_reason": reason})
                    continue
                by_pmid[pmid] = score(rules, rec, ref_year)
                if rec["doi"]:
                    seen_dois[rec["doi"]] = pmid
                n_new += 1
        print(f"[{topic}] {n_new} new records kept")

    # Recompute recency for everything (scores drift as the year changes).
    for rec in by_pmid.values():
        score(rules, rec, ref_year)

    records = sorted(by_pmid.values(), key=lambda r: r["quality_score"], reverse=True)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with excluded_path.open("a") as f:
        for rec in excluded:
            f.write(json.dumps({k: rec[k] for k in EXCLUDED_KEYS if k in rec}, ensure_ascii=False) + "\n")
    return records, excluded


def print_summary(records: list[dict[str, Any]], excluded: list[dict[str, Any]]) -> None:
    print("\n=== Corpus summary ===")
    print(f"Kept: {len(records)}   Excluded this run: {len(excluded)}")
    print("By design:", dict(Counter(r["design"] for r in records).most_common()))
    print("By topic: ", dict(Counter(t for r in records for t in r["topics"]).most_common()))
    print("Design decided by:", dict(Counter(r["design_method"] for r in records)))
    print("Exclusions:", dict(Counter(r["exclusion_reason"] for r in excluded).most_common()))
    print("Flags:", dict(Counter(f for r in records for f in r["flags"]).most_common()))
    if records:
        print("\nTop 5 by quality score:")
        for r in records[:5]:
            print(f"  {r['quality_score']:.3f}  {r['design']:<18} {r['year']}  {r['title'][:80]}")
