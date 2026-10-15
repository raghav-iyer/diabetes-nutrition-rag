"""PubMed search, fetch, parsing, design classification and first-pass scoring.

Originally ported from legacy/ingest_pubmed.py. Phase 1 changes (each tested):
  - searches are split by year (and further by date if a year exceeds PubMed's
    10k cap) so large topics are fetched completely
  - design text rules only look at the title and the parts of the abstract that
    describe this study, not background sentences about prior work
  - `Observational Study` defers to MeSH; `Multicenter Study` is not a design;
    prospective cohorts are labelled separately
  - sample sizes 1900-2100 are only skipped when they read like a year
  - parsing keeps abstract sections, the NLM journal ID and citation status
Orchestration (merging, refresh, screening, output) lives in pipeline.py.
"""

from __future__ import annotations

import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterator

from nrag.net import CachedHttp

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Rules compiled from config
# ---------------------------------------------------------------------------
@dataclass
class Rules:
    pubtype_rules: list[tuple[str, set[str], float, bool]]
    mesh_rules: list[tuple[str, set[str], float]]
    text_rules: list[tuple[str, re.Pattern, float]]
    background_label: re.Pattern
    prior_work: re.Pattern
    unknown_score: float
    n_patterns: list[re.Pattern]
    n_min: int
    n_max: int
    n_skip: tuple[int, int]
    n_year_context: re.Pattern
    exclusion: dict[str, Any]
    scoring: dict[str, Any]

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Rules":
        d, ss = cfg["design"], cfg["sample_size"]
        scope = d["text_scope"]
        return cls(
            pubtype_rules=[(r["label"], set(r["pub_types"]), r["score"], r.get("defer", False))
                           for r in d["pubtype_rules"]],
            mesh_rules=[(r["label"], set(r["mesh"]), r["score"]) for r in d["mesh_rules"]],
            text_rules=[(r["label"], re.compile(r["pattern"], re.I), r["score"]) for r in d["text_rules"]],
            background_label=re.compile(scope["background_labels"], re.I),
            prior_work=re.compile(scope["prior_work_cues"], re.I),
            unknown_score=d["unknown_score"],
            n_patterns=[re.compile(p) for p in ss["patterns"]],
            n_min=ss["min"],
            n_max=ss["max"],
            n_skip=tuple(ss["skip_range"]),
            n_year_context=re.compile(ss["year_context"], re.I),
            exclusion=cfg["exclusion"],
            scoring=cfg["scoring"],
        )


# ---------------------------------------------------------------------------
# Study-design classification
# ---------------------------------------------------------------------------
def design_text(rules: Rules, title: str, sections: list[list[str]]) -> str:
    """Title plus the abstract text that describes this study.

    Structured abstracts drop background-type sections; unstructured ones drop
    sentences that talk about prior work ("previous meta-analyses showed...").
    """
    parts = [title]
    for label, text in sections:
        if label:
            if not rules.background_label.search(label):
                parts.append(text)
        else:
            parts.extend(s for s in SENTENCE_SPLIT.split(text) if not rules.prior_work.search(s))
    return " ".join(parts)


def classify_design(rules: Rules, pub_types, mesh_terms, title, abstract_sections):
    """Return (design_label, base_score, method). method says how it was decided."""
    pts = set(pub_types)
    deferred = None
    for label, types, score, defer in rules.pubtype_rules:
        if pts & types:
            if defer:
                # e.g. "Review" or "Observational Study": let MeSH/text be more specific first.
                deferred = (label, score)
                break
            return label, score, "pubtype"
    meshes = set(mesh_terms)
    for label, terms, score in rules.mesh_rules:
        if meshes & terms:
            return label, score, "mesh"
    text = design_text(rules, title, abstract_sections)
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
            raw = m.group(1)
            try:
                n = int(raw.replace(",", ""))
            except ValueError:
                continue
            if not rules.n_min <= n <= rules.n_max:
                continue
            # "in 2015 women were..." is a year; "2,015 women" or "n = 2015" is not.
            if lo <= n <= hi and "," not in raw and rules.n_year_context.search(abstract[:m.start(1)]):
                continue
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
    """Basic hygiene filters. Credibility rules are applied separately (credibility.py)."""
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
        sections = []
        for ab in a.findall("Abstract/AbstractText"):
            txt = _text(ab)
            if txt:
                sections.append([ab.get("Label") or "", txt])
        abstract = "\n".join(f"{label}: {txt}" if label else txt for label, txt in sections)

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
            "abstract_sections": sections,
            "journal": _text(a.find("Journal/Title")),
            "issn": _text(a.find("Journal/ISSN")) or None,
            "nlm_unique_id": _text(mc.find("MedlineJournalInfo/NlmUniqueID")) or None,
            "year": year,
            "languages": languages,
            "pub_types": pub_types,
            "mesh": mesh,
            "mesh_major": mesh_major,
            "citation_status": mc.get("Status"),
            "pubmed_retracted": bool(set(pub_types) & {"Retracted Publication"}) or "RetractionIn" in corrections,
            "pubmed_expression_of_concern": "ExpressionOfConcernIn" in corrections,
            "has_erratum": "ErratumIn" in corrections,
            "coi_statement": coi or None,
            "funders": grants,
            "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }


# ---------------------------------------------------------------------------
# NCBI client
# ---------------------------------------------------------------------------
def _d(s: str) -> date:
    y, m, d = (int(x) for x in s.split("/"))
    return date(y, m, d)


def _s(d: date) -> str:
    return d.strftime("%Y/%m/%d")


def year_slices(mindate: str, maxdate: str, today: date) -> list[tuple[date, date]]:
    """Calendar-year slices covering [mindate, min(maxdate, today)], newest first."""
    start, end = _d(mindate), min(_d(maxdate), today)
    out = []
    for y in range(end.year, start.year - 1, -1):
        out.append((max(start, date(y, 1, 1)), min(end, date(y, 12, 31))))
    return out


def allocate(limit: int, counts: list[int]) -> list[int]:
    """Split `limit` across slices in proportion to their counts (largest remainder)."""
    total = sum(counts)
    if total <= limit:
        return list(counts)
    raw = [limit * c / total for c in counts]
    q = [min(math.floor(r), c) for r, c in zip(raw, counts)]
    order = sorted(range(len(counts)), key=lambda i: raw[i] - q[i], reverse=True)
    i = 0
    while sum(q) < limit:
        j = order[i % len(order)]
        if q[j] < counts[j]:
            q[j] += 1
        i += 1
    return q


class PubMedClient:
    """E-utilities client. esearch returns PMIDs (sliced by date to stay under the
    10k cap); efetch POSTs ID batches, so every response is keyed by its inputs."""

    def __init__(self, http: CachedHttp, ncbi_cfg: dict[str, Any], cache_cfg: dict[str, Any],
                 email: str | None = None, api_key: str | None = None, sort: str = "pub_date"):
        self.http = http
        self.cfg = ncbi_cfg
        self.cache_cfg = cache_cfg
        self.sort = sort
        self.base = {"tool": ncbi_cfg["tool"]}
        if email:
            self.base["email"] = email
        if api_key:
            self.base["api_key"] = api_key

    def _esearch(self, term, start: date, end: date, datetype: str, retmax: int) -> dict[str, Any]:
        content = self.http.get(f"{self.cfg['eutils_base']}/esearch.fcgi", params={
            **self.base, "db": "pubmed", "term": term, "retmode": "json", "sort": self.sort,
            "datetype": datetype, "mindate": _s(start), "maxdate": _s(end), "retmax": retmax,
        }, ttl_days=self.cache_cfg["esearch_ttl_days"])
        return json.loads(content)["esearchresult"]

    def count(self, term, start: date, end: date, datetype: str) -> int:
        return int(self._esearch(term, start, end, datetype, 0)["count"])

    def _ids(self, term, start: date, end: date, datetype: str, want: int) -> list[str]:
        cap = self.cfg["esearch_max_ids"]
        if want <= cap:
            return self._esearch(term, start, end, datetype, want).get("idlist", [])[:want]
        if start == end:
            print(f"  WARNING: {want} records on {_s(start)} exceed the {cap} cap; truncating.", file=sys.stderr)
            return self._esearch(term, start, end, datetype, cap).get("idlist", [])
        mid = start + (end - start) // 2
        ids = []
        for a, b in ((mid + timedelta(days=1), end), (start, mid)):   # newest half first
            ids += self._ids(term, a, b, datetype, self.count(term, a, b, datetype))
        return ids

    def search(self, term: str, mindate: str, maxdate: str, datetype: str = "pdat",
               limit: int | None = None, today: date | None = None) -> tuple[int, list[str]]:
        """Return (total hits, PMIDs). With `limit`, PMIDs are sampled across years
        in proportion to each year's hits, so a small test run spans the date range.

        A paper can match two year slices (print and electronic dates differ), so
        the total comes from one full-range count and PMIDs are deduplicated."""
        slices = year_slices(mindate, maxdate, today or date.today())
        total = self.count(term, slices[-1][0], slices[0][1], datetype)
        counts = [self.count(term, a, b, datetype) for a, b in slices]
        wants = allocate(limit, counts) if limit else counts
        ids: list[str] = []
        for (a, b), want in zip(slices, wants):
            if want:
                ids += self._ids(term, a, b, datetype, want)
        ids = list(dict.fromkeys(ids))
        return total, ids[:limit] if limit else ids

    def fetch(self, pmids: list[str], ttl_days: float | None = None) -> Iterator[bytes]:
        batch = self.cfg["efetch_batch"]
        ttl = self.cache_cfg["efetch_ttl_days"] if ttl_days is None else ttl_days
        for start in range(0, len(pmids), batch):
            yield self.http.post(f"{self.cfg['eutils_base']}/efetch.fcgi", data={
                **self.base, "db": "pubmed", "retmode": "xml",
                "id": ",".join(pmids[start:start + batch]),
            }, ttl_days=ttl)


def build_query(cfg: dict[str, Any], topic_query: str, design_filter: bool = False) -> str:
    pm = cfg["pubmed"]
    query = f"({topic_query}) AND {pm['base_filters']}"
    if design_filter:
        query = f"({query}) AND {pm['strong_designs']}"
    return query
