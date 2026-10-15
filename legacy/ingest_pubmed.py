#!/usr/bin/env python3
"""
Ingestion pipeline for a nutrition-research RAG corpus.

Fetches papers from PubMed for each topic query, then:
  1. deduplicates by PMID / DOI (merging topic tags)
  2. hard-filters: humans only, not retracted, has abstract, language
  3. classifies study design from PubMed publication types (+ text fallback)
  4. extracts rough sample size
  5. computes a first-pass quality score (design x size x recency)
  6. writes corpus.jsonl (kept) and excluded.jsonl (dropped, with reasons)

Later stages (not in this script): LLM PICO extraction, risk-of-bias,
population-match scoring, chunking and indexing.

Usage:
  pip install requests
  export NCBI_EMAIL="you@example.com"      # required by NCBI policy
  export NCBI_API_KEY="..."                # optional: raises limit 3 -> 10 req/s
  python ingest_pubmed.py                   # full run
  python ingest_pubmed.py --since 2026/10/01   # incremental refresh
  python ingest_pubmed.py --max-per-topic 50   # quick test
"""

import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL_NAME = "nutrition_rag_ingest"

# ---------------------------------------------------------------------------
# Topic queries. Version-control these; every change alters the corpus.
# ---------------------------------------------------------------------------
STRONG_DESIGNS = (
    '(randomized controlled trial[pt] OR meta-analysis[pt] OR systematic review[pt] '
    'OR practice guideline[pt] OR guideline[pt] OR cohort studies[mh])'
)
BASE_FILTERS = 'humans[mh] AND english[la] AND hasabstract'

TOPICS = {
    "diabetes_prevention": (
        '("diabetes mellitus, type 2/prevention and control"[mh] OR prediabetic state[mh] '
        'OR prediabetes[tiab]) AND (diet[mh] OR "feeding behavior"[mh] OR "dietary pattern*"[tiab] '
        'OR "lifestyle intervention"[tiab])'
    ),
    "weight_management": (
        '("weight loss"[mh] OR "obesity/diet therapy"[mh] OR "weight maintenance"[tiab]) '
        'AND (diet[mh] OR "diet, reducing"[mh] OR "dietary pattern*"[tiab] OR "eating behavio*"[tiab])'
    ),
    "maternal_nutrition": (
        '(pregnancy[mh] OR "pregnant women"[mh] OR "prenatal nutritional physiological phenomena"[mh] '
        'OR "lactation"[mh]) AND (diet[mh] OR "dietary supplements"[mh] OR "maternal nutrition"[tiab] '
        'OR "gestational weight gain"[mh])'
    ),
    "gestational_diabetes": (
        '"diabetes, gestational"[mh] AND (diet[mh] OR "diet therapy"[sh] OR "dietary pattern*"[tiab] '
        'OR prevention and control[sh])'
    ),
}

DEFAULT_DATE_RANGE = ("2010/01/01", "3000/12/31")

# ---------------------------------------------------------------------------
# Study-design classification
# ---------------------------------------------------------------------------
# Ordered: first match wins, so stronger designs come first.
DESIGN_RULES = [
    ("guideline",          {"Practice Guideline", "Guideline", "Consensus Development Conference"}, 1.00),
    ("meta_analysis",      {"Meta-Analysis"}, 0.90),
    ("systematic_review",  {"Systematic Review"}, 0.85),
    ("rct",                {"Randomized Controlled Trial"}, 0.75),
    ("controlled_trial",   {"Controlled Clinical Trial", "Clinical Trial", "Clinical Trial, Phase III",
                            "Clinical Trial, Phase IV", "Pragmatic Clinical Trial"}, 0.60),
    ("observational",      {"Observational Study", "Multicenter Study"}, 0.45),
    ("narrative_review",   {"Review"}, 0.15),
    ("opinion",            {"Editorial", "Comment", "Letter", "News"}, 0.05),
]

# Used when PubMed publication types are generic (e.g. only "Journal Article").
TEXT_RULES = [
    ("meta_analysis",     re.compile(r"\bmeta-?analys[ie]s\b", re.I), 0.90),
    ("systematic_review", re.compile(r"\bsystematic(ally)? review", re.I), 0.85),
    ("rct",               re.compile(r"\brandomi[sz]ed\b.{0,40}\btrial\b", re.I), 0.75),
    ("cohort",            re.compile(r"\b(prospective|retrospective)?\s*cohort\b", re.I), 0.55),
    ("case_control",      re.compile(r"\bcase-?control\b", re.I), 0.40),
    ("cross_sectional",   re.compile(r"\bcross-?sectional\b", re.I), 0.30),
]

# MeSH cohort-design terms that PubMed indexes as subject headings, not pub types.
MESH_DESIGNS = [
    ("cohort",          {"Cohort Studies", "Prospective Studies", "Longitudinal Studies"}, 0.55),
    ("case_control",    {"Case-Control Studies"}, 0.40),
    ("cross_sectional", {"Cross-Sectional Studies"}, 0.30),
]

EXCLUDE_PUBTYPES = {"Retracted Publication", "Retraction of Publication", "Published Erratum",
                    "Expression of Concern"}


def classify_design(pub_types, mesh_terms, title, abstract):
    """Return (design_label, base_score, method). method says how it was decided."""
    pts = set(pub_types)
    for label, types, score in DESIGN_RULES:
        if pts & types:
            # "Review" alone is narrative, but check text for systematic review first.
            if label == "narrative_review":
                break
            return label, score, "pubtype"
    meshes = set(mesh_terms)
    for label, terms, score in MESH_DESIGNS:
        if meshes & terms:
            return label, score, "mesh"
    text = f"{title} {abstract}"
    for label, pattern, score in TEXT_RULES:
        if pattern.search(text):
            return label, score, "text"
    if "Review" in pts:
        return "narrative_review", 0.15, "pubtype"
    for label, types, score in DESIGN_RULES:
        if label == "opinion" and pts & types:
            return label, score, "pubtype"
    return "unknown", 0.20, "none"


# ---------------------------------------------------------------------------
# Sample size (rough; replace with LLM extraction later)
# ---------------------------------------------------------------------------
N_PATTERNS = [
    re.compile(r"\b[nN]\s*=\s*([\d,]{2,9})"),
    re.compile(r"\b([\d,]{2,9})\s+(?:participants|subjects|adults|women|men|patients|"
               r"pregnant women|individuals|people|children|mothers)\b", re.I),
    re.compile(r"\b([\d,]{1,4})\s+(?:studies|trials|RCTs)\b", re.I),  # for reviews
]


def extract_sample_size(abstract):
    """Return the largest plausible participant count found, or None."""
    best = None
    for pattern in N_PATTERNS[:2]:
        for m in pattern.finditer(abstract):
            try:
                n = int(m.group(1).replace(",", ""))
            except ValueError:
                continue
            if 5 <= n <= 50_000_000 and not (1900 <= n <= 2100):  # skip years
                best = max(best or 0, n)
    return best


def size_factor(design, n):
    """Small studies count for less. Not applied to guidelines/reviews."""
    if design in {"guideline", "meta_analysis", "systematic_review", "narrative_review", "opinion"}:
        return 1.0
    if n is None:
        return 0.85
    if n < 50:
        return 0.60
    if n < 200:
        return 0.80
    if n < 1000:
        return 0.90
    return 1.00


def recency_factor(year, half_life=10.0, floor=0.6):
    """Gentle tiebreaker: never drops below `floor`, halves the remainder every half_life years."""
    if not year:
        return floor
    age = max(0, date.today().year - year)
    return floor + (1 - floor) * (0.5 ** (age / half_life))


# ---------------------------------------------------------------------------
# NCBI client
# ---------------------------------------------------------------------------
class PubMed:
    def __init__(self, email, api_key=None):
        self.session = requests.Session()
        self.base = {"tool": TOOL_NAME, "email": email}
        if api_key:
            self.base["api_key"] = api_key
        self.delay = 0.11 if api_key else 0.34
        self._last = 0.0

    def _get(self, endpoint, params, retries=4):
        for attempt in range(retries):
            wait = self.delay - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.session.get(f"{EUTILS}/{endpoint}", params={**self.base, **params}, timeout=60)
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                return r
            except (requests.RequestException, requests.HTTPError) as e:
                if attempt == retries - 1:
                    raise
                backoff = 2 ** attempt
                print(f"  retry {attempt + 1} after error: {e} (sleep {backoff}s)", file=sys.stderr)
                time.sleep(backoff)

    def search(self, term, mindate, maxdate, datetype="pdat", retmax=100_000):
        r = self._get("esearch.fcgi", {
            "db": "pubmed", "term": term, "retmode": "json", "usehistory": "y",
            "datetype": datetype, "mindate": mindate, "maxdate": maxdate, "retmax": 0,
        })
        res = r.json()["esearchresult"]
        return int(res["count"]), res["webenv"], res["querykey"]

    def fetch(self, webenv, query_key, total, batch=200):
        for start in range(0, total, batch):
            r = self._get("efetch.fcgi", {
                "db": "pubmed", "query_key": query_key, "WebEnv": webenv,
                "retstart": start, "retmax": batch, "retmode": "xml",
            })
            yield r.content


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------
def _text(el):
    return "".join(el.itertext()).strip() if el is not None else ""


def parse_articles(xml_bytes):
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
        languages = [_text(l) for l in a.findall("Language")]

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
# Filtering and scoring
# ---------------------------------------------------------------------------
def exclusion_reason(rec):
    if rec["retracted"]:
        return "retracted"
    if set(rec["pub_types"]) & EXCLUDE_PUBTYPES:
        return "notice_not_article"
    if not rec["abstract"] or len(rec["abstract"]) < 200:
        return "no_or_short_abstract"
    if "Humans" not in rec["mesh"] and rec["mesh"]:
        # Only reject when MeSH exists; brand-new records may not be indexed yet.
        return "not_human"
    if rec["languages"] and "eng" not in rec["languages"]:
        return "not_english"
    if rec["design"] == "opinion":
        return "opinion_piece"
    return None


def score(rec):
    sf = size_factor(rec["design"], rec["sample_size"])
    rf = recency_factor(rec["year"])
    q = rec["design_score"] * sf * rf
    flags = []
    if rec["expression_of_concern"]:
        q *= 0.5
        flags.append("expression_of_concern")
    if not rec["medline_indexed"]:
        flags.append("not_medline_indexed")  # check journal manually or via DOAJ
    if not rec["mesh"]:
        flags.append("not_yet_mesh_indexed")
    rec["score_components"] = {
        "design": rec["design_score"], "size": round(sf, 3), "recency": round(rf, 3),
    }
    rec["quality_score"] = round(q, 4)
    rec["flags"] = flags
    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_query(topic_query):
    return f"({topic_query}) AND {BASE_FILTERS}"


def run(args):
    email = os.environ.get("NCBI_EMAIL")
    if not email:
        sys.exit("Set NCBI_EMAIL (NCBI requires a contact email).")
    pm = PubMed(email, os.environ.get("NCBI_API_KEY"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.jsonl"
    excluded_path = out_dir / "excluded.jsonl"

    # Load existing corpus so refreshes merge rather than overwrite.
    by_pmid = {}
    if corpus_path.exists():
        with corpus_path.open() as f:
            for line in f:
                rec = json.loads(line)
                by_pmid[rec["pmid"]] = rec
        print(f"Loaded {len(by_pmid)} existing records")

    mindate = args.since or DEFAULT_DATE_RANGE[0]
    maxdate = DEFAULT_DATE_RANGE[1]
    datetype = "edat" if args.since else "pdat"  # entry date for refreshes

    excluded = []
    seen_dois = {r["doi"]: r["pmid"] for r in by_pmid.values() if r.get("doi")}
    topics = {k: v for k, v in TOPICS.items() if not args.topics or k in args.topics}

    for topic, q in topics.items():
        query = build_query(q)
        if args.strong_only:
            query = f"({query}) AND {STRONG_DESIGNS}"
        count, webenv, qk = pm.search(query, mindate, maxdate, datetype)
        total = min(count, args.max_per_topic) if args.max_per_topic else count
        print(f"[{topic}] {count} hits, fetching {total}")

        n_new = 0
        for chunk in pm.fetch(webenv, qk, total):
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
                    rec["pub_types"], rec["mesh"], rec["title"], rec["abstract"])
                rec.update({
                    "topics": [topic],
                    "design": design,
                    "design_score": dscore,
                    "design_method": method,
                    "sample_size": extract_sample_size(rec["abstract"]),
                    "superseded_by": None,       # filled by a later supersession pass
                    "pico": None,                # filled by LLM extraction stage
                    "ingested": date.today().isoformat(),
                })
                reason = exclusion_reason(rec)
                if reason:
                    excluded.append({**rec, "exclusion_reason": reason})
                    continue
                by_pmid[pmid] = score(rec)
                if rec["doi"]:
                    seen_dois[rec["doi"]] = pmid
                n_new += 1
        print(f"[{topic}] {n_new} new records kept")

    # Recompute recency for everything (scores drift as the year changes).
    for rec in by_pmid.values():
        score(rec)

    records = sorted(by_pmid.values(), key=lambda r: r["quality_score"], reverse=True)
    with corpus_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with excluded_path.open("a") as f:
        for rec in excluded:
            f.write(json.dumps({k: rec[k] for k in ("pmid", "doi", "title", "year", "topics",
                                                     "exclusion_reason") if k in rec},
                               ensure_ascii=False) + "\n")

    print_summary(records, excluded)


def print_summary(records, excluded):
    from collections import Counter
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data", help="output directory")
    p.add_argument("--since", help="incremental: only records entered since YYYY/MM/DD")
    p.add_argument("--max-per-topic", type=int, help="cap records per topic (testing)")
    p.add_argument("--topics", nargs="*", choices=list(TOPICS), help="run only these topics")
    p.add_argument("--strong-only", action="store_true",
                   help="restrict to guidelines, reviews, RCTs and cohorts at search time")
    run(p.parse_args())


if __name__ == "__main__":
    main()
