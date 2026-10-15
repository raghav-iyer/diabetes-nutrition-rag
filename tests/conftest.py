import importlib.util
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from nrag.config import load_config
from nrag.ingest.pubmed import Rules

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
PUBMED_BASE = FIXTURES / "pubmed" / "pubmed_base.xml"
PUBMED_REFRESH = FIXTURES / "pubmed" / "pubmed_refresh.xml"
PMC_ARTICLES = FIXTURES / "pmc" / "pmc_articles.xml"
RW_SAMPLE = FIXTURES / "retractions" / "retraction_watch_sample.csv"


@pytest.fixture(scope="session")
def cfg():
    return load_config()  # honours NRAG_CONFIG


@pytest.fixture(scope="session")
def rules(cfg):
    return Rules.from_config(cfg)


@pytest.fixture(scope="session")
def legacy():
    spec = importlib.util.spec_from_file_location("legacy_ingest", ROOT / "legacy" / "ingest_pubmed.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def split_articles(xml_path: Path) -> dict[str, str]:
    """Map PMID -> serialized <PubmedArticle> element."""
    root = ET.parse(xml_path).getroot()
    return {art.findtext("MedlineCitation/PMID"): ET.tostring(art, encoding="unicode")
            for art in root.iter("PubmedArticle")}


def as_xml(articles: list[str]) -> bytes:
    return ("<PubmedArticleSet>" + "".join(articles) + "</PubmedArticleSet>").encode()


def pmids_in(xml_path: Path) -> list[str]:
    return re.findall(r'<PMID Version="1">(\d+)</PMID>', xml_path.read_text())


def _year_of(article_xml: str) -> int:
    m = re.search(r"<PubDate><Year>(\d{4})</Year>", article_xml) or re.search(r"<MedlineDate>(\d{4})", article_xml) \
        or re.search(r'<ArticleDate[^>]*><Year>(\d{4})</Year>', article_xml)
    return int(m.group(1))


class FakeEutils:
    """Offline stand-in for CachedHttp, routing NCBI and Retraction Watch calls to fixtures.

    pubmed esearch: `results` maps query string -> ordered PMIDs; date slices filter by
    the fixture's publication year; efetch serves the fixture XML for the requested IDs.
    """

    def __init__(self, results: dict[str, list[str]], xml_files=(PUBMED_BASE,), indexed_nlm=("1001",),
                 rw_csv: Path = RW_SAMPLE, pmc_xml: Path = PMC_ARTICLES):
        self.articles = {}
        for f in xml_files:
            self.articles.update(split_articles(f))
        self.results = results
        self.indexed = set(indexed_nlm)
        self.rw = rw_csv.read_bytes()
        root = ET.parse(pmc_xml).getroot()
        self.pmc = {a.findtext(".//article-id[@pub-id-type='pmcid']"): ET.tostring(a, encoding="unicode")
                    for a in root.iter("article")}
        self.calls = []
        self.stats = {"hits": 0, "misses": 0}

    def get(self, url, params=None, **kw):
        self.calls.append(("GET", url, dict(params or {})))
        if url.endswith(".csv"):
            return self.rw
        if url.endswith("esearch.fcgi") and params["db"] == "pubmed":
            return self._esearch_pubmed(params)
        raise AssertionError(f"unexpected GET {url} {params}")

    def post(self, url, data=None, params=None, **kw):
        self.calls.append(("POST", url, dict(data or {})))
        if url.endswith("efetch.fcgi") and data["db"] == "pubmed":
            return as_xml([self.articles[i] for i in data["id"].split(",") if i in self.articles])
        if url.endswith("efetch.fcgi") and data["db"] == "pmc":
            arts = [self.pmc[i] for i in data["id"].split(",") if i in self.pmc]
            return ("<pmc-articleset>" + "".join(arts) + "</pmc-articleset>").encode()
        if url.endswith("esearch.fcgi") and data["db"] == "nlmcatalog":
            asked = set(re.findall(r"(\d+)\[nlmid\]", data["term"]))
            ids = sorted(asked & self.indexed)
            return json.dumps({"esearchresult": {"count": str(len(ids)), "idlist": ids}}).encode()
        raise AssertionError(f"unexpected POST {url} {data}")

    def _esearch_pubmed(self, p):
        y0, y1 = int(p["mindate"][:4]), int(p["maxdate"][:4])
        # Publication-date searches filter by the fixture's year; entry-date (refresh)
        # searches treat every configured result as newly entered.
        ids = [i for i in self.results.get(p["term"], [])
               if i in self.articles and (p["datetype"] == "edat" or y0 <= _year_of(self.articles[i]) <= y1)]
        retmax = int(p["retmax"])
        return json.dumps({"esearchresult": {"count": str(len(ids)), "idlist": ids[:retmax]}}).encode()
