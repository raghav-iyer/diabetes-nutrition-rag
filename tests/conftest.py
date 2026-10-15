import importlib.util
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
    out = {}
    for art in root.iter("PubmedArticle"):
        out[art.findtext("MedlineCitation/PMID")] = ET.tostring(art, encoding="unicode")
    return out


def as_xml(articles: list[str]) -> bytes:
    return ("<PubmedArticleSet>" + "".join(articles) + "</PubmedArticleSet>").encode()


class FakePubMedClient:
    """Stands in for PubMedClient: query string -> ordered PMIDs, served from fixture XML."""

    def __init__(self, articles: dict[str, str], results: dict[str, list[str]], batch: int = 200):
        self.articles = articles
        self.results = results
        self.batch = batch
        self.searches = []

    def search(self, term, mindate, maxdate, datetype="pdat", limit=None):
        self.searches.append((term, mindate, maxdate, datetype))
        ids = self.results.get(term, [])
        return len(ids), ids[:limit] if limit else list(ids)

    def fetch(self, pmids):
        for i in range(0, len(pmids), self.batch):
            yield as_xml([self.articles[p] for p in pmids[i:i + self.batch]])


def pmids_in(xml_path: Path) -> list[str]:
    return re.findall(r'<PMID Version="1">(\d+)</PMID>', xml_path.read_text())
