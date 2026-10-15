"""The ported module must reproduce legacy/ingest_pubmed.py exactly.

Runs both implementations end to end on the same fixture XML (a fake NCBI
client stands in for the network) and compares every output record.
"""

import argparse
import json

import pytest
from conftest import PUBMED_BASE, PUBMED_REFRESH, FakePubMedClient, as_xml, pmids_in, split_articles

from nrag.ingest.pubmed import build_query, classify_design, extract_sample_size, ingest, parse_articles


def test_queries_match_legacy(cfg, legacy):
    assert set(cfg["pubmed"]["topics"]) == set(legacy.TOPICS)
    for topic, q in legacy.TOPICS.items():
        assert build_query(cfg, cfg["pubmed"]["topics"][topic]) == legacy.build_query(q)
    assert cfg["pubmed"]["strong_designs"] == legacy.STRONG_DESIGNS
    assert tuple(cfg["pubmed"]["date_range"]) == legacy.DEFAULT_DATE_RANGE


def test_record_level_parity(rules, legacy):
    xml = PUBMED_BASE.read_bytes()
    new = list(parse_articles(xml))
    old = list(legacy.parse_articles(xml))
    assert new == old
    for r in old:
        args = (r["pub_types"], r["mesh"], r["title"], r["abstract"])
        assert classify_design(rules, *args) == legacy.classify_design(*args)
        assert extract_sample_size(rules, r["abstract"]) == legacy.extract_sample_size(r["abstract"])


def _legacy_fake(articles, topic_ids, legacy):
    by_query = {legacy.build_query(legacy.TOPICS[t]): ids for t, ids in topic_ids.items()}

    class FakeLegacyPubMed:
        def __init__(self, email, api_key=None):
            pass

        def search(self, term, mindate, maxdate, datetype="pdat", retmax=100_000):
            return len(by_query.get(term, [])), term, "1"

        def fetch(self, webenv, query_key, total, batch=200):
            ids = by_query.get(webenv, [])[:total]
            for i in range(0, len(ids), batch):
                yield as_xml([articles[p] for p in ids[i:i + batch]])

    return FakeLegacyPubMed


def _run_both(cfg, legacy, tmp_path, monkeypatch, xml, topic_ids, since=None):
    articles = split_articles(xml)
    monkeypatch.setenv("NCBI_EMAIL", "test@example.com")
    monkeypatch.setattr(legacy, "PubMed", _legacy_fake(articles, topic_ids, legacy))
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    legacy.run(argparse.Namespace(out=str(old_dir), since=since, max_per_topic=None,
                                  topics=None, strong_only=False))

    results = {build_query(cfg, cfg["pubmed"]["topics"][t]): ids for t, ids in topic_ids.items()}
    ingest(cfg, FakePubMedClient(articles, results), new_dir / "corpus.jsonl",
           new_dir / "excluded.jsonl", since=since)
    return old_dir, new_dir


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _assert_same(old_dir, new_dir):
    for name in ("corpus.jsonl", "excluded.jsonl"):
        old, new = _lines(old_dir / name), _lines(new_dir / name)
        assert len(old) == len(new) > 0, name
        assert old == new, name


def test_end_to_end_parity(cfg, legacy, tmp_path, monkeypatch, capsys):
    ids = pmids_in(PUBMED_BASE)
    topic_ids = {"diabetes_prevention": ids[:9], "maternal_nutrition": ids, "gestational_diabetes": ids[5:]}
    old_dir, new_dir = _run_both(cfg, legacy, tmp_path, monkeypatch, PUBMED_BASE, topic_ids)
    _assert_same(old_dir, new_dir)


def test_refresh_parity(cfg, legacy, tmp_path, monkeypatch, capsys):
    ids = pmids_in(PUBMED_BASE)
    old_dir, new_dir = _run_both(cfg, legacy, tmp_path, monkeypatch, PUBMED_BASE, {"diabetes_prevention": ids})
    # Second run over the same output dirs: a retraction appears, plus a new paper.
    _run_both(cfg, legacy, tmp_path, monkeypatch, PUBMED_REFRESH,
              {"diabetes_prevention": pmids_in(PUBMED_REFRESH)}, since="2026/10/01")
    _assert_same(old_dir, new_dir)
    reasons = [e["exclusion_reason"] for e in _lines(new_dir / "excluded.jsonl")]
    assert "retracted_since_ingest" in reasons


@pytest.mark.parametrize("year", [2010, 2018, 2026, None])
def test_recency_parity(cfg, legacy, year):
    from datetime import date

    from nrag.ingest.pubmed import recency_factor
    rc = cfg["scoring"]["recency"]
    assert recency_factor(year, date.today().year, rc["half_life_years"], rc["floor"]) == legacy.recency_factor(year)


def test_exclusion_grid_parity(rules, legacy):
    """Boundary inputs the fixtures don't cover (abstract lengths, MeSH, language, notices)."""
    from itertools import product

    from nrag.ingest.pubmed import exclusion_reason
    grid = product(
        [False, True],                                                     # retracted
        [["Journal Article"], ["Published Erratum"], ["Expression of Concern"], ["Retraction of Publication"]],
        [0, 150, 199, 200, 250],                                           # abstract length
        [[], ["Humans"], ["Animals"], ["Animals", "Humans"]],
        [[], ["eng"], ["ger"], ["eng", "fre"]],
        ["rct", "opinion", "unknown", "narrative_review"],
    )
    n = 0
    for retracted, pts, alen, mesh, langs, design in grid:
        rec = {"retracted": retracted, "pub_types": pts, "abstract": "a" * alen, "mesh": mesh,
               "languages": langs, "design": design}
        assert exclusion_reason(rules, rec) == legacy.exclusion_reason(rec), rec
        n += 1
    assert n == 2 * 4 * 5 * 4 * 4 * 4


def test_scoring_grid_parity(rules, legacy):
    from datetime import date
    from itertools import product

    from nrag.ingest.pubmed import score, size_factor
    designs = ["guideline", "meta_analysis", "systematic_review", "rct", "controlled_trial", "observational",
               "cohort", "case_control", "cross_sectional", "narrative_review", "opinion", "unknown"]
    sizes = [None, 5, 49, 50, 199, 200, 999, 1000, 10**6]
    for design, n in product(designs, sizes):
        assert size_factor(rules.scoring["size"], design, n) == legacy.size_factor(design, n)
    ref = date.today().year
    for design, n, year, eoc, medline, mesh in product(
            designs, sizes[:4], [None, 2010, 2020, ref], [False, True], [False, True], [[], ["Humans"]]):
        base = {"design": design, "design_score": 0.5, "sample_size": n, "year": year,
                "expression_of_concern": eoc, "medline_indexed": medline, "mesh": mesh}
        assert score(rules, dict(base), ref) == legacy.score(dict(base))
