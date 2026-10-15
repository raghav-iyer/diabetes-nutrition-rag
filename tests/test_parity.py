"""Behaviour relative to legacy/ingest_pubmed.py.

Phase 0 matched the legacy script exactly. Phase 1 changed some behaviour on
purpose. These tests pin down *exactly* which outputs differ on the fixtures,
and require parity everywhere else, so any further unintended drift fails.
"""

from datetime import date
from itertools import product

from conftest import PUBMED_BASE

from nrag.ingest.pubmed import (
    build_query, classify_design, exclusion_reason, extract_sample_size, parse_articles, recency_factor, score,
    size_factor,
)

RENAMED = {"retracted": "pubmed_retracted", "expression_of_concern": "pubmed_expression_of_concern"}


def test_topic_queries_unchanged(cfg, legacy):
    assert set(cfg["pubmed"]["topics"]) == set(legacy.TOPICS)
    for topic, q in legacy.TOPICS.items():
        assert build_query(cfg, cfg["pubmed"]["topics"][topic]) == legacy.build_query(q)
    assert cfg["pubmed"]["strong_designs"] == legacy.STRONG_DESIGNS
    # Intentional: "post 2012".
    assert cfg["pubmed"]["date_range"][0] == "2013/01/01" != legacy.DEFAULT_DATE_RANGE[0]


def test_parse_parity_on_shared_fields(legacy):
    xml = PUBMED_BASE.read_bytes()
    for new, old in zip(parse_articles(xml), legacy.parse_articles(xml), strict=True):
        for key, value in old.items():
            if key == "medline_indexed":
                assert (new["citation_status"] == "MEDLINE") == value
            else:
                assert new[RENAMED.get(key, key)] == value, key


# PMID -> (legacy design, new design). Every other fixture record must agree.
DESIGN_CHANGES = {
    "90000015": ("observational", "cohort"),             # Observational Study defers to MeSH
    "90000019": ("cohort", "prospective_cohort"),
    "90000020": ("cohort", "prospective_cohort"),
    "90000025": ("meta_analysis", "cross_sectional"),    # background sentence no longer counts
}
SAMPLE_SIZE_CHANGES = {"90000015": (None, 1950)}         # "1,950" was skipped as a year


def test_design_and_sample_size_changes_are_exactly_the_intended_ones(rules, legacy):
    design_diff, n_diff = {}, {}
    for r in parse_articles(PUBMED_BASE.read_bytes()):
        old = legacy.classify_design(r["pub_types"], r["mesh"], r["title"], r["abstract"])[0]
        new = classify_design(rules, r["pub_types"], r["mesh"], r["title"], r["abstract_sections"])[0]
        if old != new:
            design_diff[r["pmid"]] = (old, new)
        old_n, new_n = legacy.extract_sample_size(r["abstract"]), extract_sample_size(rules, r["abstract"])
        if old_n != new_n:
            n_diff[r["pmid"]] = (old_n, new_n)
    assert design_diff == DESIGN_CHANGES
    assert n_diff == SAMPLE_SIZE_CHANGES


def test_hygiene_exclusion_parity_grid(rules, legacy):
    grid = product(
        [False, True],
        [["Journal Article"], ["Published Erratum"], ["Expression of Concern"], ["Retraction of Publication"]],
        [0, 150, 199, 200, 250],
        [[], ["Humans"], ["Animals"], ["Animals", "Humans"]],
        [[], ["eng"], ["ger"], ["eng", "fre"]],
        ["rct", "opinion", "unknown", "narrative_review"],
    )
    for retracted, pts, alen, mesh, langs, design in grid:
        rec = {"retracted": retracted, "pub_types": pts, "abstract": "a" * alen, "mesh": mesh,
               "languages": langs, "design": design}
        assert exclusion_reason(rules, rec) == legacy.exclusion_reason(rec), rec


def test_scoring_parity_grid(rules, legacy):
    designs = ["guideline", "meta_analysis", "systematic_review", "rct", "controlled_trial", "observational",
               "cohort", "case_control", "cross_sectional", "narrative_review", "opinion", "unknown"]
    sizes = [None, 5, 49, 50, 199, 200, 999, 1000, 10**6]
    for design, n in product(designs, sizes):
        assert size_factor(rules.scoring["size"], design, n) == legacy.size_factor(design, n)
    ref = date.today().year
    rc = rules.scoring["recency"]
    for year in (None, 2010, 2018, ref):
        assert recency_factor(year, ref, rc["half_life_years"], rc["floor"]) == legacy.recency_factor(year)
    for design, n, year, eoc, medline, mesh in product(
            designs, sizes[:4], [None, 2010, 2020, ref], [False, True], [False, True], [[], ["Humans"]]):
        base = {"design": design, "design_score": 0.5, "sample_size": n, "year": year,
                "expression_of_concern": eoc, "medline_indexed": medline, "mesh": mesh}
        assert score(rules, dict(base), ref) == legacy.score(dict(base))
