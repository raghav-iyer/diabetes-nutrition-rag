"""End-to-end Phase 1 pipeline on fixtures (no network)."""

import json
from datetime import date

import pytest
import yaml
from conftest import PUBMED_BASE, PUBMED_REFRESH, FakeEutils, pmids_in
from test_guidelines import make_pdf

from nrag.ingest import pipeline
from nrag.ingest.pubmed import PubMedClient, build_query

TODAY = date(2026, 10, 15)

EXPECTED_STATUS = {
    "90000001": ("included", None),
    "90000002": ("included", None),
    "90000003": ("provisional", "design_from_text_only"),
    "90000004": ("excluded", "retracted"),               # Retraction Watch, matched by PMID
    "90000005": ("excluded", "design_not_credible"),     # review of a (non-prospective) cohort
    "90000006": ("excluded", "opinion_piece"),
    "90000007": ("excluded", "retracted"),
    "90000008": ("excluded", "retracted"),
    "90000009": ("excluded", "expression_of_concern"),
    "90000010": ("excluded", "not_english"),
    "90000011": ("excluded", "no_or_short_abstract"),
    "90000012": ("excluded", "not_human"),
    "90000013": ("excluded", "design_not_credible"),     # cross-sectional
    "90000014": ("excluded", "duplicate_doi"),
    "90000015": ("excluded", "design_not_credible"),     # Observational Study -> cohort via MeSH
    "90000016": ("excluded", "notice_not_article"),
    "90000019": ("included", None),                      # prospective cohort, n = 5,000
    "90000020": ("excluded", "below_min_sample_size"),   # prospective cohort, n = 300
    "90000021": ("excluded", "excluded_pub_type"),       # trial protocol
    "90000022": ("excluded", "retracted"),               # Retraction Watch, matched by DOI
    "90000023": ("excluded", "not_medline_journal"),
    "90000024": ("included", None),                      # In-Process citation in an indexed journal
    "90000025": ("excluded", "design_not_credible"),     # "previous meta-analyses" no longer fools it
    "90000026": ("excluded", "expression_of_concern"),   # Retraction Watch EoC
}


@pytest.fixture()
def env(cfg, tmp_path):
    pdf = tmp_path / "guidelines" / "test_ada.pdf"
    pdf.parent.mkdir()
    make_pdf(pdf)
    registry = tmp_path / "guidelines" / "registry.yaml"
    registry.write_text(yaml.safe_dump({"guidelines": [
        {"id": "test_ada_2026", "title": "Test Standards 2026", "issuing_body": "ADA", "year": 2026,
         "topics": ["diabetes_prevention"], "population": ["prediabetes"], "file": str(pdf),
         "supersedes": "test_ada_2025", "grade_scheme": "ada"},
        {"id": "test_ada_2025", "title": "Test Standards 2025", "issuing_body": "ADA", "year": 2025,
         "topics": ["diabetes_prevention"], "population": ["prediabetes"], "file": str(pdf), "status": "superseded"},
        {"id": "awaiting", "title": "No PDF yet", "issuing_body": "WHO", "year": 2016,
         "topics": ["maternal_nutrition"], "population": ["pregnant"], "file": str(tmp_path / "missing.pdf")},
    ]}))
    paths = {k: str(tmp_path / "out" / v.split("/", 1)[1]) for k, v in cfg["paths"].items()}
    paths["guidelines_registry"] = str(registry)
    return {**cfg, "paths": paths}, tmp_path


def _clients(cfg, results, xml_files=(PUBMED_BASE,)):
    http = FakeEutils(results, xml_files=xml_files)
    pubmed = PubMedClient(http, cfg["ncbi"], cfg["cache"], sort=cfg["pubmed"]["sort"])
    return pipeline.Clients(pubmed=pubmed, ncbi=http, web=http, ncbi_params={"tool": "t"}), http


def _q(cfg, topic):
    return build_query(cfg, cfg["pubmed"]["topics"][topic], design_filter=cfg["pubmed"]["design_filter"])


def _read(path):
    return [json.loads(line) for line in open(path)]


def _first_run(cfg):
    clients, http = _clients(cfg, {_q(cfg, "maternal_nutrition"): pmids_in(PUBMED_BASE),
                                   _q(cfg, "diabetes_prevention"): ["90000001", "90000024"]})
    return pipeline.run(cfg, clients, today=TODAY), http


def test_statuses_match_policy(env):
    cfg, _ = env
    _first_run(cfg)
    docs = {d["doc_id"]: d for d in _read(cfg["paths"]["documents"])}
    excl = {e["doc_id"]: e for e in _read(cfg["paths"]["excluded"])}
    got = {d["pmid"]: (d["status"], d["status_reason"]) for d in docs.values() if d.get("pmid")}
    got.update({e["pmid"]: ("excluded", e["exclusion_reason"]) for e in excl.values()})
    assert got == EXPECTED_STATUS                       # 90000017 (2011) is outside the searched dates
    assert excl["PMID:90000014"]["duplicate_of"] == "PMID:90000001"


def test_document_fields_topics_fulltext_and_guidelines(env):
    cfg, _ = env
    summary, _ = _first_run(cfg)
    docs = {d["doc_id"]: d for d in _read(cfg["paths"]["documents"])}
    d1 = docs["PMID:90000001"]
    assert d1["topics"] == ["diabetes_prevention", "maternal_nutrition"]
    assert d1["licence"] == "CC-BY" and d1["fulltext_sections"] == ["conclusion", "discussion", "introduction", "methods", "results"]
    assert d1["n_tables"] == 2 and d1["source_type"] == "primary"
    ft = json.loads(open(d1["fulltext_path"]).read())
    assert ft["doc_id"] == "PMID:90000001" and len(ft["tables"]) == 2
    d2 = docs["PMID:90000002"]
    assert d2["licence"] == "pmc-covid-temporary" and d2["fulltext_path"] is None and d2["source_type"] == "review"
    assert docs["PMID:90000024"]["medline_indexed"] and "not_yet_mesh_indexed" in docs["PMID:90000024"]["flags"]

    g = docs["GUIDELINE:test_ada_2026"]
    assert g["source_type"] == "guideline" and g["n_recommendations"] == 2 and g["superseded_by"] is None
    assert docs["GUIDELINE:test_ada_2025"]["superseded_by"] == "GUIDELINE:test_ada_2026"
    assert "superseded" in docs["GUIDELINE:test_ada_2025"]["flags"]
    assert summary["guidelines"]["awaiting_pdf"] == ["awaiting"]

    scores = [d["quality_score"] for d in _read(cfg["paths"]["documents"])]
    assert scores == sorted(scores, reverse=True)


def test_summary_counts(env):
    cfg, _ = env
    summary, _ = _first_run(cfg)
    assert summary["documents"] == 7 and summary["excluded"] == 19
    assert summary["by_status"] == {"included": 6, "provisional:design_from_text_only": 1}
    assert summary["by_design"]["prospective_cohort"] == 1 and summary["without_prospective_cohorts"] == 6
    assert summary["retracted_in_store"] == {"pubmed_only": 2, "retraction_watch_only": 2, "both": 0}
    assert summary["fulltext_licences_seen"] == {"CC-BY": 1, "pmc-covid-temporary": 1}
    assert summary["fulltext_stored"] == 1
    assert json.loads(open(cfg["paths"]["summary"]).read()) == summary


def test_rerun_is_deterministic_and_does_not_duplicate_excluded(env):
    cfg, _ = env
    _first_run(cfg)
    docs1, excl1 = _read(cfg["paths"]["documents"]), _read(cfg["paths"]["excluded"])
    _first_run(cfg)
    assert _read(cfg["paths"]["documents"]) == docs1
    assert _read(cfg["paths"]["excluded"]) == excl1
    assert not (env[1] / "out" / "corpus" / "removals_log.jsonl").exists()


def test_refresh_detects_later_retraction_and_logs_removal(env):
    cfg, _ = env
    _first_run(cfg)
    clients, http = _clients(cfg, {_q(cfg, "diabetes_prevention"): pmids_in(PUBMED_REFRESH)},
                             xml_files=(PUBMED_BASE, PUBMED_REFRESH))
    summary = pipeline.run(cfg, clients, since="2026/10/01", today=TODAY)

    searches = [c[2] for c in http.calls if c[0] == "GET" and c[2].get("db") == "pubmed"]
    assert all(s["datetype"] == "edat" and s["mindate"] >= "2026/10/01" for s in searches)
    docs = {d["doc_id"] for d in _read(cfg["paths"]["documents"])}
    excl = {e["doc_id"]: e for e in _read(cfg["paths"]["excluded"])}
    assert "PMID:90000001" not in docs and excl["PMID:90000001"]["exclusion_reason"] == "retracted"
    # Its DOI duplicate (PMID 90000014) must not resurface once the original is retracted.
    assert "PMID:90000014" not in docs and excl["PMID:90000014"]["exclusion_reason"] == "retracted"
    assert "PMID:90000018" in docs                                   # new paper from the refresh
    log = _read(cfg["paths"]["removals_log"])
    assert log == [{"doc_id": "PMID:90000001", "date": "2026-10-15", "reason": "retracted"}]
    assert summary["removed_this_run"] == log
    # Every stored record was refetched, not just the refresh hits.
    fetched = {i for c in http.calls if c[0] == "POST" and c[2].get("db") == "pubmed" for i in c[2]["id"].split(",")}
    assert {"90000002", "90000026"} <= fetched


def test_store_keeps_raw_records_with_topics(env):
    cfg, _ = env
    _first_run(cfg)
    store = {r["pmid"]: r for r in _read(cfg["paths"]["pubmed_records"])}
    assert len(store) == 24
    assert store["90000024"]["topics"] == ["diabetes_prevention", "maternal_nutrition"]
    assert store["90000024"]["first_seen"] == "2026-10-15"
    assert "status" not in store["90000024"]                         # derived fields are not stored


def test_offline_switches(env):
    cfg, _ = env
    clients, http = _clients(cfg, {_q(cfg, "maternal_nutrition"): pmids_in(PUBMED_BASE)})
    summary = pipeline.run(cfg, clients, today=TODAY, fulltext=False, guidelines=False)
    assert not any(c[2].get("db") == "pmc" for c in http.calls)
    assert summary["guidelines"] == {"parsed": [], "awaiting_pdf": []}


def _iq(cfg, topic):
    from nrag.ingest.pubmed import india_query
    return india_query(cfg, topic, design_filter=cfg["pubmed"]["design_filter"])


def test_india_boost_adds_tagged_records_full_run(env):
    cfg, _ = env
    clients, _ = _clients(cfg, {_q(cfg, "diabetes_prevention"): ["90000001", "90000002"],
                                _iq(cfg, "diabetes_prevention"): ["90000001", "90000019", "90000024"]})
    summary = pipeline.run(cfg, clients, today=TODAY, fulltext=False, guidelines=False)
    store = {r["pmid"]: r for r in _read(cfg["paths"]["pubmed_records"])}
    assert store["90000001"]["search_sources"] == ["main"]           # main hit; India overlap not double-counted
    assert store["90000019"]["search_sources"] == ["india_boost"]
    assert store["90000019"]["topics"] == ["diabetes_prevention"]
    assert summary["india_boost_only"] == 2                      # 90000019 and 90000024 (both included)
    assert summary["pubmed_hits_by_topic"]["diabetes_prevention:india_boost"] == 3


def test_india_boost_quota_in_sample_runs(env):
    cfg, _ = env
    india = ["90000019", "90000024", "90000020", "90000022", "90000026"]
    clients, _ = _clients(cfg, {_q(cfg, "weight_management"): ["90000001", "90000002", "90000003"],
                                _iq(cfg, "weight_management"): ["90000001"] + india})
    pipeline.run(cfg, clients, today=TODAY, max_per_topic=3, topics=["weight_management"],
                 fulltext=False, guidelines=False)
    store = {r["pmid"]: r for r in _read(cfg["paths"]["pubmed_records"])}
    boost_only = [p for p, r in store.items() if r["search_sources"] == ["india_boost"]]
    assert len(boost_only) == 2                                  # ceil(0.45 x 3) = 2, overlap with main excluded
    assert "90000001" not in boost_only


def test_india_boost_can_be_disabled(env):
    cfg, _ = env
    cfg = {**cfg, "pubmed": {**cfg["pubmed"], "india_boost": {**cfg["pubmed"]["india_boost"], "enabled": False}}}
    clients, http = _clients(cfg, {_q(cfg, "diabetes_prevention"): ["90000001"]})
    summary = pipeline.run(cfg, clients, today=TODAY, fulltext=False, guidelines=False)
    assert not any(":india_boost" in k for k in summary["pubmed_hits_by_topic"])
    assert summary["india_boost_only"] == 0
