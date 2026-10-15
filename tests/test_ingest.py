import json
from datetime import date

from conftest import PUBMED_BASE, PUBMED_REFRESH, FakePubMedClient, pmids_in, split_articles

from nrag.ingest.pubmed import build_query, ingest

TODAY = date(2026, 10, 15)


def _client(cfg, topic_ids, xml=PUBMED_BASE):
    topics = cfg["pubmed"]["topics"]
    results = {build_query(cfg, topics[t]): ids for t, ids in topic_ids.items()}
    return FakePubMedClient(split_articles(xml), results)


def _run(cfg, tmp_path, client, **kw):
    return ingest(cfg, client, tmp_path / "corpus.jsonl", tmp_path / "excluded.jsonl", today=TODAY, **kw)


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_full_run_keeps_excludes_and_dedups(cfg, tmp_path):
    ids = pmids_in(PUBMED_BASE)
    records, excluded = _run(cfg, tmp_path, _client(cfg, {"maternal_nutrition": ids}))

    kept = {r["pmid"] for r in records}
    assert kept == {"90000001", "90000002", "90000003", "90000004", "90000005",
                    "90000009", "90000013", "90000015", "90000017"}
    reasons = {e["pmid"]: e["exclusion_reason"] for e in excluded}
    assert reasons == {
        "90000006": "opinion_piece", "90000007": "retracted", "90000008": "retracted",
        "90000010": "not_english", "90000011": "no_or_short_abstract", "90000012": "not_human",
        "90000016": "notice_not_article",
    }
    # 90000014 shares a DOI with 90000001: silently merged, neither kept nor excluded.
    assert "90000014" not in kept and "90000014" not in reasons


def test_outputs_sorted_and_excluded_slimmed(cfg, tmp_path):
    _run(cfg, tmp_path, _client(cfg, {"maternal_nutrition": pmids_in(PUBMED_BASE)}))
    corpus = _read(tmp_path / "corpus.jsonl")
    scores = [r["quality_score"] for r in corpus]
    assert scores == sorted(scores, reverse=True)
    assert corpus[0]["pmid"] == "90000002"            # meta-analysis ranks first
    assert all(r["ingested"] == "2026-10-15" for r in corpus)
    excluded = _read(tmp_path / "excluded.jsonl")
    assert set(excluded[0]) <= {"pmid", "doi", "title", "year", "topics", "exclusion_reason"}


def test_topic_tags_merge_across_topics(cfg, tmp_path):
    client = _client(cfg, {"diabetes_prevention": ["90000001", "90000005"],
                           "gestational_diabetes": ["90000002", "90000014", "90000005"]})
    records, _ = _run(cfg, tmp_path, client)
    by = {r["pmid"]: r for r in records}
    assert by["90000005"]["topics"] == ["diabetes_prevention", "gestational_diabetes"]
    # Duplicate DOI under a different PMID adds the topic to the original.
    assert by["90000001"]["topics"] == ["diabetes_prevention", "gestational_diabetes"]


def test_max_per_topic_and_topic_selection(cfg, tmp_path):
    client = _client(cfg, {"maternal_nutrition": pmids_in(PUBMED_BASE),
                           "weight_management": ["90000009"]})
    records, excluded = _run(cfg, tmp_path, client, max_per_topic=3, topics=["maternal_nutrition"])
    assert {r["pmid"] for r in records} == {"90000001", "90000002", "90000003"}
    assert len(client.searches) == 1


def test_incremental_refresh_merges_and_drops_retracted(cfg, tmp_path):
    _run(cfg, tmp_path, _client(cfg, {"diabetes_prevention": ["90000001", "90000002"]}))
    client = _client(cfg, {"diabetes_prevention": pmids_in(PUBMED_REFRESH)}, xml=PUBMED_REFRESH)
    records, excluded = _run(cfg, tmp_path, client, since="2026/10/01")

    assert client.searches[0][1:] == ("2026/10/01", "3000/12/31", "edat")
    assert {r["pmid"] for r in records} == {"90000002", "90000018"}
    assert excluded == [e for e in excluded if e["pmid"] == "90000001"]
    assert excluded[0]["exclusion_reason"] == "retracted_since_ingest"


def test_strong_only_changes_query(cfg):
    q = build_query(cfg, "x[tiab]", strong_only=True)
    assert q.startswith("((x[tiab]) AND humans[mh]")
    assert q.endswith("cohort studies[mh])")
