from conftest import PUBMED_BASE

from nrag.ingest.pubmed import parse_articles


def records():
    return {r["pmid"]: r for r in parse_articles(PUBMED_BASE.read_bytes())}


def test_parses_all_articles():
    assert len(records()) == 17


def test_structured_abstract_keeps_labels():
    r = records()["90000001"]
    assert r["abstract"].startswith("BACKGROUND: South Asian adults")
    assert "\nMETHODS: We randomised n = 240" in r["abstract"]


def test_ids_and_doi_lowercased():
    r = records()["90000001"]
    assert r["doi"] == "10.1000/test.0001"
    assert r["pmcid"] == "PMC9000001"
    assert r["pubmed_url"] == "https://pubmed.ncbi.nlm.nih.gov/90000001/"


def test_doi_falls_back_to_elocation():
    assert records()["90000013"]["doi"] == "10.1000/test.0013"


def test_year_fallbacks():
    recs = records()
    assert recs["90000001"]["year"] == 2020           # PubDate/Year
    assert recs["90000003"]["year"] == 2015           # MedlineDate "2015 Jan-Feb"
    assert recs["90000013"]["year"] == 2026           # ArticleDate only


def test_mesh_pubtypes_and_major_topics():
    r = records()["90000001"]
    assert r["pub_types"] == ["Journal Article", "Randomized Controlled Trial"]
    assert r["mesh"] == ["Humans", "Prediabetic State", "Diet"]
    assert r["mesh_major"] == ["Prediabetic State"]


def test_retraction_and_correction_flags():
    recs = records()
    assert recs["90000007"]["retracted"]              # Retracted Publication pub type
    assert recs["90000008"]["retracted"]              # RetractionIn link
    assert recs["90000009"]["expression_of_concern"]
    assert recs["90000009"]["has_erratum"]
    assert not recs["90000001"]["retracted"]


def test_medline_status_language_coi_funders():
    recs = records()
    assert recs["90000001"]["medline_indexed"]
    assert not recs["90000003"]["medline_indexed"]    # PubMed-not-MEDLINE
    assert recs["90000010"]["languages"] == ["ger"]
    assert recs["90000001"]["coi_statement"].startswith("The authors declare")
    assert recs["90000001"]["funders"] == ["Indian Council of Medical Research"]
    assert recs["90000002"]["coi_statement"] is None
