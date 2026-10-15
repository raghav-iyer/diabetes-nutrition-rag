import json

from conftest import RW_SAMPLE

from nrag.ingest.journals import currently_indexed, is_medline
from nrag.ingest.retractions import RetractionIndex, load_retraction_index


def idx():
    return RetractionIndex.from_csv(RW_SAMPLE.read_bytes())


def test_retraction_by_doi_case_insensitive():
    hit = idx().lookup("10.1000/test.0022", None)
    assert hit["retracted"] and not hit["expression_of_concern"]
    assert hit["notices"] == [{"nature": "Retraction", "date": "2025-03-02", "notice_doi": "10.1000/retraction.0022"}]


def test_expression_of_concern_by_doi_and_pmid_merged():
    hit = idx().lookup("10.1000/test.0026", "90000026")
    assert hit["expression_of_concern"] and not hit["retracted"]
    assert len(hit["notices"]) == 1                     # same notice via DOI and PMID counted once


def test_reinstatement_reverses_retraction():
    hit = idx().lookup("10.1000/reinstated", None)
    assert not hit["retracted"]
    assert [n["nature"] for n in hit["notices"]] == ["Retraction", "Reinstatement"]


def test_pmid_only_match_and_placeholders_ignored():
    i = idx()
    assert i.lookup(None, "90000004")["retracted"]
    assert "unavailable" not in i.by_doi and "0" not in i.by_pmid


def test_no_match():
    assert idx().lookup("10.1000/clean", "123") == {"retracted": False, "expression_of_concern": False, "notices": []}


def test_load_uses_configured_url_and_ttl(cfg):
    class Http:
        def get(self, url, **kw):
            self.args = (url, kw)
            return RW_SAMPLE.read_bytes()

    http = Http()
    assert len(load_retraction_index(http, cfg)) > 0
    assert http.args == (cfg["retractions"]["retraction_watch_csv"], {"ttl_days": cfg["retractions"]["ttl_days"]})


class CatalogHttp:
    def __init__(self, indexed):
        self.indexed = indexed
        self.terms = []

    def post(self, url, data=None, **kw):
        self.terms.append(data["term"])
        assert data["db"] == "nlmcatalog" and data["term"].endswith("AND currentlyindexed[All]")
        import re
        asked = re.findall(r"(\d+)\[nlmid\]", data["term"])
        return json.dumps({"esearchresult": {"idlist": [a for a in asked if a in self.indexed]}}).encode()


def test_currently_indexed_normalises_and_batches(cfg):
    cfg2 = {**cfg, "journals": {**cfg["journals"], "nlmcatalog_batch": 2}}
    http = CatalogHttp({"372516", "1001"})
    found = currently_indexed(http, cfg2, ["0372516", "0001001", "9999999", None, "0372516"], {"tool": "t"})
    assert found == {"372516", "1001"}
    assert len(http.terms) == 2                          # 3 unique ids, batch of 2


def test_is_medline_citation_or_journal():
    indexed = {"1001"}
    assert is_medline({"citation_status": "MEDLINE", "nlm_unique_id": "9999999"}, indexed)
    assert is_medline({"citation_status": "In-Process", "nlm_unique_id": "0001001"}, indexed)   # fixes legacy flag
    assert not is_medline({"citation_status": "Publisher", "nlm_unique_id": "9999999"}, indexed)
    assert not is_medline({"citation_status": "Publisher", "nlm_unique_id": None}, indexed)
