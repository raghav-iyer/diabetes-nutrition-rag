import json
from datetime import date

import pytest

from nrag.ingest.pubmed import PubMedClient, allocate, year_slices


def test_year_slices_newest_first_and_clamped():
    s = year_slices("2013/01/01", "3000/12/31", date(2015, 6, 30))
    assert s == [(date(2015, 1, 1), date(2015, 6, 30)), (date(2014, 1, 1), date(2014, 12, 31)),
                 (date(2013, 1, 1), date(2013, 12, 31))]
    assert year_slices("2026/10/01", "3000/12/31", date(2026, 10, 15)) == [(date(2026, 10, 1), date(2026, 10, 15))]


@pytest.mark.parametrize("limit, counts, expected", [
    (10, [100, 100], [5, 5]),
    (10, [30, 10], [8, 2]),          # 7.5 / 2.5 -> largest remainder breaks the tie toward the first
    (10, [3, 100], [0, 10]),         # strictly proportional: a tiny year can get nothing
    (10, [5, 5, 1], [5, 4, 1]),      # remainders go to the largest fractional parts
    (500, [3, 4], [3, 4]),           # limit above total: take everything
    (7, [0, 5, 5], [0, 4, 3]),
])
def test_allocate(limit, counts, expected):
    out = allocate(limit, counts)
    assert out == expected
    assert sum(out) == min(limit, sum(counts))
    assert all(o <= c for o, c in zip(out, counts))


class DatedEsearch:
    """Fake esearch over synthetic PMIDs with known publication dates."""

    def __init__(self, docs: dict[str, date]):
        self.docs = docs
        self.calls = []

    def get(self, url, params=None, **kw):
        self.calls.append(params)
        lo = date(*map(int, params["mindate"].split("/")))
        hi = date(*map(int, params["maxdate"].split("/")))
        ids = sorted((p for p, d in self.docs.items() if lo <= d <= hi), key=lambda p: self.docs[p], reverse=True)
        return json.dumps({"esearchresult": {"count": str(len(ids)), "idlist": ids[:int(params["retmax"])]}}).encode()


def _client(cfg, docs, cap=None):
    ncbi = dict(cfg["ncbi"])
    if cap:
        ncbi["esearch_max_ids"] = cap
    http = DatedEsearch(docs)
    return PubMedClient(http, ncbi, cfg["cache"], email=None), http


def _docs():
    docs = {}
    for y, n in ((2013, 6), (2014, 2), (2015, 12)):
        for i in range(n):
            docs[f"{y}{i:03d}"] = date(y, 1 + i % 12, 1 + i % 28)
    return docs


def test_full_search_collects_everything_across_years(cfg):
    client, http = _client(cfg, _docs())
    total, ids = client.search("q", "2013/01/01", "3000/12/31", today=date(2015, 12, 31))
    assert total == 20 and len(ids) == 20 and len(set(ids)) == 20
    assert all(c["sort"] == "pub_date" for c in http.calls)
    assert "email" not in http.calls[0] and http.calls[0]["tool"] == "nutrition_rag_ingest"


def test_slices_over_cap_are_split_until_they_fit(cfg):
    client, http = _client(cfg, _docs(), cap=5)
    total, ids = client.search("q", "2013/01/01", "3000/12/31", today=date(2015, 12, 31))
    assert total == 20 and sorted(ids) == sorted(_docs())
    assert max(int(c["retmax"]) for c in http.calls) <= 5


def test_limit_samples_proportionally_across_years(cfg):
    client, _ = _client(cfg, _docs())
    total, ids = client.search("q", "2013/01/01", "3000/12/31", limit=10, today=date(2015, 12, 31))
    by_year = {y: sum(1 for i in ids if i.startswith(y)) for y in ("2013", "2014", "2015")}
    assert total == 20 and len(ids) == 10
    assert by_year == {"2013": 3, "2014": 1, "2015": 6}


def test_double_dated_papers_are_deduplicated(cfg):
    # A paper matching two year slices (epub vs print date) must appear once.
    class Overlap(DatedEsearch):
        def get(self, url, params=None, **kw):
            out = json.loads(super().get(url, params))
            if params["mindate"].startswith("2014") and int(params["retmax"]):
                out["esearchresult"]["idlist"].append("2013000")
                out["esearchresult"]["count"] = str(int(out["esearchresult"]["count"]) + 1)
            return json.dumps(out).encode()

    http = Overlap(_docs())
    client = PubMedClient(http, cfg["ncbi"], cfg["cache"])
    total, ids = client.search("q", "2013/01/01", "3000/12/31", today=date(2015, 12, 31))
    assert ids.count("2013000") == 1
    assert total == 20                       # from the single full-range count, not the slice sum
