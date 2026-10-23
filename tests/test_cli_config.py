import json

import pytest

from typer.testing import CliRunner

from nrag.cli import app
from nrag.config import load_config
from nrag.ingest.pubmed import PubMedClient

runner = CliRunner()

COMMANDS = ["ingest", "guidelines", "extract", "extract-accuracy", "score", "index", "ask", "eval", "ablate",
            "failures", "compare"]


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in COMMANDS:
        assert cmd in result.output


def test_not_yet_helper_exits_with_code_2():
    import typer
    from nrag.cli import _not_yet
    with pytest.raises(typer.Exit) as e:
        _not_yet(9)
    assert e.value.exit_code == 2


def test_ingest_rejects_unknown_topic():
    result = runner.invoke(app, ["ingest", "--topic", "astrology"])
    assert result.exit_code != 0


def test_config_has_four_topics_and_sane_weights():
    cfg = load_config()
    assert set(cfg["pubmed"]["topics"]) == {
        "diabetes_prevention", "weight_management", "maternal_nutrition", "gestational_diabetes"}
    for rule in cfg["design"]["pubtype_rules"] + cfg["design"]["mesh_rules"] + cfg["design"]["text_rules"]:
        assert 0 < rule["score"] <= 1


class _RecordingHttp:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params=None, **kw):
        self.calls.append(("GET", url, params, kw))
        return self.payload

    def post(self, url, data=None, **kw):
        self.calls.append(("POST", url, data, kw))
        return b"<PubmedArticleSet/>"


def test_pubmed_client_identity_and_batches():
    cfg = load_config()
    http = _RecordingHttp(json.dumps({"esearchresult": {"count": "450", "idlist": [str(i) for i in range(450)]}}).encode())
    client = PubMedClient(http, cfg["ncbi"], cfg["cache"], email="me@example.com", api_key="K")
    from datetime import date
    total, ids = client.search("q", "2026/01/01", "3000/12/31", limit=450, today=date(2026, 10, 15))
    assert total == 450 and len(ids) == 450
    _, url, params, kw = http.calls[0]
    assert url.endswith("/esearch.fcgi")
    assert params["tool"] == "nutrition_rag_ingest" and params["email"] == "me@example.com"
    assert params["api_key"] == "K" and params["sort"] == "pub_date"
    assert kw["ttl_days"] == cfg["cache"]["esearch_ttl_days"]

    list(client.fetch(ids))
    posts = [c for c in http.calls if c[0] == "POST"]
    assert len(posts) == 3                                # 200 + 200 + 50
    assert posts[2][2]["id"].count(",") == 49
    assert posts[0][3]["ttl_days"] == cfg["cache"]["efetch_ttl_days"]


def test_pubmed_client_without_email_or_key():
    cfg = load_config()
    http = _RecordingHttp(json.dumps({"esearchresult": {"count": "0", "idlist": []}}).encode())
    client = PubMedClient(http, cfg["ncbi"], cfg["cache"])
    from datetime import date
    client.search("q", "2026/01/01", "3000/12/31", today=date(2026, 10, 15))
    params = http.calls[0][2]
    assert params["tool"] == "nutrition_rag_ingest"
    assert "email" not in params and "api_key" not in params


def test_guidelines_command_lists_registry():
    result = runner.invoke(app, ["guidelines"])
    assert result.exit_code == 0
    assert "icmr_nin_dgi_2024" in result.output and "rssdi_cpr_t2dm_2022" in result.output
