import json

from typer.testing import CliRunner

from nrag.cli import app
from nrag.config import load_config
from nrag.ingest.pubmed import PubMedClient

runner = CliRunner()

COMMANDS = ["ingest", "extract", "extract-accuracy", "score", "index", "ask", "eval", "ablate",
            "failures", "compare"]


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in COMMANDS:
        assert cmd in result.output


def test_unimplemented_commands_exit_cleanly():
    result = runner.invoke(app, ["score"])
    assert result.exit_code == 2


def test_ingest_requires_email(monkeypatch, tmp_path):
    monkeypatch.delenv("NCBI_EMAIL", raising=False)
    monkeypatch.setattr("nrag.config.load_dotenv", lambda *a, **k: None)
    result = runner.invoke(app, ["ingest", "--max-per-topic", "1"])
    assert result.exit_code != 0
    assert "NCBI_EMAIL" in str(result.exception) + result.output


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


def test_pubmed_client_sends_identity_and_batches():
    cfg = load_config()
    http = _RecordingHttp(json.dumps({"esearchresult": {"count": "450", "idlist": [str(i) for i in range(450)]}}).encode())
    client = PubMedClient(http, cfg["ncbi"], cfg["cache"], "me@example.com", api_key="K")
    count, ids = client.search("q", "2010/01/01", "3000/12/31", limit=450)
    assert count == 450 and len(ids) == 450
    _, url, params, kw = http.calls[0]
    assert url.endswith("/esearch.fcgi")
    assert params["tool"] == "nutrition_rag_ingest" and params["email"] == "me@example.com"
    assert params["api_key"] == "K" and params["retmax"] == 450
    assert kw["ttl_days"] == cfg["cache"]["esearch_ttl_days"]

    list(client.fetch(ids))
    posts = [c for c in http.calls if c[0] == "POST"]
    assert len(posts) == 3                                # 200 + 200 + 50
    assert posts[2][2]["id"].count(",") == 49


def test_pubmed_client_caps_retmax():
    cfg = load_config()
    http = _RecordingHttp(json.dumps({"esearchresult": {"count": "25000", "idlist": []}}).encode())
    client = PubMedClient(http, cfg["ncbi"], cfg["cache"], "me@example.com")
    client.search("q", "2010/01/01", "3000/12/31")
    assert http.calls[0][2]["retmax"] == cfg["ncbi"]["esearch_max_ids"]
    assert "api_key" not in http.calls[0][2]
