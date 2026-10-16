import json
import typing

import pytest
import requests
from pydantic import ValidationError

from nrag.extract import pico
from nrag.extract.schemas import Extraction, PopulationTag, extraction_json_schema
from nrag.llm import EndpointPool, LLMClient, LLMError, LLMResponse

ABSTRACT = ("BACKGROUND: South Asians are at high risk.\nMETHODS: We randomised 240 adults with prediabetes in Chennai, "
            "India to a low glycaemic index diet or usual care for 12 months (CTRI/2019/01/001).\n"
            "RESULTS: Diabetes incidence was lower with the intervention (HR 0.58).")
DOC = {"doc_id": "PMID:1", "title": "Low GI diet in Indian adults with prediabetes", "abstract": ABSTRACT}


def good_extraction(**kw):
    e = {
        "study_design": "rct",
        "population": {"description": "Adults with prediabetes in Chennai", "tags": ["prediabetes", "indian", "south_asian"]},
        "intervention": "Low glycaemic index diet", "comparison": "Usual care",
        "outcomes": [{"name": "Diabetes incidence", "type": "hard"}],
        "effect_direction": "benefit", "key_finding": "The diet lowered diabetes incidence.",
        "sample_size": 240, "duration_weeks": 52.2, "country_or_region": "India",
        "risk_of_bias_signals": {"randomisation_described": "not_reported", "blinding": "not_reported",
                                 "low_dropout": "not_reported", "pre_registered": "yes"},
        "funding_industry": "unclear",
        "quotes": {"sample_size": "We randomised 240 adults", "key_finding": "Diabetes incidence was lower"},
        "confidence": "high",
    }
    e.update(kw)
    return e


# --- schema -------------------------------------------------------------------

def test_population_vocabulary_matches_config(cfg):
    assert set(typing.get_args(PopulationTag)) == set(cfg["population_tags"])


def test_schema_is_strict():
    Extraction.model_validate(good_extraction())
    with pytest.raises(ValidationError):
        Extraction.model_validate({**good_extraction(), "extra_field": 1})
    with pytest.raises(ValidationError):
        Extraction.model_validate(good_extraction(effect_direction="helpful"))
    with pytest.raises(ValidationError):
        Extraction.model_validate(good_extraction(population={"description": "x", "tags": ["martians"]}))
    schema = extraction_json_schema()
    assert schema["additionalProperties"] is False and "population" in schema["required"]


# --- LLM client -----------------------------------------------------------------

class FakeHTTP:
    def __init__(self, responses):
        self.responses = list(responses)
        self.posts = []

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class R:
    def __init__(self, status, payload):
        self.status_code, self.payload, self.text = status, payload, str(payload)

    def json(self):
        return self.payload


def test_ollama_request_and_response():
    http = FakeHTTP([R(200, {"message": {"content": "{}", "thinking": "hmm"}, "prompt_eval_count": 500, "eval_count": 120})])
    c = LLMClient("ollama", session=http, num_ctx=4096)
    resp = c.chat("http://h:11434/", "gpt-oss:20b", [{"role": "user", "content": "x"}], schema={"type": "object"},
                  think="low", max_tokens=100)
    url, body = http.posts[0]
    assert url == "http://h:11434/api/chat"
    assert body["format"] == {"type": "object"} and body["think"] == "low" and body["stream"] is False
    assert body["options"] == {"temperature": 0.0, "seed": 0, "num_ctx": 4096, "num_predict": 100}
    assert (resp.content, resp.thinking, resp.prompt_tokens, resp.completion_tokens) == ("{}", "hmm", 500, 120)


def test_openai_compat_request_and_response():
    http = FakeHTTP([R(200, {"choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 9, "completion_tokens": 3}})])
    c = LLMClient("openai_compat", session=http)
    resp = c.chat("http://h:8080", "m", [], schema={"type": "object"}, think="low")
    url, body = http.posts[0]
    assert url == "http://h:8080/v1/chat/completions"
    assert body["response_format"]["json_schema"]["schema"] == {"type": "object"} and body["reasoning_effort"] == "low"
    assert (resp.prompt_tokens, resp.completion_tokens) == (9, 3)


def test_client_retries_server_errors_but_not_client_errors(monkeypatch):
    monkeypatch.setattr("nrag.llm.time.sleep", lambda s: None)
    http = FakeHTTP([R(500, {}), requests.ConnectionError("down"), R(200, {"message": {"content": "ok"}})])
    assert LLMClient("ollama", session=http).chat("http://h", "m", []).content == "ok"
    with pytest.raises(LLMError, match="HTTP 404"):
        LLMClient("ollama", session=FakeHTTP([R(404, {"error": "model not found"})])).chat("http://h", "m", [])
    with pytest.raises(LLMError):
        LLMClient("ollama", session=FakeHTTP([R(503, {})] * 3)).chat("http://h", "m", [])


def test_endpoint_pool_round_robin():
    pool = EndpointPool(["a", "b"], per_endpoint=2)
    assert pool.size == 4
    with pool.acquire() as e1, pool.acquire() as e2:
        assert {e1, e2} == {"a", "b"}
    with pytest.raises(ValueError):
        EndpointPool([])


# --- extraction -----------------------------------------------------------------

class ScriptedClient:
    """Returns scripted response contents in order; records calls."""

    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []

    def chat(self, endpoint, model, messages, schema=None, think=None, max_tokens=None):
        self.calls.append({"endpoint": endpoint, "model": model, "messages": messages, "think": think})
        c = self.contents.pop(0)
        if isinstance(c, Exception):
            raise c
        return LLMResponse(content=c, prompt_tokens=100, completion_tokens=50, latency_s=1.5, endpoint=endpoint)


@pytest.fixture()
def ec(cfg, tmp_path):
    return pico.ExtractConfig.from_config(cfg, "gpt-oss:20b", cache_dir=tmp_path / "llm")


def test_prompt_fills_placeholders(ec):
    msgs = pico.build_messages(ec, "T", "A")
    assert msgs[0]["role"] == "system" and "at or below 20%" in msgs[0]["content"]
    assert msgs[1]["content"] == "Title: T\n\nAbstract:\nA"
    assert not any(p in m["content"] for m in msgs for p in ("{title}", "{abstract}", "{dropout_threshold_pct}"))


def test_extract_valid_first_try_then_cached(ec):
    client = ScriptedClient([json.dumps(good_extraction())])
    r = pico.extract_one(client, "http://a", ec, DOC)
    assert r["status"] == "ok" and r["attempts"] == 1 and not r["cached"]
    assert r["quote_checks"] == {"sample_size": True, "key_finding": True}
    assert client.calls[0]["think"] == "low"                  # from config extract.think
    again = pico.extract_one(ScriptedClient([]), "http://a", ec, DOC)
    assert again["cached"] and again["extraction"] == r["extraction"]


def test_invalid_output_is_retried_with_feedback(ec):
    client = ScriptedClient(["not json", json.dumps(good_extraction(effect_direction="helpful")),
                             json.dumps(good_extraction())])
    r = pico.extract_one(client, "http://a", ec, DOC)
    assert r["status"] == "ok" and r["attempts"] == 3 and len(r["errors"]) == 2
    feedback = client.calls[2]["messages"][-1]["content"]
    assert feedback.startswith("That output failed validation") and "effect_direction" in feedback


def test_failure_after_max_attempts_is_not_cached(ec):
    r = pico.extract_one(ScriptedClient(["{}"] * 3), "http://a", ec, DOC)
    assert r["status"] == "failed" and r["attempts"] == 3 and r["extraction"] is None
    assert not list(ec.cache_dir.rglob("*.json"))


def test_llm_error_is_recorded(ec):
    r = pico.extract_one(ScriptedClient([LLMError("connection refused")]), "http://a", ec, DOC)
    assert r["status"] == "failed" and "llm_error" in r["errors"][0]


def test_input_hash_depends_on_model_think_and_text(cfg, ec, tmp_path):
    h = pico.input_hash(ec, "T", "A")
    other_think = pico.ExtractConfig.from_config(cfg, "gpt-oss:20b", think="high", cache_dir=tmp_path)
    other_model = pico.ExtractConfig.from_config(cfg, "qwen3.8:27b", cache_dir=tmp_path)
    assert len({h, pico.input_hash(other_think, "T", "A"), pico.input_hash(other_model, "T", "A"),
                pico.input_hash(ec, "T", "A2")}) == 4
    assert pico.model_slug("qwen3.8:27b", False) == "qwen3-8-27b"
    assert pico.model_slug("gpt-oss:20b", "low") == "gpt-oss-20b__think-low"


@pytest.mark.parametrize("quote, expected", [
    ("We randomised 240 adults", True),
    ("we  RANDOMISED 240 adults,", True),                    # case/spacing/punctuation tolerant
    ("We randomised 2400 adults", False),                    # changed number is not supported
    ("A totally invented sentence about kale", False),
    ("low glycaemic index diet or usual care for 12\u2011months", True),     # non-breaking hyphen vs space
    ("\u201cDiabetes incidence was lower\u201d", True),                       # curly quotes
    ("We randomised 240 adults ... Diabetes incidence was lower", True),     # elided quote
    ("We randomised 240 adults ... incidence tripled", False),               # one elided part invented
    (None, None),
])
def test_quote_supported(quote, expected):
    assert pico.quote_supported(quote, ABSTRACT) == expected


def test_run_extraction_uses_all_endpoints_and_keeps_order(ec):
    docs = [{**DOC, "doc_id": f"PMID:{i}", "abstract": ABSTRACT + f" Variant {i}."} for i in range(6)]

    class Echo(ScriptedClient):
        def __init__(self):
            super().__init__([])

        def chat(self, endpoint, model, messages, **kw):
            self.calls.append(endpoint)
            return LLMResponse(content=json.dumps(good_extraction()), latency_s=2.0, endpoint=endpoint)

    client = Echo()
    results = pico.run_extraction(client, EndpointPool(["http://mac", "http://laptop"]), ec, docs, progress=False)
    assert [r["doc_id"] for r in results] == [d["doc_id"] for d in docs]
    assert set(client.calls) == {"http://mac", "http://laptop"}
    s = pico.summarize_run(results)
    assert s["schema_failure_rate"] == 0.0 and s["first_attempt_valid_rate"] == 1.0 and s["latency_s"]["median"] == 2.0


def test_prompt_mode_embeds_schema_and_sends_no_format(cfg, tmp_path):
    ec = pico.ExtractConfig.from_config(cfg, "qwen3.8:27b", cache_dir=tmp_path)
    assert ec.output_mode == "prompt"
    sys_msg = pico.build_messages(ec, "T", "A")[0]["content"]
    assert '"effect_direction"' in sys_msg and "no markdown" in sys_msg
    seen = {}

    class C(ScriptedClient):
        def chat(self, endpoint, model, messages, schema=None, think=None, max_tokens=None):
            seen["schema"] = schema
            return super().chat(endpoint, model, messages, schema, think, max_tokens)

    fenced = "Here you go:\n```json\n" + json.dumps(good_extraction()) + "\n```"
    r = pico.extract_one(C([fenced]), "http://a", ec, DOC)
    assert r["status"] == "ok" and r["output_mode"] == "prompt" and seen["schema"] is None
    # Different output mode -> different cache key than schema mode would have.
    schema_ec = pico.ExtractConfig.from_config(cfg, "qwen3.8:27b", cache_dir=tmp_path)
    schema_ec.output_mode = "schema"
    assert pico.input_hash(ec, "T", "A") != pico.input_hash(schema_ec, "T", "A")


@pytest.mark.parametrize("content, expected", [
    ('{"a": 1}', '{"a": 1}'),
    ('```json\n{"a": 1}\n```', '{"a": 1}'),
    ('Sure! {"a": {"b": 2}} Hope this helps.', '{"a": {"b": 2}}'),
])
def test_json_payload(content, expected):
    assert pico.json_payload(content) == expected


def test_incomplete_ollama_response_is_retried_then_fails(monkeypatch):
    monkeypatch.setattr("nrag.llm.time.sleep", lambda s: None)
    crashed = R(200, {"model": "", "done": False})
    ok = R(200, {"done": True, "message": {"content": "{}"}})
    assert LLMClient("ollama", session=FakeHTTP([crashed, ok])).chat("http://h", "m", []).content == "{}"
    with pytest.raises(LLMError, match="incomplete response"):
        LLMClient("ollama", session=FakeHTTP([crashed] * 3)).chat("http://h", "m", [])
