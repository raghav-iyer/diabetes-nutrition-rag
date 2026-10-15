import pytest
import requests

from nrag.net import CachedHttp, _redact, cache_key


class FakeResponse:
    def __init__(self, status, content=b"ok"):
        self.status_code = status
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, params=None, data=None, timeout=None):
        self.calls.append((method, url, params, data))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class FakeClock:
    def __init__(self):
        self.t = 1_000_000.0
        self.sleeps = []

    def time(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def _http(tmp_path, responses, **kw):
    clock = FakeClock()
    session = FakeSession(responses)
    http = CachedHttp(tmp_path, session=session, clock=clock.time, sleep=clock.sleep, **kw)
    return http, session, clock


def test_cache_hit_avoids_second_request(tmp_path):
    http, session, _ = _http(tmp_path, [FakeResponse(200, b"payload")])
    assert http.get("https://x/a", params={"q": 1}) == b"payload"
    assert http.get("https://x/a", params={"q": 1}) == b"payload"
    assert len(session.calls) == 1
    assert http.stats == {"hits": 1, "misses": 1}


def test_cache_key_ignores_identity_params():
    a = cache_key("GET", "u", {"q": 1, "api_key": "SECRET", "email": "a@b"}, None)
    b = cache_key("GET", "u", {"q": 1}, None)
    assert a == b
    assert cache_key("POST", "u", None, {"id": "1,2"}) != cache_key("POST", "u", None, {"id": "1,3"})


def test_secret_not_written_to_cache(tmp_path):
    http, _, _ = _http(tmp_path, [FakeResponse(200, b"body")])
    http.get("https://x/a", params={"api_key": "SECRET123"})
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert b"SECRET123" not in f.read_bytes()
            assert "SECRET123" not in f.name


def test_ttl_expiry_refetches(tmp_path):
    http, session, clock = _http(tmp_path, [FakeResponse(200, b"v1"), FakeResponse(200, b"v2")])
    assert http.get("https://x/a", ttl_days=1) == b"v1"
    clock.t += 2 * 86400
    # mtime is real wall-clock time; make the cache file look old relative to the fake clock.
    import os
    for f in tmp_path.rglob("*.bin"):
        os.utime(f, (clock.t - 2 * 86400, clock.t - 2 * 86400))
    assert http.get("https://x/a", ttl_days=1) == b"v2"
    assert len(session.calls) == 2


def test_use_cache_false_bypasses(tmp_path):
    http, session, _ = _http(tmp_path, [FakeResponse(200, b"v1"), FakeResponse(200, b"v2")])
    http.get("https://x/a")
    assert http.get("https://x/a", use_cache=False) == b"v2"


def test_retries_on_5xx_and_429_with_backoff(tmp_path):
    http, session, clock = _http(tmp_path, [FakeResponse(503), FakeResponse(429), FakeResponse(200, b"ok")],
                                 delay_s=0.0)
    assert http.get("https://x/a") == b"ok"
    assert len(session.calls) == 3
    assert clock.sleeps == [1, 2]


def test_gives_up_after_retries(tmp_path):
    http, _, _ = _http(tmp_path, [FakeResponse(500)] * 3, retries=3, delay_s=0.0)
    with pytest.raises(requests.HTTPError):
        http.get("https://x/a")


def test_client_error_not_cached(tmp_path):
    http, _, _ = _http(tmp_path, [FakeResponse(400)], retries=1)
    with pytest.raises(requests.HTTPError):
        http.get("https://x/a")
    assert not list(tmp_path.rglob("*.bin"))


def test_rate_limit_spaces_requests(tmp_path):
    http, _, clock = _http(tmp_path, [FakeResponse(200)] * 3, delay_s=0.34)
    for i in range(3):
        http.get(f"https://x/{i}")
    # First call needs no wait; the next two each wait the full delay (fake clock doesn't advance otherwise).
    assert clock.sleeps == pytest.approx([0.34, 0.34])


def test_redact():
    assert _redact("GET ?api_key=abc&email=x@y.z&db=pubmed") == "GET ?api_key=REDACTED&email=REDACTED&db=pubmed"
