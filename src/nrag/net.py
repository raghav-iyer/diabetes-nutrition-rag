"""Rate-limited, retrying HTTP client with an on-disk response cache.

Every network call in the project goes through `CachedHttp`, so re-runs are
cheap and reproducible. Cache keys ignore identity/secret params (api_key,
email, tool, mailto) so adding a key doesn't invalidate the cache, and those
values never end up in cache files or logs.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import requests

UNCACHED_PARAMS = {"api_key", "email", "tool", "mailto"}


def cache_key(method: str, url: str, params: dict[str, Any] | None, data: dict[str, Any] | None) -> str:
    def clean(d):
        return sorted((k, str(v)) for k, v in (d or {}).items() if k not in UNCACHED_PARAMS)

    blob = json.dumps([method.upper(), url, clean(params), clean(data)])
    return hashlib.sha256(blob.encode()).hexdigest()


class CachedHttp:
    def __init__(
        self,
        cache_dir: str | Path,
        delay_s: float = 0.34,
        timeout_s: float = 60,
        retries: int = 4,
        session: Any = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cache_dir = Path(cache_dir)
        self.delay_s = delay_s
        self.timeout_s = timeout_s
        self.retries = retries
        self.session = session or requests.Session()
        self._clock = clock
        self._sleep = sleep
        self._last = 0.0
        self.stats = {"hits": 0, "misses": 0}

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.bin"

    def _cached(self, key: str, ttl_days: float | None) -> bytes | None:
        p = self._path(key)
        if not p.exists():
            return None
        if ttl_days is not None and self._clock() - p.stat().st_mtime > ttl_days * 86400:
            return None
        return p.read_bytes()

    def _store(self, key: str, content: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(content)
        tmp.replace(p)

    def request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        ttl_days: float | None = None,
        use_cache: bool = True,
    ) -> bytes:
        key = cache_key(method, url, params, data)
        if use_cache:
            hit = self._cached(key, ttl_days)
            if hit is not None:
                self.stats["hits"] += 1
                return hit
        self.stats["misses"] += 1
        content = self._send(method, url, params, data)
        self._store(key, content)
        return content

    def get(self, url, params=None, **kw) -> bytes:
        return self.request("GET", url, params=params, **kw)

    def post(self, url, data=None, params=None, **kw) -> bytes:
        return self.request("POST", url, params=params, data=data, **kw)

    def _send(self, method, url, params, data) -> bytes:
        for attempt in range(self.retries):
            wait = self.delay_s - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = self._clock()
            try:
                r = self.session.request(method, url, params=params, data=data, timeout=self.timeout_s)
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                return r.content
            except requests.RequestException as e:
                if attempt == self.retries - 1:
                    raise
                backoff = 2**attempt
                # Only the exception text is printed; URLs with secrets are not.
                print(f"  retry {attempt + 1} after error: {type(e).__name__}: {_redact(str(e))} "
                      f"(sleep {backoff}s)", file=sys.stderr)
                self._sleep(backoff)
        raise RuntimeError("unreachable")


def _redact(text: str) -> str:
    return re.sub(r"(api_key|email|mailto)=[^&\s]+", r"\1=REDACTED", text)
