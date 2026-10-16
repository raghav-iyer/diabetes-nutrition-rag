"""Local-first LLM client with schema-constrained JSON output.

Backends:
  ollama         native /api/chat: `format` = JSON schema (constrained decoding),
                 `think` controls reasoning (bool, or "low"/"medium"/"high" for gpt-oss)
  openai_compat  /v1/chat/completions with response_format=json_schema
                 (llama.cpp server, LM Studio, vLLM, Ollama's /v1)

Several endpoints (e.g. the Mac and a GPU laptop on the LAN) can share work;
see `EndpointPool`. Responses are not cached here; callers cache by input hash.
"""

from __future__ import annotations

import json
import queue
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import requests


@dataclass
class LLMResponse:
    content: str
    thinking: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float = 0.0
    endpoint: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, backend: str, timeout_s: float = 600, temperature: float = 0.0, seed: int = 0,
                 num_ctx: int = 8192, keep_alive: str = "30m", retries: int = 3, session: Any = None):
        if backend not in {"ollama", "openai_compat"}:
            raise ValueError(f"unknown LLM backend {backend!r}")
        self.backend = backend
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.seed = seed
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.retries = retries
        self.session = session or requests.Session()

    def chat(self, endpoint: str, model: str, messages: list[dict[str, str]], schema: dict | None = None,
             think: bool | str | None = None, max_tokens: int | None = None) -> LLMResponse:
        if self.backend == "ollama":
            url = endpoint.rstrip("/") + "/api/chat"
            body: dict[str, Any] = {
                "model": model, "messages": messages, "stream": False, "keep_alive": self.keep_alive,
                "options": {"temperature": self.temperature, "seed": self.seed, "num_ctx": self.num_ctx},
            }
            if max_tokens:
                body["options"]["num_predict"] = max_tokens
            if schema is not None:
                body["format"] = schema
            if think is not None:
                body["think"] = think
        else:
            url = endpoint.rstrip("/") + "/v1/chat/completions"
            body = {"model": model, "messages": messages, "temperature": self.temperature, "seed": self.seed}
            if max_tokens:
                body["max_tokens"] = max_tokens
            if schema is not None:
                body["response_format"] = {"type": "json_schema",
                                           "json_schema": {"name": "extraction", "schema": schema, "strict": True}}
            if isinstance(think, str):
                body["reasoning_effort"] = think

        start = time.monotonic()
        data = self._post(url, body)
        latency = time.monotonic() - start
        if self.backend == "ollama":
            msg = data.get("message") or {}
            return LLMResponse(content=msg.get("content", ""), thinking=msg.get("thinking"),
                               prompt_tokens=data.get("prompt_eval_count"), completion_tokens=data.get("eval_count"),
                               latency_s=latency, endpoint=endpoint, raw=data)
        choice = (data.get("choices") or [{}])[0].get("message") or {}
        usage = data.get("usage") or {}
        return LLMResponse(content=choice.get("content") or "", thinking=choice.get("reasoning_content"),
                           prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                           latency_s=latency, endpoint=endpoint, raw=data)

    def _post(self, url: str, body: dict) -> dict:
        for attempt in range(self.retries):
            try:
                r = self.session.post(url, json=body, timeout=self.timeout_s)
                if r.status_code >= 500 or r.status_code == 429:
                    raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code >= 400:
                    raise LLMError(f"HTTP {r.status_code} from {url}: {r.text[:300]}")
                data = r.json()
                # Ollama answers 200 with {"done": false} and no message when the runner crashes
                # (e.g. llama.cpp "decode() failed" on GPU memory pressure). Treat it as retryable.
                if self.backend == "ollama" and not data.get("done", True):
                    raise requests.HTTPError("incomplete response (runner error; check GPU memory / num_ctx)")
                return data
            except (requests.RequestException, json.JSONDecodeError) as e:
                if attempt == self.retries - 1:
                    raise LLMError(f"{url}: {e}") from e
                print(f"  LLM retry {attempt + 1} after {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
                time.sleep(2 ** attempt)
        raise LLMError("unreachable")

    def list_models(self, endpoint: str) -> list[str]:
        if self.backend == "ollama":
            r = self.session.get(endpoint.rstrip("/") + "/api/tags", timeout=10)
            return [m["name"] for m in r.json().get("models", [])]
        r = self.session.get(endpoint.rstrip("/") + "/v1/models", timeout=10)
        return [m["id"] for m in r.json().get("data", [])]


class EndpointPool:
    """Hands out endpoints to worker threads, `per_endpoint` concurrent jobs each."""

    def __init__(self, endpoints: list[str], per_endpoint: int = 1):
        if not endpoints:
            raise ValueError("at least one LLM endpoint is required")
        self._q: queue.Queue[str] = queue.Queue()
        for _ in range(per_endpoint):
            for e in endpoints:
                self._q.put(e)
        self.size = len(endpoints) * per_endpoint

    @contextmanager
    def acquire(self) -> Iterator[str]:
        endpoint = self._q.get()
        try:
            yield endpoint
        finally:
            self._q.put(endpoint)


def client_from_config(llm_cfg: dict[str, Any], provider: str) -> LLMClient:
    p = llm_cfg["providers"][provider]
    return LLMClient(backend=p["backend"], timeout_s=llm_cfg["timeout_s"], temperature=llm_cfg["temperature"],
                     seed=llm_cfg["seed"], num_ctx=llm_cfg["num_ctx"], keep_alive=llm_cfg["keep_alive"],
                     retries=llm_cfg["retries"])
