"""Query understanding: population, topic, intent, answerability and a rewritten query.

The LLM (extract.active_model) fills a JSON template; the result is validated with
Pydantic and cached by (model, prompt, question). The original question is always kept.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nrag.extract.pico import json_payload
from nrag.extract.schemas import PopulationTag
from nrag.llm import LLMClient, LLMError

Topic = Literal["diabetes_prevention", "weight_management", "maternal_nutrition", "gestational_diabetes", "other"]
Intent = Literal["what_works", "is_it_safe", "how_much", "compare", "other"]


class QueryAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    population_tags: list[PopulationTag] = Field(default_factory=list)
    topics: list[Topic] = Field(default_factory=list)
    intent: Intent = "other"
    answerable: bool = True
    answerable_reason: str = ""
    rewritten_query: str
    keywords: list[str] = Field(default_factory=list)


PROMPT = """You analyse questions for a nutrition-research search engine (diabetes prevention, weight management,
maternal nutrition, gestational diabetes; many users are in India).

Fill in this JSON template. Use only the allowed values.
{
 "population_tags": ["<zero or more of: general_adults | prediabetes | type2_diabetes | overweight_obese | pregnant | lactating | gestational_diabetes | south_asian | indian | children | older_adults; only those the question names or clearly implies>"],
 "topics": ["<one or more of: diabetes_prevention | weight_management | maternal_nutrition | gestational_diabetes | other>"],
 "intent": "<one of: what_works | is_it_safe | how_much | compare | other>",
 "answerable": <true if published nutrition research could answer it; false for personal diagnosis, individual dosing, or questions outside nutrition research>,
 "answerable_reason": "<short reason>",
 "rewritten_query": "<the question rewritten as a precise search query using research terms (e.g. 'low glycaemic index diet gestational diabetes incidence randomised trials'); keep the population>",
 "keywords": ["<3-8 key search terms or synonyms>"]
}
Return one JSON object only."""


def _hash(model: str, question: str) -> str:
    return hashlib.sha256(json.dumps([model, PROMPT, question]).encode()).hexdigest()


class QueryUnderstanding:
    def __init__(self, client: LLMClient | None, endpoint: str | None, model: str, cache_dir: Path,
                 max_attempts: int = 3, enabled: bool = True):
        self.client, self.endpoint, self.model = client, endpoint, model
        self.cache_dir, self.max_attempts, self.enabled = cache_dir, max_attempts, enabled

    def fallback(self, question: str) -> QueryAnalysis:
        return QueryAnalysis(rewritten_query=question, answerable_reason="query understanding disabled")

    def analyse(self, question: str) -> tuple[QueryAnalysis, dict[str, Any]]:
        if not self.enabled or self.client is None:
            return self.fallback(question), {"source": "disabled"}
        path = self.cache_dir / f"{_hash(self.model, question)}.json"
        if path.exists():
            return QueryAnalysis.model_validate_json(path.read_text()), {"source": "cache"}
        messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": question}]
        errors = []
        for _ in range(self.max_attempts):
            try:
                resp = self.client.chat(self.endpoint, self.model, messages, max_tokens=600)
            except LLMError as e:
                errors.append(str(e)[:200])
                break
            try:
                qa = QueryAnalysis.model_validate_json(json_payload(resp.content))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(qa.model_dump_json())
                return qa, {"source": "llm", "latency_s": round(resp.latency_s, 2), "attempts": len(errors) + 1}
            except ValidationError as e:
                err = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:5])
                errors.append(err)
                messages = messages + [{"role": "assistant", "content": resp.content[:3000]},
                                       {"role": "user", "content": f"Invalid: {err}. Return corrected JSON only."}]
        # Never block retrieval: fall back to the original question.
        return self.fallback(question), {"source": "fallback", "errors": errors}
