"""Phase 6: cited, evidence-labelled answers from retrieved sources, with safety checks.

question -> safety pre-check -> retrieval (Phase 5) -> structured LLM answer (claims with citations)
-> citation checks (ids must be retrieved; LLM verifies each claim is supported) -> deterministic
caveats (strength cap, non-Indian evidence note) -> safety post-check -> markdown with sources.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nrag.extract.pico import json_payload
from nrag.generate import safety
from nrag.llm import LLMClient, LLMError

PROMPT_DIR = Path(__file__).parent / "prompts"
STRENGTH_ORDER = ["insufficient", "limited", "mixed", "moderate", "strong"]


class Claim(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str
    citations: list[str] = Field(default_factory=list)


class Strength(BaseModel):
    model_config = ConfigDict(extra="ignore")
    label: Literal["strong", "moderate", "limited", "mixed", "insufficient"]
    explanation: str = ""


class AnswerOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    direct_answer: str
    claims: list[Claim] = Field(default_factory=list)
    evidence_strength: Strength
    conflicts: str = ""
    abstain: bool = False
    missing: str = ""


class Support(BaseModel):
    model_config = ConfigDict(extra="ignore")
    claim: int
    supported: Literal["yes", "partial", "no"]
    reason: str = ""


VERIFY_PROMPT = """You check citations. For each numbered claim, decide whether its cited source text supports it:
"yes" (clearly stated), "partial" (related but weaker or narrower), or "no" (not stated or contradicted).
Judge only from the source text shown. Return JSON only: {"checks": [{"claim": <number>, "supported": "<yes|partial|no>", "reason": "<short>"}]}"""


def load_prompt(version: str) -> tuple[str, str]:
    text = (PROMPT_DIR / f"{version}.md").read_text()
    system = text.split("<!-- SYSTEM -->", 1)[1].split("<!-- USER -->", 1)[0].strip()
    return system, text.split("<!-- USER -->", 1)[1].strip()


def pack_sources(results: list[dict[str, Any]], max_sources: int, max_chars: int) -> tuple[str, dict[str, dict]]:
    """Group retrieved chunks by document; returns (prompt text, {source_id: info})."""
    by_doc: dict[str, dict] = {}
    for r in results:
        if r["doc_id"] not in by_doc and len(by_doc) >= max_sources:
            continue                                  # cap on distinct sources
        d = by_doc.setdefault(r["doc_id"], {"doc_id": r["doc_id"], "title": r["title"], "year": r.get("year"),
                                            "design": r.get("design"), "source_type": r.get("source_type"),
                                            "population_tags": r.get("population_tags") or [],
                                            "quality_score": r.get("quality_score"), "url": r.get("url"),
                                            "texts": []})
        parent = r.get("parent_text") or ""
        text = parent if parent and len(parent) <= max_chars else r["text"]
        if text not in d["texts"]:
            d["texts"].append(text)
    blocks = []
    for sid, d in list(by_doc.items())[:max_sources]:
        body = "\n...\n".join(d["texts"])[:max_chars]
        pop = ", ".join(d["population_tags"]) or "not tagged"
        blocks.append(f"[{sid}] {d['title']} ({d['year']}; design: {d['design']}; population: {pop})\n{body}")
        d["body"] = body
    return "\n\n".join(blocks), dict(list(by_doc.items())[:max_sources])


def _cached_chat(client, endpoint, model, messages, cache_dir: Path, max_tokens: int):
    key = hashlib.sha256(json.dumps([model, messages, max_tokens]).encode()).hexdigest()
    path = cache_dir / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["content"], True
    resp = client.chat(endpoint, model, messages, max_tokens=max_tokens)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"content": resp.content}))
    return resp.content, False


def _parse(content: str, model_cls):
    return model_cls.model_validate_json(json_payload(content))


class Answerer:
    def __init__(self, cfg: dict[str, Any], retriever, client: LLMClient, endpoint: str, model: str,
                 cache_dir: Path):
        self.cfg, self.gc, self.sc = cfg, cfg["generation"], cfg["safety"]
        self.retriever, self.client, self.endpoint, self.model, self.cache_dir = retriever, client, endpoint, model, cache_dir

    def _llm(self, messages, max_tokens, model_cls, attempts: int = 3):
        errors = []
        for _ in range(attempts):
            try:
                content, cached = _cached_chat(self.client, self.endpoint, self.model, messages, self.cache_dir, max_tokens)
            except LLMError as e:
                errors.append(str(e)[:200])
                break
            try:
                return _parse(content, model_cls), errors
            except (ValidationError, ValueError) as e:
                errors.append(str(e)[:300])
                messages = messages + [{"role": "assistant", "content": content[:4000]},
                                       {"role": "user", "content": f"Invalid JSON for the template: {str(e)[:300]}. "
                                                                   "Return corrected JSON only."}]
        return None, errors

    def answer(self, question: str) -> dict[str, Any]:
        gc, sc = self.gc, self.sc
        pre = safety.assess_question(question, sc)
        out: dict[str, Any] = {"question": question, "safety": pre.__dict__, "status": None, "sources": {},
                               "answer": None, "verification": [], "errors": []}
        if pre.mode == "supportive_only":
            out.update(status="supportive_only", markdown=safety.supportive_reply(pre, sc["support_disclaimer"]))
            return out
        ret = self.retriever.retrieve(question)
        out["retrieval"] = {"status": ret.status, "analysis": ret.analysis, "retried": ret.retried,
                            "timings": ret.timings}
        pre = safety.assess_question(question, sc, ret.analysis)        # re-check with the extracted population
        out["safety"] = pre.__dict__
        if ret.status == "not_answerable":
            reason = ret.analysis.get("answerable_reason") or "it needs individual medical advice"
            out.update(status="not_answerable",
                       markdown=(f"I can't answer this from research: {reason}. Please ask your doctor, who can "
                                 f"advise for your situation.\n\n_{gc['disclaimer']}_"))
            return out
        if ret.status == "insufficient_evidence" or not ret.results:
            out.update(status="insufficient_evidence",
                       markdown=("I couldn't find research in the corpus that answers this well enough to give an "
                                 f"evidence-based answer.\n\n_{gc['disclaimer']}_"))
            return out

        source_text, sources = pack_sources(ret.results, gc["max_sources"], gc["max_source_chars"])
        out["sources"] = {k: {kk: v for kk, v in d.items() if kk not in ("texts", "body")} for k, d in sources.items()}
        system, user = load_prompt(gc["prompt_version"])
        extra = "\n".join(f"- {i}" for i in pre.instructions)
        messages = [{"role": "system", "content": system.replace("{extra_rules}", extra)},
                    {"role": "user", "content": user.replace("{question}", question).replace("{sources}", source_text)}]
        ans, errors = self._llm(messages, gc["max_tokens"], AnswerOut)
        out["errors"] += errors
        if ans is None:
            out.update(status="generation_failed", markdown="Sorry, I couldn't generate an answer. Please try again.")
            return out
        a = ans.model_dump()

        # Citation check 1: ids must be among the retrieved sources.
        for c in a["claims"]:
            bad = [x for x in c["citations"] if x not in sources]
            c["citations"] = [x for x in c["citations"] if x in sources]
            if bad:
                c.setdefault("flags", []).append(f"unknown_citation:{','.join(bad)}")
        a["claims"] = [c for c in a["claims"] if c["citations"]] if gc["drop_unsupported"] else a["claims"]

        # Citation check 2: does the cited text support the claim?
        if gc["verify_citations"] and a["claims"]:
            listing = "\n\n".join(
                # The verifier must see exactly the text the generator saw, or truncation causes false "no"s.
                f"Claim {i}: {c['text']}\n" + "\n".join(f"Source {x}: {sources[x]['body']}" for x in c["citations"])
                for i, c in enumerate(a["claims"], 1))
            ver, verrs = self._llm([{"role": "system", "content": VERIFY_PROMPT}, {"role": "user", "content": listing}],
                                   1200, _Checks)
            out["errors"] += verrs
            checks = {c.claim: c for c in (ver.checks if ver else [])}
            for i, c in enumerate(a["claims"], 1):
                chk = checks.get(i)
                c["support"] = chk.supported if chk else "unchecked"
                out["verification"].append({"claim": i, "supported": c["support"], "reason": chk.reason if chk else ""})
            if gc["drop_unsupported"]:
                a["claims"] = [c for c in a["claims"] if c["support"] != "no"]

        # Deterministic caveats.
        cited = {x for c in a["claims"] for x in c["citations"]}
        cited_designs = {sources[x]["design"] for x in cited}
        notes = []
        if a["evidence_strength"]["label"] == "strong" and not cited_designs & set(gc["strong_designs"]):
            a["evidence_strength"]["label"] = "moderate"
            notes.append("strength_capped")
        if cited and not any({"indian", "south_asian"} & set(sources[x]["population_tags"]) for x in cited):
            a["india_note"] = gc["india_note"]
        if not a["claims"] and not a["abstain"]:
            a["abstain"] = True
            notes.append("no_supported_claims")
        a = safety.check_answer(a, pre, sc)
        a["notes"] = notes
        out.update(status="abstained" if a["abstain"] else "answered", answer=a,
                   markdown=render(a, sources, pre, gc["disclaimer"]))
        return out


class _Checks(BaseModel):
    model_config = ConfigDict(extra="ignore")
    checks: list[Support] = Field(default_factory=list)


REF_MARKERS = re.compile(r"\s*\[(?:\d+(?:\s*[-–,]\s*\d+)*|(?:PMID|GUIDELINE):[^\]]+)\]")


def clean_claim(text: str) -> str:
    """Drop the source paper's own reference markers ("[21]", "[2, 3]") and inline ids the model
    copied into the claim text; citations are rendered separately."""
    return REF_MARKERS.sub("", text).strip()


def render(a: dict[str, Any], sources: dict[str, dict], pre: safety.SafetyAssessment, disclaimer: str) -> str:
    lines = [a["direct_answer"].strip(), ""]
    if a["claims"]:
        lines.append("**What the evidence shows**")
        for c in a["claims"]:
            cites = " ".join(f"[{x}]" for x in c["citations"])
            partial = " _(partly supported)_" if c.get("support") == "partial" else ""
            lines.append(f"- {clean_claim(c['text'])} {cites}{partial}")
        lines.append("")
    s = a["evidence_strength"]
    lines.append(f"**Strength of evidence: {s['label']}.** {s['explanation']}".strip())
    if a.get("conflicts"):
        lines += ["", f"**Where studies disagree:** {a['conflicts']}"]
    if a.get("abstain") and a.get("missing"):
        lines += ["", f"**What's missing:** {a['missing']}"]
    if a.get("india_note"):
        lines += ["", f"_{a['india_note']}_"]
    for n in pre.notes:
        lines += ["", f"**{n}**"]
    cited = []
    for c in a["claims"]:
        for x in c["citations"]:
            if x not in cited:
                cited.append(x)
    if cited:
        lines += ["", "**Sources**"]
        for x in cited:
            d = sources[x]
            link = f" {d['url']}" if d.get("url") else ""
            lines.append(f"- [{x}] {d['title']} ({d['year']}, {d['design']}){link}")
    lines += ["", f"_{disclaimer}_"]
    return "\n".join(lines)


def answerer_from_config(cfg: dict[str, Any]) -> Answerer:
    from nrag.config import resolve_path
    from nrag.extract import runner
    from nrag.llm import client_from_config
    from nrag.retrieve.pipeline import retriever_from_config

    model = runner.active_model(cfg)
    provider = runner.provider_for(cfg, model)
    return Answerer(cfg, retriever_from_config(cfg), client_from_config(cfg["llm"], provider),
                    runner.endpoints(cfg, provider)[0], model, resolve_path(cfg["generation"]["cache_dir"]))
