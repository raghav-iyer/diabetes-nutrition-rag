"""LLM-as-judge for answer correctness, and its validation against human labels.

The judge compares the system's answer with the reference answer using a fixed rubric
(2 = correct, 1 = partially correct, 0 = incorrect). Judge scores are only reported next to
their agreement with human labels on a validation sample (accuracy and Cohen's kappa).
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nrag.extract.pico import json_payload

RUBRIC = """You grade answers from a nutrition-research assistant against a reference answer written by the
evaluation team. Judge the substance, not the wording:
2 = correct: agrees with the reference on the main point (direction of effect, recommendation or "evidence is
    mixed/limited") with no important errors
1 = partially correct: right direction but misses an important qualification, or adds a minor error
0 = incorrect: contradicts the reference, or answers a different question
Do not reward length or citations. Return JSON only: {"score": <0|1|2>, "reason": "<one sentence>"}"""


class _Verdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    score: int = Field(ge=0, le=2)
    reason: str = ""


def judge(question: str, reference: str, answer: str, chat: Callable[[list[dict]], str],
          cache_dir: Path) -> dict[str, Any]:
    key = hashlib.sha256(json.dumps([RUBRIC, question, reference, answer]).encode()).hexdigest()
    path = cache_dir / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    msgs = [{"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"Question: {question}\n\nReference answer: {reference}\n\nSystem answer: {answer}"}]
    for _ in range(3):
        try:
            v = _Verdict.model_validate_json(json_payload(chat(msgs))).model_dump()
            cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(v))
            return v
        except (ValidationError, ValueError):
            continue
    return {"score": None, "reason": "judge failed"}


VALIDATION_COLUMNS = ["run", "question_id", "question", "reference_answer", "system_answer", "judge_score",
                      "judge_reason", "your_score", "notes"]


def export_validation(path: Path, run_name: str, rows: list[dict[str, Any]], n: int = 50, seed: int = 0) -> int:
    """Sample up to n judged answers, stratified by judge score, for human labelling."""
    judged = [r for r in rows if r.get("judge_score") is not None]
    rng = random.Random(seed)
    by_score: dict[int, list] = {}
    for r in judged:
        by_score.setdefault(r["judge_score"], []).append(r)
    for v in by_score.values():
        rng.shuffle(v)
    sample = []
    while len(sample) < min(n, len(judged)):
        for s in sorted(by_score):
            if by_score[s] and len(sample) < n:
                sample.append(by_score[s].pop())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=VALIDATION_COLUMNS)
        w.writeheader()
        for r in sample:
            w.writerow({"run": run_name, "question_id": r["question_id"], "question": r["question"],
                        "reference_answer": r["reference_answer"], "system_answer": r["system_answer"],
                        "judge_score": r["judge_score"], "judge_reason": r["judge_reason"], "your_score": "", "notes": ""})
    return len(sample)


def agreement(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    pairs = [(int(r["judge_score"]), int(r["your_score"])) for r in rows
             if str(r.get("your_score", "")).strip() in {"0", "1", "2"} and str(r.get("judge_score", "")).strip() in {"0", "1", "2"}]
    if not pairs:
        return None
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n": n, "accuracy": round(po, 4), "cohen_kappa": round(kappa, 4),
            "judge_mean": round(sum(a for a, _ in pairs) / n, 3), "human_mean": round(sum(b for _, b in pairs) / n, 3)}
