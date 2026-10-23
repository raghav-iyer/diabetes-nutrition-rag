"""Safety layer: rules applied before and after answer generation.

Before generation (`assess_question`):
  - disordered-eating signals  -> no diet advice: a supportive reply pointing to professional help
  - individual medical advice  -> no diagnosis, dosing or medication changes: refuse that part,
                                  still summarise general research if any
  - pregnancy / lactation + a risk topic (supplements, medicines, alcohol, high-mercury fish,
    food safety, GDM management) -> answer with the guideline position, no individual doses or
    calorie targets, and tell the user to confirm with their doctor or midwife
After generation (`check_answer`):
  - individual doses / calorie targets in a restricted answer are removed and flagged
  - the disclaimer is added exactly once
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SafetyAssessment:
    mode: str = "normal"                         # normal | restricted | supportive_only
    categories: list[str] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)   # extra rules for the answer prompt
    notes: list[str] = field(default_factory=list)          # sentences appended to the answer

    @property
    def restricted(self) -> bool:
        return self.mode != "normal"


def _rx(p: str) -> re.Pattern:
    return re.compile(p, re.I)


def assess_question(question: str, sc: dict[str, Any], analysis: dict[str, Any] | None = None) -> SafetyAssessment:
    a = SafetyAssessment()
    pops = set((analysis or {}).get("population_tags") or [])
    if _rx(sc["disordered_eating"]).search(question):
        a.mode = "supportive_only"
        a.categories.append("disordered_eating")
        a.notes.append(sc["support_resources"])
        return a                                  # no diet advice at all
    if _rx(sc["individual_advice"]).search(question):
        a.mode = "restricted"
        a.categories.append("individual_advice")
        a.instructions.append("Do not diagnose, give an individual dose, or suggest starting, stopping or changing "
                              "a medicine. Say that this needs their doctor, then summarise only what research "
                              "shows in general.")
    pregnant = bool(_rx(sc["pregnancy_terms"]).search(question)) or bool(pops & {"pregnant", "lactating",
                                                                                "gestational_diabetes"})
    if pregnant:
        for name, pattern in sc["pregnancy_risk_topics"].items():
            if _rx(pattern).search(question):
                a.categories.append(f"pregnancy_{name}")
        if any(c.startswith("pregnancy_") for c in a.categories):
            a.mode = "restricted"
            a.instructions.append("This is a pregnancy or breastfeeding safety question. State the guideline "
                                  "position first if a guideline source is provided. Do not give individual doses, "
                                  "amounts or calorie targets.")
            a.notes.append(sc["clinician_note"])
    return a


def supportive_reply(a: SafetyAssessment, disclaimer: str) -> str:
    return ("It sounds like food or weight may be weighing on you, and you deserve real support with that. "
            "I can't suggest a diet plan for this, but you don't have to handle it alone.\n\n"
            + "\n\n".join(a.notes) + f"\n\n_{disclaimer}_")


def strip_doses(text: str, dose_pattern: str) -> tuple[str, int]:
    """Remove sentences that state individual doses or calorie targets."""
    rx = _rx(dose_pattern)
    kept, removed = [], 0
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if rx.search(sent):
            removed += 1
        else:
            kept.append(sent)
    return " ".join(kept).strip(), removed


def check_answer(answer: dict[str, Any], a: SafetyAssessment, sc: dict[str, Any]) -> dict[str, Any]:
    """Post-generation check on the structured answer (mutates and returns it)."""
    flags = answer.setdefault("safety_flags", [])
    if a.restricted:
        for key in ("direct_answer",):
            answer[key], n = strip_doses(answer.get(key) or "", sc["dose_pattern"])
            if n:
                flags.append(f"removed_{n}_dose_sentence(s)_from_{key}")
        kept = []
        for claim in answer.get("claims") or []:
            text, n = strip_doses(claim["text"], sc["dose_pattern"])
            if n and not text:
                flags.append("dropped_dose_claim")
                continue
            if n:
                flags.append("removed_dose_from_claim")
            kept.append({**claim, "text": text})
        answer["claims"] = kept
    return answer
