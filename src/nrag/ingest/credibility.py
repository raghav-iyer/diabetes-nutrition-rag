"""The "extremely credible research" policy (config `credibility`).

`assess` returns (status, reason):
  ("included", None)       meets every rule
  ("provisional", reason)  probably meets the rules, but needs the Phase 2 LLM
                           extraction to confirm (design from text only, or
                           sample size unknown)
  ("excluded", reason)     fails a rule
"""

from __future__ import annotations

from typing import Any


def assess(cred: dict[str, Any], rec: dict[str, Any]) -> tuple[str, str | None]:
    if set(rec["pub_types"]) & set(cred["exclude_pub_types"]):
        return "excluded", "excluded_pub_type"
    if rec.get("year") and rec["year"] < cred["min_year"]:
        return "excluded", "before_min_year"
    if cred["exclude_expression_of_concern"] and rec["expression_of_concern"]:
        return "excluded", "expression_of_concern"
    if cred["require_medline_journal"] and not rec["medline_indexed"]:
        return "excluded", "not_medline_journal"
    if rec["design"] not in cred["include_designs"]:
        return "excluded", "design_not_credible"

    provisional = None
    min_n = cred["min_n"].get(rec["design"])
    if min_n is not None:
        n = rec["sample_size"]
        if n is None:
            if cred["unknown_n"] == "exclude":
                return "excluded", "sample_size_unknown"
            if cred["unknown_n"] == "provisional":
                provisional = "sample_size_unknown"
        elif n < min_n:
            return "excluded", "below_min_sample_size"
    if rec["design_method"] not in cred["trusted_design_methods"]:
        provisional = provisional or "design_from_text_only"
    return ("provisional", provisional) if provisional else ("included", None)
