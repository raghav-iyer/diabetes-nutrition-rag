"""Markdown report for an evaluation run (headline metrics, per topic/type, ablations, tuning, failures)."""

from __future__ import annotations

from typing import Any

FIX_HINTS = {
    "not_in_corpus": "add sources (guideline PDFs, broader / Indian-food searches)",
    "not_retrieved": "improve recall: query rewriting, BM25 weighting, chunking strategy",
    "ranked_low": "tune fusion / reranker / alpha",
    "filtered_out": "relax or fix metadata filters",
    "ignored_by_llm": "prompt the generator to use all relevant sources",
    "unfaithful": "stricter generation prompt or claim verification",
    "query_misread": "improve query-understanding prompt (population / answerability)",
    "wrong_abstention": "tune the confidence threshold and answerability rules",
    "safety_miss": "tighten safety patterns (over- or under-triggering)",
}


def _fmt(v):
    if v is None:
        return "–"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _ci(ci):
    return f" ({ci[0]:.3f}–{ci[1]:.3f})" if ci else ""


def render(run_name: str, metrics: dict[str, Any], ablation: dict[str, Any] | None = None,
           tuning: dict[str, Any] | None = None, failures: list[dict[str, Any]] | None = None,
           failure_counts: dict[str, int] | None = None, judge_agreement: dict[str, Any] | None = None,
           examples: dict[str, list[dict]] | None = None) -> str:
    L = [f"# Evaluation run `{run_name}`", ""]
    ls = metrics.get("label_sources") or {}
    L += [f"Questions: {metrics['n_questions']} (retrieval-labelled: {metrics['n_retrieval_labelled']}). "
          f"Relevance labels: {ls.get('gold_labels', 0)} human (gold), {ls.get('silver_labels', 0)} LLM-drafted (silver). "
          f"Golden questions expert-verified: {ls.get('expert_verified', 0)}/{ls.get('items', 0)}. "
          f"Median seconds/question: {metrics.get('median_seconds')}.", ""]
    if ls.get("silver_labels") and not ls.get("gold_labels"):
        L += ["> **Caveat:** all relevance labels are LLM drafts not yet checked by a human; treat retrieval "
              "numbers as provisional.", ""]
    r = metrics.get("retrieval")
    if r:
        L += ["## Retrieval", "", "| Metric | Overall (95% CI) |", "|---|---|"]
        for k, v in r["overall"].items():
            if k != "n":
                L.append(f"| {k} | {_fmt(v)}{_ci(r['overall_ci'].get(k))} |")
        for group, key in (("topic", "by_topic"), ("type", None)):
            table = r.get("by_topic") if group == "topic" else metrics.get("retrieval_by_type")
            if not table:
                continue
            L += ["", f"**By {group}**", "", "| " + group + " | n | recall@10 | mrr | ndcg@10 |", "|---|---|---|---|---|"]
            for g, m in table.items():
                L.append(f"| {g} | {m['n']} | {_fmt(m.get('recall@10'))} | {_fmt(m.get('mrr'))} | {_fmt(m.get('ndcg@10'))} |")
        L.append("")
    a = metrics.get("answers")
    if a:
        L += ["## Answers", "", "| Metric | Overall (95% CI) |", "|---|---|"]
        for k, v in a["overall"].items():
            if k != "n":
                L.append(f"| {k} | {_fmt(v)}{_ci(a['overall_ci'].get(k))} |")
        if judge_agreement:
            L += ["", f"Judge agreement with human labels: accuracy {judge_agreement['accuracy']}, Cohen's kappa "
                      f"{judge_agreement['cohen_kappa']} (n = {judge_agreement['n']})."]
        else:
            L += ["", "> `judge_score` is reported without human validation yet (fill eval/judge_validation.csv)."]
        L += ["", "**By type**", "", "| type | n | abstention | safety | faithfulness | judge |", "|---|---|---|---|---|---|"]
        for g, m in a["by_type"].items():
            L.append(f"| {g} | {m['n']} | {_fmt(m.get('abstention_correct'))} | {_fmt(m.get('safety_compliant'))} | "
                     f"{_fmt(m.get('faithfulness'))} | {_fmt(m.get('judge_score'))} |")
        L.append("")
    if ablation:
        ms = list(ablation["base"])
        L += ["## Ablations (delta vs base, 95% CI)", "", "| Variant | " + " | ".join(ms) + " |", "|---|" + "---|" * len(ms),
              "| base | " + " | ".join(_fmt(ablation["base"][m]) for m in ms) + " |"]
        for name, v in ablation["variants"].items():
            cells = []
            for m in ms:
                d = v["delta_vs_base"].get(m)
                cells.append("–" if not d else f"{d['delta']:+.3f} ({d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f})")
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
    if tuning:
        ta, tt = tuning.get("alpha"), tuning.get("threshold")
        L += ["## Tuning (chosen on dev, reported on test)", ""]
        if ta:
            L.append(f"- alpha: best on dev = {ta['best_alpha_dev']}; test {ta['metric']} at best = {_fmt(ta['test_at_best'])}, "
                     f"at current = {_fmt(ta['test_at_current'])}")
        if tt:
            L.append(f"- confidence threshold: best on dev = {tt['best_threshold_dev']}; test balanced abstention accuracy "
                     f"= {_fmt(tt['test_at_best'])}")
        L.append("")
    if failures is not None:
        L += [f"## Failures ({len(failures)} questions)", "", "| Bucket | Count |", "|---|---|"]
        for b, n in (failure_counts or {}).items():
            L.append(f"| {b} | {n} |")
        for b, exs in (examples or {}).items():
            L += ["", f"**{b}** examples:"]
            for e in exs[:3]:
                L.append(f"- `{e['question_id']}` {e['question']} (status {e.get('status')}, recall@10 {_fmt(e.get('recall@10'))})")
        L += ["", "## What to fix next", ""]
        for i, (b, n) in enumerate(sorted((failure_counts or {}).items(), key=lambda x: -x[1]), 1):
            L.append(f"{i}. **{b}** ({n} questions): {FIX_HINTS.get(b, '')}")
    return "\n".join(L) + "\n"
