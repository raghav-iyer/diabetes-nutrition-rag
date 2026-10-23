"""Evaluation runner: retrieval + answers on the golden set, offline tuning, saved runs.

A run writes eval/runs/<timestamp>_<label>/{results.jsonl, metrics.json} so runs can be
compared (`nrag compare`). Retrieval metrics use document rankings built from the pipeline's
logged stages (final results first, then the remaining reranked candidates), so recall@20 is
defined even though only 10 results reach the generator.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from nrag.eval import answer_metrics, golden, judge as judge_mod, retrieval_metrics as rm

RETRIEVAL_KEYS = ["recall@5", "recall@10", "recall@20", "precision@5", "precision@10", "mrr", "ndcg@10"]
ANSWER_KEYS = ["abstention_correct", "safety_compliant", "faithfulness", "citation_precision", "citation_recall",
               "must_include_coverage", "rouge_l", "semantic_similarity", "judge_score"]


def ranked_docs(stages: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for _, doc, *_ in stages.get("final", []):
        if doc not in out:
            out.append(doc)
    for row in stages.get("reranked", []):
        if row[1] not in out:
            out.append(row[1])
    for row in stages.get("after_filters", []):
        if len(row) == 3 and row[1] not in out:
            out.append(row[1])
    return out


def evaluate_item(item, retriever, answerer=None, judge_chat=None, judge_cache: Path | None = None,
                  embed=None, safety_cfg: dict | None = None) -> dict[str, Any]:
    t0 = time.monotonic()
    ret = retriever.retrieve(item.question)
    row: dict[str, Any] = {"question_id": item.id, "question": item.question, "topic": item.topic, "type": item.type,
                           "split": golden.split(item.id), "retrieval_status": ret.status, "retried": ret.retried,
                           "analysis": ret.analysis, "stages": ret.stages, "timings": dict(ret.timings),
                           "ranked_docs": ranked_docs(ret.stages)}
    if item.graded_relevance and not item.should_abstain:
        row.update(rm.per_question(row["ranked_docs"], item.graded_relevance))
    if answerer is not None:
        out = answerer.answer(item.question, retrieval=ret)
        row["answer_status"] = out["status"]
        row["markdown"] = out.get("markdown")
        row["sources"] = list((out.get("sources") or {}).keys())
        row["contexts"] = out.get("contexts") or []
        row["direct_answer"] = (out.get("answer") or {}).get("direct_answer")
        row["cited"] = sorted({x for c in ((out.get("answer") or {}).get("claims") or []) for x in c.get("citations", [])})
        row["verification"] = out.get("verification")
        row["safety"] = out.get("safety")
        row.update(answer_metrics.score_answer(item, out, safety_cfg["dose_pattern"], safety_cfg["dose_redaction"], embed))
        a = out.get("answer") or {}
        if judge_chat and item.reference_answer and not row["abstained"] and not item.should_abstain:
            v = judge_mod.judge(item.question, item.reference_answer, a.get("direct_answer") or "", judge_chat, judge_cache)
            row["judge_score"], row["judge_reason"] = v["score"], v["reason"]
            row["judge_score_norm"] = v["score"] / 2 if v["score"] is not None else None
    row["seconds"] = round(time.monotonic() - t0, 1)
    return row


def summarise(rows: list[dict[str, Any]], items_by_id: dict[str, Any]) -> dict[str, Any]:
    labelled = [r for r in rows if "ndcg@10" in r]
    for r in rows:
        for k in ("abstention_correct", "safety_compliant"):
            if isinstance(r.get(k), bool):
                r[k] = float(r[k])
        if "judge_score_norm" in r:
            r["judge_score"] = r["judge_score_norm"]
    out = {"n_questions": len(rows), "n_retrieval_labelled": len(labelled),
           "retrieval": rm.aggregate(labelled, RETRIEVAL_KEYS, "topic") if labelled else None,
           "retrieval_by_type": rm.aggregate(labelled, RETRIEVAL_KEYS, "type")["by_type"] if labelled else None,
           "answers": rm.aggregate(rows, ANSWER_KEYS, "type") if any("abstained" in r for r in rows) else None,
           "label_sources": golden.coverage(list(items_by_id.values())),
           "median_seconds": sorted(r["seconds"] for r in rows)[len(rows) // 2] if rows else None}
    return out


def save_run(rows: list[dict[str, Any]], metrics: dict[str, Any], runs_dir: Path, label: str,
             config_snapshot: dict[str, Any]) -> Path:
    d = runs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{label}"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "results.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    (d / "config.json").write_text(json.dumps(config_snapshot, indent=1, default=str))
    return d


def load_run(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [json.loads(line) for line in (path / "results.jsonl").read_text().splitlines() if line.strip()]
    return rows, json.loads((path / "metrics.json").read_text())


def compare_runs(a: list[dict[str, Any]], b: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    """Paired deltas (a - b) per metric over questions present in both runs."""
    bi = {r["question_id"]: r for r in b}
    common = [r for r in a if r["question_id"] in bi]
    return {k: rm.paired_delta([r.get(k) for r in common], [bi[r["question_id"]].get(k) for r in common])
            for k in keys}


# ---------------------------------------------------------------------------
# Offline tuning from logged candidates (no re-retrieval)
# ---------------------------------------------------------------------------
def rerank_offline(reranked: list[list], alpha: float, guideline_bonus: float, threshold: float,
                   max_per_doc: int, keep: int) -> list[str]:
    scored = []
    for chunk_id, doc, rr, q, stype in reranked:
        s = alpha * rr + (1 - alpha) * (q or 0.0)
        if stype == "guideline" and rr >= threshold:
            s += guideline_bonus
        scored.append((s, doc))
    scored.sort(key=lambda x: -x[0])
    out, per = [], {}
    for _, doc in scored:
        if per.get(doc, 0) >= max_per_doc:
            continue
        per[doc] = per.get(doc, 0) + 1
        if doc not in out:
            out.append(doc)
        if len(out) >= keep:
            break
    return out


def tune_alpha(rows: list[dict[str, Any]], items_by_id: dict[str, Any], alphas: list[float], rc: dict[str, Any],
               metric: str = "ndcg@10") -> dict[str, Any]:
    """Pick alpha on the dev split by mean metric; report it on the test split."""
    res = {}
    for split_name in ("dev", "test"):
        res[split_name] = {}
        for a in alphas:
            vals = []
            for r in rows:
                it = items_by_id[r["question_id"]]
                if r["split"] != split_name or not it.graded_relevance or it.should_abstain:
                    continue
                docs = rerank_offline(r["stages"].get("reranked", []), a, rc["final"]["guideline_bonus"],
                                      rc["confidence"]["min_top_rerank"], rc["max_chunks_per_doc"], 20)
                vals.append(rm.per_question(docs, it.graded_relevance)[metric])
            res[split_name][a] = rm.mean(vals)
    dev = {a: v for a, v in res["dev"].items() if v is not None}
    best = max(dev, key=dev.get) if dev else None
    return {"metric": metric, "by_split": res, "best_alpha_dev": best,
            "test_at_best": res["test"].get(best) if best is not None else None,
            "test_at_current": res["test"].get(rc["final"]["alpha"])}


def tune_threshold(rows: list[dict[str, Any]], items_by_id: dict[str, Any], thresholds: list[float]) -> dict[str, Any]:
    """Confidence threshold on the top rerank score, chosen on dev for balanced abstention accuracy."""
    def top(r):
        rr = r["stages"].get("reranked") or []
        return max((x[2] for x in rr), default=0.0)

    def balanced(rs, t):
        pos = [r for r in rs if items_by_id[r["question_id"]].should_abstain]
        neg = [r for r in rs if not items_by_id[r["question_id"]].should_abstain]
        if not pos or not neg:
            return None
        tpr = sum(1 for r in pos if top(r) < t or r["retrieval_status"] == "not_answerable") / len(pos)
        tnr = sum(1 for r in neg if top(r) >= t and r["retrieval_status"] != "not_answerable") / len(neg)
        return round((tpr + tnr) / 2, 4)

    res = {s: {t: balanced([r for r in rows if r["split"] == s], t) for t in thresholds} for s in ("dev", "test")}
    dev = {t: v for t, v in res["dev"].items() if v is not None}
    best = max(dev, key=dev.get) if dev else None
    return {"by_split": res, "best_threshold_dev": best, "test_at_best": res["test"].get(best) if best is not None else None}
