"""`nrag score`: write scores_v2.jsonl and a report comparing v1 and v2."""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

from nrag.config import resolve_path
from nrag.extract import runner as extract_runner
from nrag.extract.pico import model_slug
from nrag.score import v2


def run(cfg: dict[str, Any], model: str | None = None) -> tuple[Path, Path]:
    docs = extract_runner.load_documents(cfg)
    model = model or extract_runner.active_model(cfg)
    ext_path = extract_runner.output_path(cfg, model_slug(model, extract_runner.parse_think(cfg["extract"]["think"].get(model))
                                                          if cfg["extract"]["think"].get(model) else None))
    exts = v2.load_extractions(ext_path)
    rows = {d["doc_id"]: v2.score_doc(cfg, d, exts.get(d["doc_id"])) for d in docs}
    sup = v2.find_supersession(cfg, docs, exts)
    for doc_id, link in sup.items():
        rows[doc_id].update(link)
        rows[doc_id]["flags"].append("superseded")
    out = resolve_path(cfg["score_v2"]["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for d in docs:
            f.write(json.dumps(rows[d["doc_id"]]) + "\n")
    report = resolve_path(cfg["score_v2"]["report"])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_report(docs, rows, sup, exts, ext_path))
    return out, report


def _report(docs, rows, sup, exts, ext_path) -> str:
    by_id = {d["doc_id"]: d for d in docs}
    ex_rows = [r for r in rows.values() if r["extracted"]]
    v1 = [r["quality_score_v1"] for r in ex_rows]
    v2s = [r["quality_score_v2"] for r in ex_rows]
    lines = ["# Quality score v2 and supersession", "",
             f"Extractions: `{ext_path.name}`; {len(ex_rows)}/{len(docs)} documents extracted (others keep v1).", ""]
    if len(ex_rows) > 2:
        changed = sum(1 for a, b in zip(v1, v2s) if a != b)
        lines += [f"- Spearman rho (v1 vs v2) over extracted docs: {v2.spearman(v1, v2s):.3f}",
                  f"- Documents whose score changed: {changed}",
                  f"- Mean v1 {sum(v1)/len(v1):.3f}, mean v2 {sum(v2s)/len(v2s):.3f}", ""]
    lines += ["## By design (mean v1 -> v2)", "", "| design | n | v1 | v2 |", "|---|---|---|---|"]
    groups = collections.defaultdict(list)
    for r in ex_rows:
        groups[by_id[r["doc_id"]]["design"]].append(r)
    for g, rs in sorted(groups.items()):
        lines.append(f"| {g} | {len(rs)} | {sum(r['quality_score_v1'] for r in rs)/len(rs):.3f} | "
                     f"{sum(r['quality_score_v2'] for r in rs)/len(rs):.3f} |")
    flags = collections.Counter(f for r in rows.values() for f in r["flags"])
    lines += ["", "## Flags", ""] + [f"- {k}: {v}" for k, v in sorted(flags.items())]
    lines += ["", f"## Supersession ({len(sup)} documents superseded)", "",
              "Spot-check these pairs by hand before trusting the thresholds in `score_v2.supersession`.", "",
              "| older | newer | title sim | pop overlap |", "|---|---|---|---|"]
    for old, l in sorted(sup.items(), key=lambda kv: -kv[1]["title_similarity"])[:25]:
        new = l["superseded_by"]
        lines.append(f"| {old} ({by_id[old]['year']}) {by_id[old]['title'][:70]} | {new} ({by_id[new]['year']}) "
                     f"{by_id[new]['title'][:70]} | {l['title_similarity']} | {l['population_overlap']} |")
    return "\n".join(lines) + "\n"
