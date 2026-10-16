"""Human review of extractions and field-level accuracy.

Workflow:
  1. `stratified_sample` picks documents balanced across (design, topic). The
     first 30 are the quick screen; the first 100 the full review set (the 30
     are a subset, so labels carry over).
  2. `write_review_csv` writes one row per (document, field) with up to two
     systems' values in blinded columns A and B (random per document; the key
     is saved separately). The reviewer fills `verdict` with A, B, both or
     neither, and `correct_value` when neither is right.
  3. `score_review` derives a gold value per row and scores every system (the
     two models, plus a rules-only baseline from PubMed metadata) against it:
     accuracy, kappa, tag precision/recall/F1, numeric tolerance, hallucinated
     "yes" answers, paired bootstrap CIs and McNemar tests, speed, and the
     pre-registered decision rule.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REVIEW_FIELDS = [
    "study_design", "population.tags", "population.description", "intervention", "comparison", "outcomes",
    "effect_direction", "key_finding", "sample_size", "duration_weeks", "country_or_region",
    "risk_of_bias_signals.randomisation_described", "risk_of_bias_signals.blinding",
    "risk_of_bias_signals.low_dropout", "risk_of_bias_signals.pre_registered", "funding_industry",
]
CATEGORICAL = {"study_design", "effect_direction", "funding_industry",
               "risk_of_bias_signals.randomisation_described", "risk_of_bias_signals.blinding",
               "risk_of_bias_signals.low_dropout", "risk_of_bias_signals.pre_registered"}
NUMERIC = {"sample_size", "duration_weeks"}
YES_NO_FIELDS = {f for f in CATEGORICAL if f.startswith("risk_of_bias_signals.")}
CSV_COLUMNS = ["doc_id", "pmid", "field", "A", "B", "verdict", "correct_value", "notes", "title", "abstract"]
PHASE1_TO_DESIGN = {"controlled_trial": "non_randomised_trial", "cohort": "retrospective_cohort",
                    "observational": "other", "unknown": "other", "opinion": "other"}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def stratified_sample(docs: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Round-robin over (design, first topic) strata, shuffled deterministically within each."""
    rng = random.Random(seed)
    strata: dict[tuple, list] = defaultdict(list)
    for d in sorted(docs, key=lambda d: d["doc_id"]):
        if d.get("pmid") and d.get("abstract"):
            strata[(d["design"], (d.get("topics") or ["none"])[0])].append(d)
    for key in strata:
        rng.shuffle(strata[key])
    keys = sorted(strata)
    out: list[dict[str, Any]] = []
    while len(out) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(out) < n:
                out.append(strata[k].pop())
    return out


# ---------------------------------------------------------------------------
# Field access and normalisation
# ---------------------------------------------------------------------------
def get_field(extraction: dict[str, Any] | None, field: str) -> Any:
    cur: Any = extraction
    for part in field.split("."):
        if cur is None:
            return None
        cur = cur.get(part)
    return cur


def render(value: Any, field: str) -> str:
    if value is None:
        return "null"
    if field == "outcomes":
        return "; ".join(f"{o['name']} [{o['type']}]" for o in value) or "none"
    if isinstance(value, list):
        return "; ".join(sorted(value)) or "none"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def norm(text: str | None) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _num(text: str) -> float | None:
    try:
        return float(str(text).replace(",", ""))
    except ValueError:
        return None


def values_match(field: str, value: str, gold: str, tol: float = 0.05) -> bool:
    v, g = norm(value), norm(gold)
    if field in NUMERIC:
        if g in {"null", ""} or v in {"null", ""}:
            return g in {"null", ""} and v in {"null", ""}
        nv, ng = _num(v), _num(g)
        return nv is not None and ng is not None and abs(nv - ng) <= tol * max(abs(ng), 1)
    if field == "population.tags":
        return set(v.split("; ")) - {""} == set(g.split("; ")) - {""}
    return v == g


# ---------------------------------------------------------------------------
# Rules-only baseline (no LLM)
# ---------------------------------------------------------------------------
def rules_baseline(doc: dict[str, Any], mesh_tags: dict[str, list[str]]) -> dict[str, str]:
    mesh = set(doc.get("mesh") or [])
    tags = {tag for tag, terms in mesh_tags.items() if mesh & set(terms)}
    if "Adult" in mesh and not tags - {"older_adults"}:
        tags.add("general_adults")
    design = PHASE1_TO_DESIGN.get(doc.get("design"), doc.get("design"))
    return {"population.tags": render(sorted(tags), "population.tags"),
            "sample_size": render(doc.get("sample_size"), "sample_size"),
            "study_design": design or "other"}


# ---------------------------------------------------------------------------
# Review CSV
# ---------------------------------------------------------------------------
def write_review_csv(path: Path, key_path: Path, docs: list[dict[str, Any]],
                     systems: dict[str, dict[str, dict[str, Any]]], seed: int) -> None:
    """`systems` maps a system name to {doc_id: extraction}. One or two systems."""
    names = list(systems)
    if not 1 <= len(names) <= 2:
        raise ValueError("review CSV supports one or two systems")
    rng = random.Random(seed)
    key: dict[str, dict[str, str]] = {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for d in docs:
            order = names[:]
            if len(order) == 2:
                rng.shuffle(order)
            key[d["doc_id"]] = dict(zip("AB", order))
            for i, field in enumerate(REVIEW_FIELDS):
                row = {"doc_id": d["doc_id"], "pmid": d.get("pmid", ""), "field": field,
                       "A": render(get_field(systems[order[0]].get(d["doc_id"]), field), field),
                       "B": render(get_field(systems[order[1]].get(d["doc_id"]), field), field) if len(order) == 2 else "",
                       "verdict": "", "correct_value": "", "notes": "",
                       "title": d.get("title", "") if i == 0 else "",
                       "abstract": d.get("abstract", "") if i == 0 else ""}
                w.writerow(row)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps({"systems": names, "assignment": key}, indent=1))


def read_review(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def gold_value(row: dict[str, str]) -> str | None:
    v = norm(row["verdict"])
    if v in {"a", "both"}:
        return row["A"]
    if v == "b":
        return row["B"]
    if v == "neither":
        return row["correct_value"] if row["correct_value"].strip() else None
    return None   # unlabelled


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def bootstrap_diff(per_doc: dict[str, tuple[float, float]], iters: int, seed: int) -> tuple[float, float, float]:
    """Mean of (x - y) over documents with a 95% percentile CI, resampling documents."""
    docs = list(per_doc)
    if not docs:
        return 0.0, 0.0, 0.0
    diffs = [per_doc[d][0] - per_doc[d][1] for d in docs]
    point = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    stats = sorted(sum(rng.choice(diffs) for _ in docs) / len(docs) for _ in range(iters))
    return round(point, 4), round(stats[int(0.025 * iters)], 4), round(stats[int(0.975 * iters) - 1], 4)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_review(rows: list[dict[str, str]], key: dict[str, Any], baseline: dict[str, dict[str, str]],
                 speeds: dict[str, dict[str, Any]], decision_cfg: dict[str, Any], iters: int = 2000,
                 seed: int = 0) -> dict[str, Any]:
    systems = key["systems"] + (["rules_baseline"] if baseline else [])
    correct: dict[str, dict[str, dict[str, bool]]] = {s: defaultdict(dict) for s in systems}   # sys -> field -> doc -> ok
    pairs: dict[str, dict[str, list]] = {s: defaultdict(list) for s in systems}
    tag_counts = {s: Counter() for s in systems}
    halluc = {s: Counter() for s in systems}
    labelled_docs, unlabelled = set(), 0

    for row in rows:
        gold = gold_value(row)
        if gold is None:
            unlabelled += 1
            continue
        doc, field = row["doc_id"], row["field"]
        labelled_docs.add(doc)
        values = {sysname: row[letter] for letter, sysname in key["assignment"][doc].items()}
        if baseline and field in baseline.get(doc, {}):
            values["rules_baseline"] = baseline[doc][field]
        for sysname, value in values.items():
            if field in CATEGORICAL or field in NUMERIC or field == "population.tags" or sysname == "rules_baseline":
                ok = values_match(field, value, gold)
            else:   # free text: the verdict is the judgement
                letter = next(k for k, v in key["assignment"][doc].items() if v == sysname)
                ok = norm(row["verdict"]) in {letter.lower(), "both"}
            correct[sysname][field][doc] = ok
            if field in CATEGORICAL:
                pairs[sysname][field].append((norm(value), norm(gold)))
            if field in YES_NO_FIELDS:
                halluc[sysname]["opportunities"] += norm(gold) == "not_reported"
                halluc[sysname]["yes_when_not_reported"] += norm(gold) == "not_reported" and norm(value) == "yes"
            if field == "population.tags":
                pv = set(norm(value).split("; ")) - {"", "none", "null"}
                gv = set(norm(gold).split("; ")) - {"", "none", "null"}
                tag_counts[sysname].update(tp=len(pv & gv), fp=len(pv - gv), fn=len(gv - pv))

    per_system = {}
    for s in systems:
        fields = {}
        for field, docs in correct[s].items():
            fields[field] = {"n": len(docs), "accuracy": round(sum(docs.values()) / len(docs), 4)}
            if field in pairs[s]:
                fields[field]["kappa"] = cohen_kappa(pairs[s][field])
        tc = tag_counts[s]
        p = tc["tp"] / (tc["tp"] + tc["fp"]) if tc["tp"] + tc["fp"] else None
        r = tc["tp"] / (tc["tp"] + tc["fn"]) if tc["tp"] + tc["fn"] else None
        f1 = 2 * p * r / (p + r) if p and r else None
        h = halluc[s]
        per_system[s] = {
            "fields": dict(sorted(fields.items())),
            "tags_micro": {"precision": _r(p), "recall": _r(r), "f1": _r(f1)},
            "hallucinated_yes_rate": _r(h["yes_when_not_reported"] / h["opportunities"]) if h["opportunities"] else None,
            "speed": speeds.get(s),
        }

    comparison = None
    models = key["systems"]
    critical = decision_cfg["critical_fields"]
    if len(models) == 2:
        a, b = models
        per_doc = {}
        for doc in labelled_docs:
            fa = [correct[a][f][doc] for f in critical if doc in correct[a].get(f, {})]
            fb = [correct[b][f][doc] for f in critical if doc in correct[b].get(f, {})]
            if fa and fb:
                per_doc[doc] = (sum(fa) / len(fa), sum(fb) / len(fb))
        diff, lo, hi = bootstrap_diff(per_doc, iters, seed)
        mcnemar = {}
        for f in critical:
            docs = set(correct[a].get(f, {})) & set(correct[b].get(f, {}))
            bc = sum(correct[a][f][d] and not correct[b][f][d] for d in docs)
            cb = sum(correct[b][f][d] and not correct[a][f][d] for d in docs)
            mcnemar[f] = {"a_only": bc, "b_only": cb, "p": round(mcnemar_p(bc, cb), 4)}
        comparison = {"a": a, "b": b, "critical_fields": critical, "docs": len(per_doc),
                      "critical_accuracy_diff_a_minus_b": diff, "ci95": [lo, hi], "mcnemar": mcnemar,
                      "decision": decide(a, b, diff, lo, hi, speeds, decision_cfg)}

    return {"labelled_documents": len(labelled_docs), "unlabelled_rows": unlabelled,
            "systems": per_system, "comparison": comparison}


def decide(a: str, b: str, diff: float, lo: float, hi: float, speeds: dict, decision_cfg: dict) -> dict[str, Any]:
    """Pre-registered rule: use the faster model unless it is worse on critical fields
    by more than `max_accuracy_drop` with a 95% CI that excludes zero."""
    sa, sb = (speeds.get(a) or {}).get("median"), (speeds.get(b) or {}).get("median")
    if sa is None or sb is None:
        return {"choice": None, "reason": "speed not measured for both models"}
    fast, slow = (a, b) if sa <= sb else (b, a)
    # Express the difference as fast - slow.
    d, l, h = (diff, lo, hi) if fast == a else (-diff, -hi, -lo)
    drop = decision_cfg["max_accuracy_drop"]
    if d < -drop and h < 0:
        return {"choice": slow, "reason": f"{fast} is {abs(d):.1%} worse on critical fields (CI {l:+.1%} to {h:+.1%})"}
    return {"choice": fast, "reason": f"{fast} is faster and not credibly worse by more than {drop:.0%} "
                                      f"(diff {d:+.1%}, CI {l:+.1%} to {h:+.1%})"}


def _r(x):
    return round(x, 4) if x is not None else None


def render_report(result: dict[str, Any]) -> str:
    lines = ["# Extraction accuracy", "",
             f"Labelled documents: {result['labelled_documents']} (unlabelled rows skipped: {result['unlabelled_rows']})", ""]
    systems = list(result["systems"])
    fields = sorted({f for s in systems for f in result["systems"][s]["fields"]})
    lines += ["| Field | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for f in fields:
        cells = []
        for s in systems:
            m = result["systems"][s]["fields"].get(f)
            cells.append("" if not m else f"{m['accuracy']:.0%}" + (f" (κ {m['kappa']})" if m.get("kappa") is not None else ""))
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    lines += ["", "| System | Tag P / R / F1 | Hallucinated 'yes' | Median s/abstract |", "|---|---|---|---|"]
    for s in systems:
        m = result["systems"][s]
        t = m["tags_micro"]
        spd = (m.get("speed") or {}).get("median")
        lines.append(f"| {s} | {t['precision']} / {t['recall']} / {t['f1']} | {m['hallucinated_yes_rate']} | {spd} |")
    c = result.get("comparison")
    if c:
        lines += ["", f"**{c['a']} − {c['b']} on critical fields** ({', '.join(c['critical_fields'])}): "
                      f"{c['critical_accuracy_diff_a_minus_b']:+.1%} (95% CI {c['ci95'][0]:+.1%} to {c['ci95'][1]:+.1%}, "
                      f"{c['docs']} documents)", ""]
        for f, m in c["mcnemar"].items():
            lines.append(f"- {f}: {c['a']} only right {m['a_only']}, {c['b']} only right {m['b_only']}, McNemar p = {m['p']}")
        lines += ["", f"**Decision:** {c['decision']['choice']}: {c['decision']['reason']}"]
    return "\n".join(lines) + "\n"
