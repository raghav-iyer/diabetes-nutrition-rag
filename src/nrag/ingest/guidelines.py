"""Guidelines registry and PDF parsing.

Guidelines are mostly not on PubMed as full text, so they come from a
hand-maintained registry (data/guidelines/registry.yaml) plus PDFs the user
downloads. Parsing: pymupdf for text with font sizes (headings are lines set
noticeably larger than body text, or bold numbered lines), pdfplumber for
tables (kept whole; their text is removed from the running sections).
Recommendations are paragraphs matching the entry's grade pattern.
"""

from __future__ import annotations

import glob
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, field_validator

from nrag.config import resolve_path

NUMBERED_HEADING = re.compile(r"^(\d+(\.\d+)*\.?|[A-Z]\.\d+(\.\d+)*|chapter \d+|section \d+|part [ivx\d]+)\s+\S", re.I)


class GuidelineEntry(BaseModel):
    id: str
    title: str
    issuing_body: str
    year: int
    version: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    topics: list[str]
    population: list[str]
    file: str | None = None
    file_glob: str | None = None
    supersedes: str | None = None
    status: Literal["current", "superseded"] = "current"
    grade_scheme: str | None = None
    verify: bool = False
    notes: str | None = None

    @field_validator("id")
    @classmethod
    def _id_ok(cls, v):
        if not re.fullmatch(r"[a-z0-9_]+", v):
            raise ValueError("id must be lowercase letters, digits and underscores")
        return v

    def files(self) -> list[Path]:
        if self.file_glob:
            return sorted(Path(p) for p in glob.glob(str(resolve_path(self.file_glob))))
        if self.file and resolve_path(self.file).exists():
            return [resolve_path(self.file)]
        return []


def load_registry(cfg: dict[str, Any], path: str | Path | None = None) -> list[GuidelineEntry]:
    path = resolve_path(path or cfg["paths"]["guidelines_registry"])
    data = yaml.safe_load(Path(path).read_text()) or {}
    entries = [GuidelineEntry(**e) for e in data.get("guidelines", [])]
    topics, pops = set(cfg["pubmed"]["topics"]), set(cfg["population_tags"])
    ids = [e.id for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate guideline ids in registry")
    for e in entries:
        if set(e.topics) - topics:
            raise ValueError(f"{e.id}: unknown topics {set(e.topics) - topics}")
        if set(e.population) - pops:
            raise ValueError(f"{e.id}: unknown population tags {set(e.population) - pops}")
        if e.supersedes and e.supersedes not in ids:
            raise ValueError(f"{e.id}: supersedes unknown id {e.supersedes}")
        if e.grade_scheme and e.grade_scheme not in cfg["guidelines"]["grade_patterns"]:
            raise ValueError(f"{e.id}: unknown grade_scheme {e.grade_scheme}")
    return entries


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------
def _inside(bbox, boxes) -> bool:
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes)


def _extract_tables(path: Path) -> dict[int, list[dict[str, Any]]]:
    import pdfplumber

    out: dict[int, list[dict[str, Any]]] = {}
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            for t in page.find_tables():
                rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in t.extract()]
                rows = [r for r in rows if any(r)]
                if rows:
                    out.setdefault(i, []).append({"page": i + 1, "bbox": list(t.bbox), "rows": rows})
    return out


def _lines(path: Path, table_boxes: dict[int, list]) -> list[dict[str, Any]]:
    import pymupdf

    lines = []
    with pymupdf.open(path) as doc:
        for pno, page in enumerate(doc):
            boxes = [t["bbox"] for t in table_boxes.get(pno, [])]
            for bno, block in enumerate(page.get_text("dict")["blocks"]):
                if block.get("type") != 0:
                    continue
                for line in block["lines"]:
                    spans = [s for s in line["spans"] if s["text"].strip()]
                    if not spans or _inside(line["bbox"], boxes):
                        continue
                    lines.append({
                        "page": pno + 1, "block": (pno, bno),
                        "text": " ".join(s["text"].strip() for s in spans),
                        "size": max(s["size"] for s in spans),
                        "bold": all((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans),
                        "chars": sum(len(s["text"]) for s in spans),
                    })
    return lines


def _body_size(lines) -> float:
    c = Counter()
    for ln in lines:
        c[round(ln["size"] * 2) / 2] += ln["chars"]
    return c.most_common(1)[0][0] if c else 10.0


def parse_guideline_pdf(path: Path, gcfg: dict[str, Any]) -> dict[str, Any]:
    tables = _extract_tables(path)
    lines = _lines(path, tables)
    body = _body_size(lines)

    def is_heading(ln):
        if ln["chars"] > gcfg["heading_max_chars"]:
            return False
        return ln["size"] >= body * gcfg["heading_size_ratio"] or (ln["bold"] and bool(NUMBERED_HEADING.match(ln["text"])))

    sections: list[dict[str, Any]] = []
    cur = {"title": "", "page_start": 1, "paragraphs": [], "_buf": [], "_block": None}

    def flush_para():
        if cur["_buf"]:
            cur["paragraphs"].append(re.sub(r"\s+", " ", " ".join(cur["_buf"])).strip())
            cur["_buf"] = []

    def close():
        flush_para()
        text = "\n\n".join(p for p in cur["paragraphs"] if p)
        if len(text) >= gcfg["min_section_chars"]:
            sections.append({"name": "guideline_section", "title": cur["title"],
                             "page_start": cur["page_start"], "paragraphs": list(cur["paragraphs"]), "text": text})

    for ln in lines:
        if is_heading(ln):
            close()
            # Consecutive heading lines (wrapped titles) join into one title.
            if not cur["paragraphs"] and not cur["_buf"] and cur["title"] and cur["_block"] == ln["block"]:
                cur["title"] += " " + ln["text"]
            else:
                cur = {"title": ln["text"], "page_start": ln["page"], "paragraphs": [], "_buf": [],
                       "_block": ln["block"]}
            continue
        if cur["_block"] != ln["block"]:
            flush_para()
            cur["_block"] = ln["block"]
        cur["_buf"].append(ln["text"])
    close()

    return {"sections": sections, "tables": [t for p in sorted(tables) for t in tables[p]], "body_font_size": body}


def find_recommendations(sections, pattern: re.Pattern, scheme: str, entry_id: str) -> list[dict[str, Any]]:
    recs = []
    for sec in sections:
        for para in sec["paragraphs"]:
            for m in pattern.finditer(para):
                recs.append({
                    "id": f"{entry_id}:r{len(recs) + 1}",
                    "text": m.group("text").strip(),
                    "grade": re.sub(r"\s+", " ", m.group("grade")).strip(),
                    "grade_scheme": scheme,
                    "section": sec["title"],
                    "page": sec["page_start"],
                })
    return recs


def parse_entry(entry: GuidelineEntry, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Parse all PDFs for an entry. Returns None if no file is present."""
    files = entry.files()
    if not files:
        return None
    gcfg = cfg["guidelines"]
    sections, tables = [], []
    for f in files:
        parsed = parse_guideline_pdf(f, gcfg)
        for s in parsed["sections"]:
            s["file"] = f.name
        for t in parsed["tables"]:
            t["file"] = f.name
        sections += parsed["sections"]
        tables += parsed["tables"]
    recs = []
    if entry.grade_scheme:
        pattern = re.compile(gcfg["grade_patterns"][entry.grade_scheme])
        recs = find_recommendations(sections, pattern, entry.grade_scheme, entry.id)
    for s in sections:
        s.pop("paragraphs", None)
    return {"doc_id": f"GUIDELINE:{entry.id}", "files": [f.name for f in files],
            "sections": sections, "tables": tables, "recommendations": recs}
