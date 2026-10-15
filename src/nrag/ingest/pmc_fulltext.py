"""PMC full text (JATS XML) with a licence gate.

PMC's efetch returns full text for many articles whose licence does NOT allow
reuse (author manuscripts "available for text mining", time-limited COVID-19
permissions). So the licence is read from each article's <permissions> block
and only allowlisted licences (config `pmc.allowed_licences`) are stored.
Sections are normalised (introduction / methods / results / discussion /
conclusion / other); tables are kept whole and stored separately.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any, Iterator

from nrag.net import CachedHttp

ALI = "{http://www.niso.org/schemas/ali/1.0/}"
XLINK = "{http://www.w3.org/1999/xlink}"
SKIP_IN_TEXT = {"table-wrap", "fig", "table-wrap-foot", "supplementary-material", "disp-formula"}

CC_URL = re.compile(r"creativecommons\.org/(licenses|publicdomain)/([a-z\-]+)", re.I)


def licence_from_url(url: str) -> str | None:
    m = CC_URL.search(url or "")
    if not m:
        return None
    kind, code = m.group(1).lower(), m.group(2).lower()
    if kind == "publicdomain":
        return "CC0" if code == "zero" else "PD"
    return "CC-" + code.upper()


def detect_licence(article: ET.Element) -> tuple[str, str | None]:
    """Return (licence code, licence URL). Unknown or non-CC licences get a descriptive code."""
    lic = article.find(".//front//permissions/license")
    if lic is None:
        lic = article.find(".//permissions/license")
    if lic is None:
        return "none", None
    candidates = [el.text or "" for el in lic.iter(f"{ALI}license_ref")]
    candidates += [lic.get(f"{XLINK}href") or ""]
    candidates += [el.get(f"{XLINK}href") or "" for el in lic.iter("ext-link")]
    for url in candidates:
        code = licence_from_url(url)
        if code:
            return code, url.strip()
    text = " ".join(lic.itertext()).lower()
    if "covid" in text:
        return "pmc-covid-temporary", None
    if "text mining" in text or "fair use" in text:
        return "pmc-text-mining-only", None
    return "other", None


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _text_excluding(el: ET.Element, skip: set[str]) -> str:
    parts = []

    def walk(e):
        if e.tag in skip:
            if e.tail:
                parts.append(e.tail)
            return
        if e.tag == "title" and e is not el:
            parts.append("\n")
        if e.text:
            parts.append(e.text)
        for child in e:
            walk(child)
        if e.tag in {"p", "title", "list-item"}:
            parts.append("\n")
        if e.tail and e is not el:
            parts.append(e.tail)

    walk(el)
    lines = [_clean(line) for line in "".join(parts).split("\n")]
    return "\n".join(line for line in lines if line)


def section_name(title: str, sec_type: str | None, section_map: dict[str, list[str]]) -> str:
    probe = f"{(sec_type or '').lower()} {title.lower()}"
    for name, keys in section_map.items():
        if any(k in probe for k in keys):
            return name
    return "other"


def parse_table(tw: ET.Element) -> dict[str, Any]:
    rows = []
    for tr in tw.iter("tr"):
        rows.append([_clean("".join(c.itertext())) for c in tr if c.tag in {"td", "th"}])
    foot = tw.find("table-wrap-foot")
    return {
        "id": tw.get("id"),
        "label": _clean("".join(tw.find("label").itertext())) if tw.find("label") is not None else None,
        "caption": _clean("".join(tw.find("caption").itertext())) if tw.find("caption") is not None else "",
        "rows": [r for r in rows if any(r)],
        "footer": _clean("".join(foot.itertext())) if foot is not None else "",
    }


def table_to_markdown(table: dict[str, Any]) -> str:
    rows = table["rows"]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def parse_jats_article(article: ET.Element, pmc_cfg: dict[str, Any]) -> dict[str, Any]:
    ids = {el.get("pub-id-type"): (el.text or "").strip() for el in article.iter("article-id")}
    pmcid = ids.get("pmcid") or ids.get("pmc")
    if pmcid and not pmcid.upper().startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    licence, licence_url = detect_licence(article)
    out = {"pmcid": pmcid, "pmid": ids.get("pmid"), "licence": licence, "licence_url": licence_url,
           "allowed": licence in pmc_cfg["allowed_licences"], "sections": [], "tables": []}
    if not out["allowed"]:
        return out

    skip_types = {t.lower() for t in pmc_cfg["skip_sec_types"]}
    skip_titles = [t.lower() for t in pmc_cfg["skip_titles"]]
    body = article.find("body")
    if body is not None:
        for sec in body.findall("sec"):
            title = _clean("".join(sec.find("title").itertext())) if sec.find("title") is not None else ""
            sec_type = sec.get("sec-type")
            if (sec_type or "").lower() in skip_types or any(t in title.lower() for t in skip_titles):
                continue
            text = _text_excluding(sec, SKIP_IN_TEXT)
            if title and text.startswith(title):
                text = text[len(title):].lstrip("\n")
            if text:
                out["sections"].append({"name": section_name(title, sec_type, pmc_cfg["section_map"]),
                                        "title": title, "text": text})
        # Untitled body paragraphs (some articles have no <sec>).
        loose = [_clean("".join(p.itertext())) for p in body.findall("p")]
        if loose and not out["sections"]:
            out["sections"].append({"name": "other", "title": "", "text": "\n".join(x for x in loose if x)})

    for container in (body, article.find("floats-group")):
        if container is not None:
            out["tables"] += [parse_table(tw) for tw in container.iter("table-wrap")]
    return out


def parse_jats(xml_bytes: bytes, pmc_cfg: dict[str, Any]) -> Iterator[dict[str, Any]]:
    root = ET.fromstring(xml_bytes)
    articles = [root] if root.tag == "article" else root.iter("article")
    for article in articles:
        yield parse_jats_article(article, pmc_cfg)


def fetch_fulltext(http: CachedHttp, cfg: dict[str, Any], pmcids: list[str],
                   base_params: dict[str, str]) -> Iterator[dict[str, Any]]:
    pc = cfg["pmc"]
    for start in range(0, len(pmcids), pc["efetch_batch"]):
        chunk = pmcids[start:start + pc["efetch_batch"]]
        content = http.post(f"{cfg['ncbi']['eutils_base']}/efetch.fcgi", data={
            **base_params, "db": "pmc", "id": ",".join(chunk),
        }, ttl_days=pc["ttl_days"])
        yield from parse_jats(content, pc)
