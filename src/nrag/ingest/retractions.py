"""Retraction and expression-of-concern checks beyond PubMed's own flags.

Uses Crossref's distribution of the Retraction Watch database (one CSV, cached
for `retractions.ttl_days`), matched by DOI and PMID. The whole corpus is
rechecked on every run, so a paper retracted after ingestion leaves the corpus
and is logged by the pipeline.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from nrag.net import CachedHttp


@dataclass
class Notice:
    nature: str          # Retraction | Expression of concern | Correction | Reinstatement
    date: str            # ISO date of the notice
    notice_doi: str | None
    reason: str


@dataclass
class RetractionIndex:
    by_doi: dict[str, list[Notice]] = field(default_factory=dict)
    by_pmid: dict[str, list[Notice]] = field(default_factory=dict)

    @classmethod
    def from_csv(cls, content: bytes) -> "RetractionIndex":
        idx = cls()
        reader = csv.DictReader(io.StringIO(content.decode("utf-8", errors="replace")))
        for row in reader:
            n = Notice(
                nature=(row.get("RetractionNature") or "").strip(),
                date=_iso(row.get("RetractionDate") or ""),
                notice_doi=(row.get("RetractionDOI") or "").strip().lower() or None,
                reason=(row.get("Reason") or "").strip(),
            )
            doi = (row.get("OriginalPaperDOI") or "").strip().lower()
            pmid = (row.get("OriginalPaperPubMedID") or "").strip()
            if doi and doi not in {"unavailable", "0"}:
                idx.by_doi.setdefault(doi, []).append(n)
            if pmid and pmid != "0":
                idx.by_pmid.setdefault(pmid, []).append(n)
        return idx

    def lookup(self, doi: str | None, pmid: str | None) -> dict[str, Any]:
        notices = {(n.nature, n.date, n.notice_doi): n
                   for n in self.by_doi.get(doi or "", []) + self.by_pmid.get(pmid or "", [])}
        ordered = sorted(notices.values(), key=lambda n: n.date)
        return {
            "retracted": _currently_retracted(ordered),
            "expression_of_concern": any(n.nature.lower() == "expression of concern" for n in ordered),
            "notices": [{"nature": n.nature, "date": n.date, "notice_doi": n.notice_doi} for n in ordered],
        }

    def __len__(self) -> int:
        return len(self.by_doi) + len(self.by_pmid)


def _currently_retracted(ordered: list[Notice]) -> bool:
    """Retracted unless a later reinstatement reverses it."""
    state = False
    for n in ordered:
        nature = n.nature.lower()
        if nature == "retraction":
            state = True
        elif nature == "reinstatement":
            state = False
    return state


def _iso(s: str) -> str:
    s = s.strip()
    for fmt in ("%m/%d/%Y %H:%M", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return s


def load_retraction_index(http: CachedHttp, cfg: dict[str, Any]) -> RetractionIndex:
    rc = cfg["retractions"]
    return RetractionIndex.from_csv(http.get(rc["retraction_watch_csv"], ttl_days=rc["ttl_days"]))
