"""Journal screen: is the journal currently indexed for MEDLINE?

A record's own citation status says whether *that citation* has been indexed,
which lags for new papers ("In-Process", "Publisher"). Journal-level status
comes from the NLM Catalog (`currentlyindexed`), looked up by NLM Unique ID.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from nrag.net import CachedHttp


def _norm(nlm_id: str) -> str:
    # The catalog returns UIDs without leading zeros ("0372516" -> "372516").
    return nlm_id.strip().lstrip("0")


def currently_indexed(http: CachedHttp, cfg: dict[str, Any], nlm_ids: Iterable[str],
                      base_params: dict[str, str]) -> set[str]:
    """Return the subset of `nlm_ids` (normalised, no leading zeros) currently indexed for MEDLINE."""
    ids = sorted({_norm(i) for i in nlm_ids if i and _norm(i)})
    batch = cfg["journals"]["nlmcatalog_batch"]
    found: set[str] = set()
    for start in range(0, len(ids), batch):
        chunk = ids[start:start + batch]
        term = "(" + " OR ".join(f"{i}[nlmid]" for i in chunk) + ") AND currentlyindexed[All]"
        content = http.post(f"{cfg['ncbi']['eutils_base']}/esearch.fcgi", data={
            **base_params, "db": "nlmcatalog", "term": term, "retmode": "json", "retmax": len(chunk),
        }, ttl_days=cfg["journals"]["ttl_days"])
        found |= {_norm(i) for i in json.loads(content)["esearchresult"].get("idlist", [])}
    return found


def is_medline(rec: dict[str, Any], indexed: set[str]) -> bool:
    if rec.get("citation_status") == "MEDLINE":
        return True
    return bool(rec.get("nlm_unique_id")) and _norm(rec["nlm_unique_id"]) in indexed
