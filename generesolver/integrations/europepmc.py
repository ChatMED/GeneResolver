from __future__ import annotations

import logging
import re
import time

import httpx

from generesolver.config import EUROPEPMC_REST_URL
from generesolver.integrations._disk_cache import disk_cached

logger = logging.getLogger(__name__)
_REQUEST_INTERVAL = 0.3
_RETRY_ATTEMPTS = 3


@disk_cached("europepmc_search")
def search(query: str, page_size: int = 5, sort: str = "") -> list[dict]:
    params = {
        "query": query,
        "format": "json",
        "pageSize": page_size,
        "resultType": "core",
    }
    if sort and sort.lower() != "relevance":
        params["sort"] = sort
    with httpx.Client(timeout=20.0) as client:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = client.get(f"{EUROPEPMC_REST_URL}/search", params=params)
                resp.raise_for_status()
                results = resp.json().get("resultList", {}).get("result", [])
                time.sleep(_REQUEST_INTERVAL)
                return _normalize(results, page_size)
            except Exception as exc:
                if attempt < _RETRY_ATTEMPTS - 1:
                    time.sleep(1.0)
                else:
                    logger.debug("Europe PMC search failed for %r: %s", query, exc)
                    return []
    return []


_TAG = re.compile(r"<[^>]+>")


def _normalize(results: list[dict], page_size: int) -> list[dict]:
    out = []
    for r in results[:page_size]:
        abstract = _TAG.sub(" ", r.get("abstractText") or "")
        abstract = re.sub(r"\s+", " ", abstract).strip()
        out.append({
            "id": r.get("id"),
            "source": r.get("source"),
            "pmid": r.get("pmid"),
            "doi": r.get("doi"),
            "title": r.get("title") or "",
            "abstract": abstract[:1500],
        })
    return out


def citation(record: dict) -> str:
    src = record.get("source") or "?"
    if record.get("pmid"):
        return f"{src} PMID:{record['pmid']}"
    return f"{src} {record.get('id', '?')}"
