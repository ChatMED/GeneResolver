from __future__ import annotations

import time
import logging

import httpx

from generesolver.config import CLINVAR_EUTILS_URL

logger = logging.getLogger(__name__)

_REQUEST_INTERVAL = 0.34
_RETRY_ATTEMPTS = 3


def _get_api_key() -> str | None:
    import os
    return os.environ.get("NCBI_API_KEY")


def lookup_by_rsid(rsid: str) -> dict | None:
    return _search(f"{rsid}[RS]")


def lookup_all_by_rsid(rsid: str) -> list[dict]:
    return _search_multi(f"{rsid}[RS]")


def lookup_by_hgvs(hgvs: str) -> dict | None:
    return _search(f"{hgvs}[HGVS]")


def lookup_by_gene_and_position(gene: str, chrom: str, pos: int) -> list[dict]:
    query = f"{gene}[GENE] AND {chrom}[CHR] AND {pos - 5}:{pos + 5}[CHRPOS]"
    return _search_multi(query)


def _search(query: str) -> dict | None:
    results = _search_multi(query)
    return results[0] if results else None


def _search_multi(query: str) -> list[dict]:
    api_key = _get_api_key()
    params: dict = {"db": "clinvar", "term": query, "retmax": 10, "retmode": "json"}
    if api_key:
        params["api_key"] = api_key

    with httpx.Client(timeout=30.0) as client:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = client.get(f"{CLINVAR_EUTILS_URL}/esearch.fcgi", params=params)
                resp.raise_for_status()
                data = resp.json()
                ids = data.get("esearchresult", {}).get("idlist", [])
                if not ids:
                    return []
                time.sleep(_REQUEST_INTERVAL)
                return _fetch_summaries(client, ids, api_key)
            except Exception as exc:
                if attempt < _RETRY_ATTEMPTS - 1:
                    time.sleep(1.0)
                else:
                    logger.error("ClinVar search failed for query '%s': %s", query, exc)
                    raise RuntimeError(f"ClinVar search failed for {query}: {exc}") from exc
    return []


def _fetch_summaries(client: httpx.Client, ids: list[str], api_key: str | None) -> list[dict]:
    params: dict = {"db": "clinvar", "id": ",".join(ids), "retmode": "json"}
    if api_key:
        params["api_key"] = api_key

    try:
        resp = client.get(f"{CLINVAR_EUTILS_URL}/esummary.fcgi", params=params)
        resp.raise_for_status()
        raw = resp.json()
        result = raw.get("result", {})
        summaries = []
        for uid in result.get("uids", []):
            entry = result.get(uid, {})
            trait_set = entry.get("germline_classification", {}).get("trait_set", [])
            summaries.append({
                "clinvar_id": f"VCV{int(uid):09d}",
                "title": entry.get("title", ""),
                "clinical_significance": _extract_significance(entry),
                "review_status": entry.get("review_status", ""),
                "conditions": [t.get("trait_name", "") for t in trait_set if t.get("trait_name")],
                "gene_symbols": [g.get("symbol", "") for g in entry.get("genes", [])],
            })
        return summaries
    except Exception as exc:
        logger.error("ClinVar summary fetch failed: %s", exc)
        raise RuntimeError(f"ClinVar summary fetch failed: {exc}") from exc


def _extract_significance(entry: dict) -> str:
    germline = entry.get("germline_classification", {})
    if isinstance(germline, dict):
        return germline.get("description", entry.get("clinical_significance", ""))
    return entry.get("clinical_significance", "")
