from __future__ import annotations

import logging
import time

import httpx

logger = logging.getLogger(__name__)
_GWAS_ASSOCIATIONS_URL = "https://www.ebi.ac.uk/gwas/rest/api/v2/associations"
_REQUEST_INTERVAL = 0.5
_RETRY_ATTEMPTS = 3

_DISCLAIMER = (
    "GWAS evidence is population-level and does not establish "
    "Mendelian causality for this individual patient."
)


def lookup(rsid: str | None = None, gene: str | None = None) -> list[dict]:
    if rsid:
        return _fetch_catalog({"rs_id": rsid}, rsid)
    if gene:
        return _fetch_catalog({"mapped_gene": gene}, gene)
    return []


def _fetch_catalog(filters: dict[str, str], query: str) -> list[dict]:
    params: dict[str, str | int] = {**filters, "page": 0, "size": 10}
    with httpx.Client(timeout=20.0) as client:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = client.get(
                    _GWAS_ASSOCIATIONS_URL,
                    params=params,
                    headers={"Accept": "application/json", "User-Agent": "GeneResolver/1.0"},
                )
                resp.raise_for_status()
                data = resp.json()
                associations = data.get("_embedded", {}).get("associations", [])
                if not isinstance(associations, list):
                    raise ValueError("GWAS Catalog v2 returned an unexpected associations payload")
                time.sleep(_REQUEST_INTERVAL)
                return [_parse_association(assoc, query) for assoc in associations[:10]]
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status < 500 and status != 429:
                    raise RuntimeError(f"GWAS Catalog v2 returned HTTP {status} for {query}") from exc
                if attempt == _RETRY_ATTEMPTS - 1:
                    raise RuntimeError(f"GWAS Catalog v2 failed for {query}: HTTP {status}") from exc
                time.sleep(1.0 * (attempt + 1))
            except (httpx.RequestError, ValueError) as exc:
                if attempt == _RETRY_ATTEMPTS - 1:
                    raise RuntimeError(f"GWAS Catalog v2 failed for {query}: {exc}") from exc
                time.sleep(1.0 * (attempt + 1))
    return []


def _parse_association(assoc: dict, query: str) -> dict:
    efo_traits = assoc.get("efo_traits") or []
    reported_traits = assoc.get("reported_trait") or []
    trait = (
        efo_traits[0].get("efo_trait") if efo_traits and isinstance(efo_traits[0], dict)
        else reported_traits[0] if reported_traits else None
    )
    return {
        "query": query,
        "trait": trait or "unknown",
        "p_value": assoc.get("p_value"),
        "odds_ratio": assoc.get("or_per_copy_num"),
        "source_db": "EBI GWAS Catalog v2",
        "support_level": "supportive_context_only",
        "warning": _DISCLAIMER,
    }
