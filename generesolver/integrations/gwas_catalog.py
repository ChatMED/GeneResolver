from __future__ import annotations

import logging
import time

import httpx

logger = logging.getLogger(__name__)
_GWAS_CATALOG_URL = "https://www.ebi.ac.uk/gwas/rest/api"
_REQUEST_INTERVAL = 0.5
_RETRY_ATTEMPTS = 3

_DISCLAIMER = (
    "GWAS evidence is population-level and does not establish "
    "Mendelian causality for this individual patient."
)


def lookup(rsid: str | None = None, gene: str | None = None) -> list[dict]:
    if rsid:
        return _gwas_catalog_by_rsid(rsid)
    if gene:
        return _gwas_catalog_by_gene(gene)
    return []


def _gwas_catalog_by_rsid(rsid: str) -> list[dict]:
    url = f"{_GWAS_CATALOG_URL}/singleNucleotidePolymorphisms/{rsid}/associations"
    return _fetch_catalog(url, rsid)


def _gwas_catalog_by_gene(gene: str) -> list[dict]:
    url = f"{_GWAS_CATALOG_URL}/singleNucleotidePolymorphisms/search/findByGene?geneName={gene}"
    return _fetch_catalog(url, gene)


def _fetch_catalog(url: str, query: str) -> list[dict]:
    with httpx.Client(timeout=20.0) as client:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = client.get(url, headers={"Accept": "application/json"})
                if resp.status_code == 404:
                    return []
                resp.raise_for_status()
                data = resp.json()
                time.sleep(_REQUEST_INTERVAL)

                associations = (
                    data.get("_embedded", {}).get("associations", [])
                    or data.get("_embedded", {}).get("singleNucleotidePolymorphisms", [])
                    or []
                )
                results: list[dict] = []
                for assoc in associations[:10]:
                    trait = assoc.get("efoTraits", [{}])[0].get("trait") if assoc.get("efoTraits") else None
                    results.append({
                        "query": query,
                        "trait": trait or "unknown",
                        "p_value": assoc.get("pvalue"),
                        "odds_ratio": assoc.get("orPerCopyNum"),
                        "source_db": "EBI GWAS Catalog",
                        "support_level": "supportive_context_only",
                        "warning": _DISCLAIMER,
                    })
                return results
            except Exception as exc:
                if attempt < _RETRY_ATTEMPTS - 1:
                    time.sleep(1.0)
                else:
                    logger.debug("GWAS Catalog fetch failed for %s: %s", query, exc)
                    return []
    return []
