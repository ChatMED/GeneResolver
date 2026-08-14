from __future__ import annotations

import logging
import time

import httpx

from generesolver.config import REACTOME_REST_URL
from generesolver.integrations._disk_cache import disk_cached

logger = logging.getLogger(__name__)
_REQUEST_INTERVAL = 0.3
_RETRY_ATTEMPTS = 3


@disk_cached("reactome_pathways_by_uniprot")
def get_pathways_for_uniprot(uniprot_id: str) -> list[str]:
    url = f"{REACTOME_REST_URL}/data/pathways/low/entity/{uniprot_id}/9606"
    return _fetch_pathway_names(url)


@disk_cached("reactome_pathways_by_gene")
def get_pathways_for_gene(gene_symbol: str) -> list[str]:
    url = f"{REACTOME_REST_URL}/data/mapping/UniProt/{gene_symbol}/pathways"
    names = _fetch_pathway_names(url)
    if names:
        return names
    return _search_by_gene(gene_symbol)


def _fetch_pathway_names(url: str) -> list[str]:
    with httpx.Client(timeout=20.0) as client:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = client.get(url, headers={"Accept": "application/json"})
                if resp.status_code == 404:
                    return []
                resp.raise_for_status()
                data = resp.json()
                time.sleep(_REQUEST_INTERVAL)
                if isinstance(data, list):
                    return [p.get("displayName") or p.get("name", "") for p in data if p.get("displayName") or p.get("name")]
                return []
            except Exception as exc:
                if attempt < _RETRY_ATTEMPTS - 1:
                    time.sleep(1.0)
                else:
                    logger.debug("Reactome pathway fetch failed for %s: %s", url, exc)
                    return []
    return []


def _search_by_gene(gene_symbol: str) -> list[str]:
    url = f"{REACTOME_REST_URL}/search/query?query={gene_symbol}&species=Homo+sapiens&types=Pathway&cluster=true"
    with httpx.Client(timeout=20.0) as client:
        try:
            resp = client.get(url, headers={"Accept": "application/json"})
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            names: list[str] = []
            for group in results:
                for entry in group.get("entries", [])[:5]:
                    name = entry.get("name") or entry.get("displayName")
                    if name:
                        names.append(name)
            return names[:10]
        except Exception:
            return []
