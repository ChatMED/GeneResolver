from __future__ import annotations

import logging
import time

import httpx

from generesolver.config import OPEN_TARGETS_GRAPHQL_URL
from generesolver.integrations._disk_cache import disk_cached

logger = logging.getLogger(__name__)
_REQUEST_INTERVAL = 0.5
_RETRY_ATTEMPTS = 3

_SEARCH_QUERY = """
query SearchGene($q: String!) {
  search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 1}) {
    hits {
      id
      name
    }
  }
}
"""

_ASSOCIATIONS_QUERY = """
query GeneAssociations($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    associatedDiseases(page: {index: 0, size: 25}) {
      rows {
        disease {
          id
          name
        }
        score
      }
    }
  }
}
"""

_SEARCH_DISEASE_QUERY = """
query SearchDisease($q: String!) {
  search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: 1}) {
    hits {
      id
      name
    }
  }
}
"""

_DISEASE_TARGETS_QUERY = """
query DiseaseTargets($efoId: String!, $size: Int!) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(page: {index: 0, size: $size}) {
      rows {
        target {
          id
          approvedSymbol
        }
        score
      }
    }
  }
}
"""


@disk_cached("open_targets_disease_assoc")
def get_gene_disease_associations(gene_symbol: str) -> list[dict]:
    ensembl_id = _resolve_ensembl_id(gene_symbol)
    if not ensembl_id:
        return []
    return _fetch_associations(ensembl_id)


@disk_cached("open_targets_disease_targets")
def get_disease_associated_targets(disease_query: str, size: int = 50) -> list[dict]:
    if not disease_query:
        return []
    efo_id = _resolve_disease_id(disease_query)
    if not efo_id:
        return []
    return _fetch_disease_targets(efo_id, size)


def _resolve_disease_id(disease_query: str) -> str | None:
    q = disease_query.strip()
    if not q:
        return None
    if "_" in q and " " not in q:
        return q
    if q.upper().startswith("ORPHA:"):
        return "Orphanet_" + q.split(":", 1)[1]
    with httpx.Client(timeout=20.0) as client:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = client.post(
                    OPEN_TARGETS_GRAPHQL_URL,
                    json={"query": _SEARCH_DISEASE_QUERY, "variables": {"q": q}},
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                hits = (resp.json().get("data") or {}).get("search", {}).get("hits", [])
                if hits:
                    return hits[0]["id"]
                return None
            except Exception as exc:
                if attempt < _RETRY_ATTEMPTS - 1:
                    time.sleep(1.0)
                else:
                    logger.warning("Open Targets disease lookup failed for %s: %s", q, exc)
                    return None
    return None


def _fetch_disease_targets(efo_id: str, size: int) -> list[dict]:
    with httpx.Client(timeout=20.0) as client:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = client.post(
                    OPEN_TARGETS_GRAPHQL_URL,
                    json={
                        "query": _DISEASE_TARGETS_QUERY,
                        "variables": {"efoId": efo_id, "size": size},
                    },
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                disease = (resp.json().get("data") or {}).get("disease") or {}
                rows = (disease.get("associatedTargets") or {}).get("rows", [])
                time.sleep(_REQUEST_INTERVAL)
                out: list[dict] = []
                for r in rows:
                    sym = (r.get("target") or {}).get("approvedSymbol")
                    if sym:
                        out.append({"gene_symbol": sym, "score": r.get("score", 0.0)})
                return out
            except Exception as exc:
                if attempt < _RETRY_ATTEMPTS - 1:
                    time.sleep(1.0)
                else:
                    logger.warning("Open Targets disease targets failed for %s: %s", efo_id, exc)
                    return []
    return []


def _resolve_ensembl_id(gene_symbol: str) -> str | None:
    with httpx.Client(timeout=20.0) as client:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = client.post(
                    OPEN_TARGETS_GRAPHQL_URL,
                    json={"query": _SEARCH_QUERY, "variables": {"q": gene_symbol}},
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                hits = (resp.json().get("data") or {}).get("search", {}).get("hits", [])
                if hits:
                    return hits[0]["id"]
                return None
            except Exception as exc:
                if attempt < _RETRY_ATTEMPTS - 1:
                    time.sleep(1.0)
                else:
                    logger.warning("Open Targets gene lookup failed for %s: %s", gene_symbol, exc)
                    return None
    return None


def _fetch_associations(ensembl_id: str) -> list[dict]:
    with httpx.Client(timeout=20.0) as client:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = client.post(
                    OPEN_TARGETS_GRAPHQL_URL,
                    json={"query": _ASSOCIATIONS_QUERY, "variables": {"ensemblId": ensembl_id}},
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                target = (resp.json().get("data") or {}).get("target", {})
                rows = (target.get("associatedDiseases") or {}).get("rows", [])
                time.sleep(_REQUEST_INTERVAL)
                return [
                    {
                        "disease_id": r["disease"]["id"],
                        "disease_name": r["disease"]["name"],
                        "score": r.get("score", 0.0),
                    }
                    for r in rows
                ]
            except Exception as exc:
                if attempt < _RETRY_ATTEMPTS - 1:
                    time.sleep(1.0)
                else:
                    logger.warning("Open Targets associations failed for %s: %s", ensembl_id, exc)
                    return []
    return []
