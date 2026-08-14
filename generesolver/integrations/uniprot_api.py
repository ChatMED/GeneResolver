from __future__ import annotations

import logging
import time

import httpx

from generesolver.config import UNIPROT_REST_URL
from generesolver.integrations._disk_cache import disk_cached

logger = logging.getLogger(__name__)
_REQUEST_INTERVAL = 0.3
_RETRY_ATTEMPTS = 3


@disk_cached("uniprot_protein_info")
def get_protein_info(gene_symbol: str) -> dict:
    entry = _search_by_gene(gene_symbol)
    if not entry:
        return {}

    return {
        "uniprot_id": entry.get("primaryAccession"),
        "function": _extract_function(entry),
        "subcellular_location": _extract_location(entry),
        "disease_involvement": _extract_diseases(entry),
        "go_terms": _extract_go_terms(entry),
    }


_FUNCTIONAL_FEATURE_TYPES = {"domain", "region", "active site", "binding site", "dna binding"}


def get_functional_regions(gene_symbol: str) -> list[dict]:
    entry = _search_by_gene(gene_symbol)
    if not entry:
        return []
    regions: list[dict] = []
    for feature in entry.get("features", []):
        ftype = (feature.get("type") or "").strip().lower()
        if ftype not in _FUNCTIONAL_FEATURE_TYPES:
            continue
        location = feature.get("location", {})
        start = (location.get("start") or {}).get("value")
        end = (location.get("end") or {}).get("value")
        if start is None or end is None:
            continue
        regions.append({
            "type": feature.get("type"),
            "start": int(start),
            "end": int(end),
            "description": feature.get("description") or "",
        })
    return regions


@disk_cached("uniprot_entry")
def _search_by_gene(gene_symbol: str) -> dict | None:
    params = {
        "query": f"gene_exact:{gene_symbol} AND organism_id:9606 AND reviewed:true",
        "format": "json",
        "fields": "accession,gene_names,protein_name,cc_function,cc_subcellular_location,"
                  "cc_disease,go,ft_domain,ft_region,ft_act_site,ft_binding,ft_dna_bind",
        "size": 1,
    }
    with httpx.Client(timeout=20.0) as client:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = client.get(f"{UNIPROT_REST_URL}/search", params=params)
                resp.raise_for_status()
                results = resp.json().get("results", [])
                time.sleep(_REQUEST_INTERVAL)
                return results[0] if results else None
            except Exception as exc:
                if attempt < _RETRY_ATTEMPTS - 1:
                    time.sleep(1.0)
                else:
                    logger.warning("UniProt lookup failed for %s: %s", gene_symbol, exc)
                    return None
    return None


def _extract_function(entry: dict) -> str:
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "FUNCTION":
            texts = comment.get("texts", [])
            if texts:
                return texts[0].get("value", "")
    return ""


def _extract_location(entry: dict) -> list[str]:
    locations: list[str] = []
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "SUBCELLULAR LOCATION":
            for loc in comment.get("subcellularLocations", []):
                location = loc.get("location")
                if isinstance(location, dict):
                    name = location.get("value")
                elif isinstance(location, str):
                    name = location
                else:
                    name = None
                if name:
                    locations.append(name)
    return locations


def _extract_diseases(entry: dict) -> list[str]:
    diseases: list[str] = []
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "DISEASE":
            d = comment.get("disease", {})
            name = d.get("diseaseName")
            if not name:
                disease_id = d.get("diseaseId", {})
                if isinstance(disease_id, dict):
                    name = disease_id.get("value")
                elif isinstance(disease_id, str):
                    name = disease_id
            if name:
                diseases.append(name)
    return diseases


def _extract_go_terms(entry: dict) -> list[str]:
    go: list[str] = []
    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") == "GO":
            for prop in xref.get("properties", []):
                if prop.get("key") == "GoTerm":
                    go.append(prop.get("value", "").lstrip("FCP:").strip())
    return go[:20]
