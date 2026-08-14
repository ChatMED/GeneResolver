from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

import httpx

from generesolver.config import PUBMED_EUTILS_URL
from generesolver.integrations._disk_cache import disk_cached
from generesolver.schemas.case import CaseState
from generesolver.schemas.evidence import LiteratureEvidence

logger = logging.getLogger(__name__)
_RETRY_ATTEMPTS = 3
_PUBMED_RETMAX = 10
_MAX_PUBMED_WORKERS = 3

_NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "").strip()
if _NCBI_API_KEY:
    _MAX_PUBMED_WORKERS = 8


def _with_api_key(params: dict) -> dict:
    return {**params, "api_key": _NCBI_API_KEY} if _NCBI_API_KEY else params


def run(state: CaseState) -> CaseState:
    genes: list[str] = [gr.gene for gr in state.ranked_genes]
    if state.input.candidate_gene and state.input.candidate_gene not in genes:
        genes.insert(0, state.input.candidate_gene)

    if not genes:
        state.mark_done("literature")
        return state

    existing = {le.gene for le in state.literature_evidence}
    new_evidence: list[LiteratureEvidence] = list(state.literature_evidence)
    pending = [g for g in genes if g not in existing]

    if pending:
        with ThreadPoolExecutor(max_workers=_MAX_PUBMED_WORKERS) as pool:
            for le in pool.map(_search_gene, pending):
                new_evidence.append(le)

    state.literature_evidence = new_evidence
    state.mark_done("literature")
    return state


def _search_gene(gene: str) -> LiteratureEvidence:
    query = (
        f"{gene}[Gene] AND "
        f"(disease[TIAB] OR pathogenic[TIAB] OR variant[TIAB] OR disorder[TIAB])"
    )
    pmids = _esearch(query)
    return LiteratureEvidence(
        gene=gene,
        query_terms=[gene, "disease", "pathogenic", "variant"],
        pmids=pmids,
        support_level=_grade(len(pmids)),
    )


@disk_cached("pubmed_esearch")
def pubmed_search(query: str, sort: str = "") -> list[str]:
    url = f"{PUBMED_EUTILS_URL}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": _PUBMED_RETMAX,
        "retmode": "json",
    }
    if sort:
        params["sort"] = sort
    with httpx.Client(timeout=20.0) as client:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = client.get(url, params=_with_api_key(params))
                resp.raise_for_status()
                return resp.json().get("esearchresult", {}).get("idlist", [])
            except Exception as exc:
                if attempt < _RETRY_ATTEMPTS - 1:
                    pass
                else:
                    logger.debug("PubMed esearch failed for %r: %s", query, exc)
                    return []
    return []


@disk_cached("pubmed_titles")
def pubmed_titles(pmids: tuple[str, ...], max_n: int = 3) -> list[str]:
    ids = [p for p in (pmids or ()) if p][:max_n]
    if not ids:
        return []
    url = f"{PUBMED_EUTILS_URL}/esummary.fcgi"
    params = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
    with httpx.Client(timeout=20.0) as client:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = client.get(url, params=_with_api_key(params))
                resp.raise_for_status()
                result = resp.json().get("result", {})
                out = []
                for pid in ids:
                    title = (result.get(pid, {}) or {}).get("title")
                    if title:
                        out.append(title.strip().rstrip("."))
                return out
            except Exception as exc:
                if attempt >= _RETRY_ATTEMPTS - 1:
                    logger.debug("PubMed esummary failed for %r: %s", ids, exc)
                    return []
    return []


@disk_cached("pubmed_abstracts")
def pubmed_abstracts(pmids: tuple[str, ...], max_n: int = 2, max_chars: int = 2000) -> list[str]:
    ids = [p for p in (pmids or ()) if p][:max_n]
    if not ids:
        return []
    url = f"{PUBMED_EUTILS_URL}/efetch.fcgi"
    params = {"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "text"}
    with httpx.Client(timeout=30.0) as client:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = client.get(url, params=_with_api_key(params))
                resp.raise_for_status()
                blocks = [b.strip() for b in re.split(r"\n\s*\n\s*\n+", resp.text.strip()) if b.strip()]
                out = []
                for b in blocks[:max_n]:
                    b = re.sub(r"\s+", " ", b).strip()
                    out.append(b[:max_chars])
                return out
            except Exception as exc:
                if attempt >= _RETRY_ATTEMPTS - 1:
                    logger.debug("PubMed efetch failed for %r: %s", ids, exc)
                    return []
    return []

_esearch = pubmed_search


def _grade(count: int) -> str:
    if count >= 5:
        return "strong"
    if count >= 2:
        return "moderate"
    if count >= 1:
        return "weak"
    return "none"
