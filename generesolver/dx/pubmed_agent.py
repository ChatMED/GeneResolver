from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor

from generesolver.components import literature

logger = logging.getLogger(__name__)

_FETCH_TITLES = 6
_KEEP_TITLES = 2
_MAX_WORKERS = 6


def _one(name: str, patient_words: frozenset[str]) -> tuple[str, dict]:
    try:
        pmids = literature.pubmed_search(name)
        titles = literature.pubmed_titles(tuple(pmids), max_n=_FETCH_TITLES) if pmids else []
    except Exception as exc:
        logger.debug("pubmed_agent: %r failed: %s", name, exc)
        titles = []
    relevant = [t for t in titles
                if patient_words & set(re.findall(r"[a-z]{5,}", t.lower()))]
    return name, {"titles": relevant[:_KEEP_TITLES]}


def evidence_for(disease_names: list[str], patient_words: set[str] | None = None) -> dict[str, dict]:
    names = [n for n in dict.fromkeys(disease_names) if n]
    pw = frozenset(patient_words or ())
    if not names or not pw:
        return {n: {"titles": []} for n in names}
    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        for name, ev in pool.map(lambda n: _one(n, pw), names):
            out[name] = ev
    return out
