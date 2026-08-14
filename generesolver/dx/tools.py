from __future__ import annotations

import re
from functools import lru_cache

from generesolver.integrations import hpo_a, hpo_lr, hpo_db

_GENE_ALIASES = {"MUT": "MMUT", "SEPT9": "SEPTIN9", "ADCK3": "COQ8A", "C12orf65": "MTRFR"}

_GENERIC_TOKEN_DOC_FREQ = 0.02


def _norm(s: str) -> str:
    s = re.sub(r"\(.*?\)", " ", (s or "").lower())
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\bacidemia\b", "aciduria", s)
    return s


def _label_int(hi: int) -> str:
    try:
        return str(hpo_db._get_ontology().get_hpo_object(hi).name)
    except Exception:
        return f"HP:{hi:07d}"


@lru_cache(maxsize=1)
def _token_doc_freq() -> dict[str, float]:
    names = hpo_lr.disease_names()
    n = max(1, len(names))
    counts: dict[str, int] = {}
    for name in names.values():
        for tok in set(_norm(name).split()):
            counts[tok] = counts.get(tok, 0) + 1
    return {tok: c / n for tok, c in counts.items()}


def _content_tokens(tokens: set[str]) -> set[str]:
    df = _token_doc_freq()
    return {t for t in tokens if df.get(t, 0.0) < _GENERIC_TOKEN_DOC_FREQ}


def search_disease(query: str, limit: int = 8) -> list[dict]:
    names = hpo_lr.disease_names()
    d2g, _ = hpo_lr._load_disease_gene_maps()
    q = _norm(query)
    if not q:
        return []
    qtok = set(q.split())
    q_content = _content_tokens(qtok)
    if not q_content:
        q_content = qtok

    scored: list[tuple[tuple[int, float], str, str]] = []
    for did, name in names.items():
        n = _norm(name)
        if not n:
            continue
        is_substring = q in n or n in q
        ntok = set(n.split())
        shared_content = q_content & ntok
        overlap_ratio = len(shared_content) / len(q_content) if q_content else 0.0
        if not is_substring and overlap_ratio == 0.0:
            continue
        scored.append(((1 if is_substring else 0, overlap_ratio), did, name))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"disease_id": did, "name": name, "genes": sorted(d2g.get(did, set()))}
            for _, did, name in scored[:limit]]


def get_disease_profile(disease_id: str, present_hpo_ints: set[int] | None = None,
                        absent_hpo_ints: set[int] | None = None, max_other_terms: int = 30) -> dict:
    names = hpo_lr.disease_names()
    d2g, _ = hpo_lr._load_disease_gene_maps()
    dz_map = hpo_a.get_disease_hpo_map()
    disease_ints = dz_map.get(disease_id, set())

    present_hpo_ints = present_hpo_ints or set()
    absent_hpo_ints = absent_hpo_ints or set()

    matched = sorted(_label_int(hi) for hi in (disease_ints & present_hpo_ints))
    contradicted = sorted(_label_int(hi) for hi in (disease_ints & absent_hpo_ints))
    remaining_ints = disease_ints - present_hpo_ints - absent_hpo_ints
    other = sorted(_label_int(hi) for hi in remaining_ints)[:max_other_terms]

    return {
        "disease_id": disease_id,
        "name": names.get(disease_id, disease_id),
        "genes": sorted(d2g.get(disease_id, set())),
        "matched_present_findings": matched,
        "contradicted_absent_findings": contradicted,
        "n_present_matched": len(matched),
        "n_absent_contradicted": len(contradicted),
        "other_known_findings": other,
    }


def search_by_gene(gene_symbol: str, limit: int = 12) -> list[dict]:
    _, g2d = hpo_lr._load_disease_gene_maps()
    names = hpo_lr.disease_names()
    g = (gene_symbol or "").strip().upper()
    dz = g2d.get(g) or g2d.get(_GENE_ALIASES.get(g, ""), set())
    return [{"disease_id": d, "name": names.get(d, d)} for d in sorted(dz)][:limit]

TOOLS = {
    "search_disease": lambda a: search_disease(a.get("query", "")),
    "search_by_gene": lambda a: search_by_gene(a.get("gene", "")),
}


def build_case_tools(present_hpo_ints: set[int], absent_hpo_ints: set[int]) -> dict:
    return {
        **TOOLS,
        "get_disease_profile": lambda a: get_disease_profile(
            a.get("disease_id", ""), present_hpo_ints, absent_hpo_ints),
    }