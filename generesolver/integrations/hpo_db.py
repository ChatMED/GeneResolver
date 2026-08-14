from __future__ import annotations

import logging
from functools import lru_cache

from pyhpo import Ontology, HPOSet

logger = logging.getLogger(__name__)

_ontology: Ontology | None = None
_gene_index: dict[str, object] | None = None

_ancestor_cache: dict[int, set[int]] = {}


def expand_with_ancestors(hpo_int_ids: set[int], max_depth: int = 3) -> set[int]:
    o = _get_ontology()
    result: set[int] = set(hpo_int_ids)
    for hid in hpo_int_ids:
        if hid in _ancestor_cache:
            result.update(_ancestor_cache[hid])
            continue
        ancestors: set[int] = set()
        current_level: set[int] = {hid}
        for _ in range(max_depth):
            next_level: set[int] = set()
            for cur_id in current_level:
                try:
                    term = o.get_hpo_object(int(cur_id))
                    for parent in (getattr(term, "parents", []) or []):
                        pid_str = str(getattr(parent, "id", parent))
                        try:
                            pid = int(pid_str.split(":")[1]) if ":" in pid_str else int(pid_str)
                        except (ValueError, IndexError):
                            continue
                        if pid not in ancestors:
                            next_level.add(pid)
                            ancestors.add(pid)
                except Exception:
                    continue
            if not next_level:
                break
            current_level = next_level
        _ancestor_cache[hid] = ancestors
        result.update(ancestors)
    return result


def _get_ontology() -> Ontology:
    global _ontology
    if _ontology is None:
        _ontology = Ontology()
    return _ontology


_PHENOTYPIC_ROOT_ID = "HP:0000118"


@lru_cache(maxsize=8192)
def is_phenotypic_abnormality(hpo_id: str) -> bool:
    try:
        term = _get_ontology().get_hpo_object(hpo_id)
    except Exception:
        return False
    if str(term.id) == _PHENOTYPIC_ROOT_ID:
        return True
    seen: set[str] = set()
    frontier = list(term.parents or [])
    while frontier:
        nxt = []
        for t in frontier:
            tid = str(getattr(t, "id", ""))
            if not tid or tid in seen:
                continue
            seen.add(tid)
            if tid == _PHENOTYPIC_ROOT_ID:
                return True
            nxt.extend(t.parents or [])
        frontier = nxt
    return False


@lru_cache(maxsize=4096)
def phenotypic_ancestors_with_labels(hpo_id: str) -> tuple[tuple[str, str], ...]:
    if not is_phenotypic_abnormality(hpo_id):
        return ()
    try:
        term = _get_ontology().get_hpo_object(hpo_id)
    except Exception:
        return ()
    out: dict[str, str] = {}
    seen: set[str] = {hpo_id}
    frontier = list(term.parents or [])
    while frontier:
        nxt = []
        for t in frontier:
            tid = str(getattr(t, "id", ""))
            if not tid or tid in seen:
                continue
            seen.add(tid)
            if tid == _PHENOTYPIC_ROOT_ID:
                continue
            if not is_phenotypic_abnormality(tid):
                continue
            out[tid] = str(getattr(t, "name", "") or "")
            nxt.extend(t.parents or [])
        frontier = nxt
    return tuple(out.items())


def get_term_omim_ic(hpo_id: str) -> float:
    if not hpo_id:
        return 0.0
    try:
        int_id = int(hpo_id.split(":")[1]) if ":" in hpo_id else int(hpo_id)
    except (ValueError, IndexError):
        return 0.0
    try:
        term = _get_ontology().get_hpo_object(int_id)
        ic = getattr(term, "information_content", None)
        if ic is None:
            return 0.0
        omim_val = getattr(ic, "omim", None)
        if omim_val is not None:
            return float(omim_val)
        if isinstance(ic, dict):
            return float(ic.get("omim", 0.0) or 0.0)
        return float(ic)
    except Exception:
        return 0.0


def get_gene_index() -> dict[str, object]:
    global _gene_index
    if _gene_index is None:
        o = _get_ontology()
        _gene_index = {str(g): g for g in o.genes}
    return _gene_index


def get_merged_gene_hpo_ids(gene_symbol: str) -> set[int]:
    from generesolver.integrations import hpo_a as _hpo_a
    index = get_gene_index()
    gene = index.get(gene_symbol)
    pyhpo_ids: set[int] = set(getattr(gene, "hpo", set())) if gene else set()
    extra_ids = _hpo_a.get_gene_hpo_map().get(gene_symbol, set())
    return pyhpo_ids | extra_ids


@lru_cache(maxsize=None)
def get_gene_hpo_set(gene_symbol: str) -> HPOSet | None:
    o = _get_ontology()
    all_ids = get_merged_gene_hpo_ids(gene_symbol)
    if not all_ids:
        return None
    try:
        terms = []
        for hpo_int_id in all_ids:
            try:
                terms.append(o.get_hpo_object(int(hpo_int_id)))
            except Exception:
                continue
        return HPOSet(terms) if terms else None
    except Exception as exc:
        logger.warning("Failed to build HPOSet for gene %s: %s", gene_symbol, exc)
        return None


def build_patient_hpo_set(patient_hpo_ids: list[str]) -> HPOSet | None:
    if not patient_hpo_ids:
        return None
    o = _get_ontology()
    terms = []
    for hid in patient_hpo_ids:
        try:
            terms.append(o.get_hpo_object(hid))
        except Exception:
            continue
    if not terms:
        return None
    try:
        return HPOSet(terms)
    except Exception as exc:
        logger.warning("Failed to build patient HPOSet: %s", exc)
        return None


def compute_similarity_with_set(patient_set: HPOSet | None, gene_symbol: str) -> float:
    if patient_set is None or len(patient_set) == 0:
        return 0.0
    gene_set = get_gene_hpo_set(gene_symbol)
    if gene_set is None or len(gene_set) == 0:
        return 0.0
    try:
        score = patient_set.similarity(gene_set, kind="omim", method="graphic")
        return max(0.0, float(score))
    except Exception as exc:
        logger.warning("HPO similarity failed for gene %s: %s", gene_symbol, exc)
        return 0.0


def compute_similarity(patient_hpo_ids: list[str], gene_symbol: str) -> float:
    if not patient_hpo_ids:
        return 0.0
    return compute_similarity_with_set(build_patient_hpo_set(patient_hpo_ids), gene_symbol)
