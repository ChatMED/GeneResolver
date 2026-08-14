from __future__ import annotations

import logging

from generesolver.integrations import hpo_a, hpo_lr, hpo_db, orphanet
from generesolver.schemas.case import CaseState

logger = logging.getLogger(__name__)

_METABOLISM_ROOT = 1939
MAX_DISEASES_PER_MARKER = 15

W_SPECIFICITY = 1.0
W_MARKER_COUNT_BONUS = 0.35
W_RARE_BONUS_HIGH = 0.75
W_RARE_BONUS_MID = 0.35
RARE_THRESHOLD_HIGH = 3
RARE_THRESHOLD_MID = 5


def _to_int(hpo_id: str) -> int | None:
    try:
        return int(str(hpo_id).split(":")[1])
    except (ValueError, IndexError):
        return None


def _is_metabolic(term_int: int) -> bool:
    if term_int == _METABOLISM_ROOT:
        return True
    try:
        obj = hpo_db._get_ontology().get_hpo_object(term_int)
    except Exception:
        return False
    for p in (getattr(obj, "all_parents", None) or []):
        if getattr(p, "index", None) == _METABOLISM_ROOT:
            return True
    return False


def _label(hpo_id: str) -> str:
    ti = _to_int(hpo_id)
    if ti is None:
        return ""
    try:
        return str(hpo_db._get_ontology().get_hpo_object(ti).name)
    except Exception:
        return ""


def _genes_for(disease_id: str) -> list[str]:
    genes = sorted(hpo_lr.genes_for_disease(disease_id) or [])
    if not genes and disease_id.upper().startswith("ORPHA"):
        try:
            genes = sorted({ga.gene_symbol for ga in orphanet.get_genes_for_disease(disease_id)})
        except Exception:
            genes = []
    return genes


def marker_diseases(
    present_ids: list[str],
    *,
    max_diseases: int = MAX_DISEASES_PER_MARKER,
) -> list[dict]:
    rev = hpo_a.get_hpo_to_diseases_map()
    names = hpo_lr.disease_names()
    by_disease: dict[str, dict] = {}

    for hid in present_ids or []:
        ti = _to_int(hid)
        if ti is None or not _is_metabolic(ti):
            continue

        diseases = rev.get(ti) or set()
        spec = len(diseases)

        if spec < 1 or spec > max_diseases:
            continue

        marker = {"hpo_id": hid, "label": _label(hid), "n_diseases": spec}

        for did in diseases:
            entry = by_disease.setdefault(
                did,
                {"markers": [], "best_specificity": 10**9},
            )
            entry["markers"].append(marker)
            entry["best_specificity"] = min(entry["best_specificity"], spec)

    def _biochemical_score(entry: dict) -> float:
        markers = entry.get("markers") or []
        if not markers:
            return 0.0

        specificity = W_SPECIFICITY * sum(1.0 / max(1, int(m["n_diseases"])) for m in markers)
        marker_bonus = W_MARKER_COUNT_BONUS * len({m["hpo_id"] for m in markers})

        best_spec = min(int(m["n_diseases"]) for m in markers)
        rare_marker_bonus = (
            W_RARE_BONUS_HIGH if best_spec <= RARE_THRESHOLD_HIGH else
            W_RARE_BONUS_MID if best_spec <= RARE_THRESHOLD_MID else
            0.0
        )

        return specificity + marker_bonus + rare_marker_bonus

    out: list[dict] = []
    for did, e in by_disease.items():
        score = _biochemical_score(e)
        out.append({
            "disease_id": did,
            "name": names.get(did, did),
            "genes": _genes_for(did),
            "markers": sorted(e["markers"], key=lambda m: m["n_diseases"]),
            "best_specificity": e["best_specificity"],
            "biochemical_score": round(score, 4),
            "source": "hpoa_marker_reverse_index",
        })

    out.sort(
        key=lambda d: (
            d.get("best_specificity", 10 ** 9),
            -len(d.get("markers", [])),
            d.get("name", ""),
            d.get("disease_id", ""),
        )
    )

    if out:
        logger.info(
            "biochemical: %d candidate disease(s) from %d specific metabolic marker(s)",
            len(out),
            len({m["hpo_id"] for d in out for m in d["markers"]}),
        )

    return out


def run(state: CaseState) -> CaseState:
    present_ids = state.hpo_profile.present_ids
    candidates = marker_diseases(present_ids) if present_ids else []
    diag = getattr(state, "diagnostics", None)
    if isinstance(diag, dict):
        diag["biochemical"] = {"candidates": candidates}
    state.mark_done("biochemical")
    return state
