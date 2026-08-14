from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from generesolver.config import DATA_DIR

logger = logging.getLogger(__name__)

_ANNOTATION_PATH = DATA_DIR / "hpo_a" / "phenotype_to_genes.txt"
_gene_hpo_map: dict[str, set[int]] | None = None
_disease_hpo_map: dict[str, set[int]] | None = None
_hpo_to_diseases_map: dict[int, set[str]] | None = None


@dataclass(frozen=True)
class CandidateDisease:
    orpha_id: str
    label: str
    overlap: int
    matched_hpo_ids: tuple[str, ...]


def _parse_hpo_int(hpo_id: str) -> int | None:
    try:
        return int(hpo_id.split(":")[1])
    except (ValueError, IndexError, AttributeError):
        return None


def _load() -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    if not _ANNOTATION_PATH.exists():
        logger.warning(
            "HPO-A annotation file not found at %s. Run "
            "scripts/download_hpo_annotations.py to fetch it. "
            "Falling back to PyHPO's built-in gene index only.",
            _ANNOTATION_PATH,
        )
        return {}, {}

    gene_map: dict[str, set[int]] = {}
    disease_map: dict[str, set[int]] = {}
    with _ANNOTATION_PATH.open(encoding="utf-8") as f:
        header_line = f.readline().rstrip("\n").lstrip("#").strip()
        header_cols = [h.strip().lower() for h in header_line.split("\t")]

        hpo_aliases = {"hpo_id", "hpo-id", "hpoid"}
        sym_aliases = {"gene_symbol", "entrez-gene-symbol", "gene-symbol", "symbol"}
        dz_aliases  = {"disease_id", "disease-id", "diseaseid"}
        hpo_col = next((i for i, h in enumerate(header_cols) if h in hpo_aliases), None)
        sym_col = next((i for i, h in enumerate(header_cols) if h in sym_aliases), None)
        dz_col  = next((i for i, h in enumerate(header_cols) if h in dz_aliases),  None)

        if hpo_col is None or sym_col is None:
            logger.warning(
                "HPO-A header missing expected columns (got: %s). Aborting load.",
                header_cols,
            )
            return {}, {}

        for line in f:
            cols = line.rstrip("\n").split("\t")
            max_needed = max(hpo_col, sym_col, dz_col if dz_col is not None else 0)
            if len(cols) <= max_needed:
                continue
            hpo_int = _parse_hpo_int(cols[hpo_col])
            if hpo_int is None:
                continue
            sym = cols[sym_col].strip()
            if sym:
                gene_map.setdefault(sym, set()).add(hpo_int)
            if dz_col is not None:
                dz = cols[dz_col].strip()
                if dz:
                    disease_map.setdefault(dz, set()).add(hpo_int)

    n_gene_anns = sum(len(v) for v in gene_map.values())
    n_dz_anns = sum(len(v) for v in disease_map.values())
    logger.info(
        "HPO-A loaded: %d genes (%d gene annotations), %d diseases "
        "(%d disease annotations) from %s",
        len(gene_map), n_gene_anns, len(disease_map), n_dz_anns,
        _ANNOTATION_PATH.name,
    )
    return gene_map, disease_map


def _ensure_loaded() -> None:
    global _gene_hpo_map, _disease_hpo_map
    if _gene_hpo_map is None or _disease_hpo_map is None:
        _gene_hpo_map, _disease_hpo_map = _load()


def get_gene_hpo_map() -> dict[str, set[int]]:
    _ensure_loaded()
    assert _gene_hpo_map is not None
    return _gene_hpo_map


def get_disease_hpo_map() -> dict[str, set[int]]:
    _ensure_loaded()
    assert _disease_hpo_map is not None
    return _disease_hpo_map


def get_hpo_to_diseases_map() -> dict[int, set[str]]:
    global _hpo_to_diseases_map
    if _hpo_to_diseases_map is not None:
        return _hpo_to_diseases_map
    dz_map = get_disease_hpo_map()
    out: dict[int, set[str]] = {}
    for dz_id, hpo_ints in dz_map.items():
        for hint in hpo_ints:
            out.setdefault(hint, set()).add(dz_id)
    _hpo_to_diseases_map = out
    return out


def candidate_diseases_for_hpo_profile(
    patient_hpo_ids: list[str] | set[str],
    *,
    min_overlap: int = 3,
    top_n: int = 15,
) -> list[CandidateDisease]:
    if not patient_hpo_ids:
        return []

    from generesolver.integrations import orphanet

    patient_ints: set[int] = set()
    for hid in patient_hpo_ids:
        try:
            patient_ints.add(int(str(hid).split(":")[1]))
        except (ValueError, IndexError):
            continue
    if not patient_ints:
        return []

    rev = get_hpo_to_diseases_map()
    per_orpha_hits: dict[str, set[int]] = {}
    for hint in patient_ints:
        for dz_id in rev.get(hint, ()):
            if dz_id.startswith("ORPHA:"):
                orpha_id = dz_id
                disease = orphanet.get_by_orpha_id(orpha_id)
                if disease is None:
                    continue
            elif dz_id.startswith("OMIM:"):
                disease = orphanet.get_orpha_by_omim(dz_id)
                if disease is None:
                    continue
                orpha_id = disease.orpha_id
            else:
                continue
            per_orpha_hits.setdefault(orpha_id, set()).add(hint)

    out: list[CandidateDisease] = []
    for orpha_id, hits in per_orpha_hits.items():
        if len(hits) < min_overlap:
            continue
        disease = orphanet.get_by_orpha_id(orpha_id)
        if disease is None:
            continue
        matched_strs = tuple(sorted(f"HP:{h:07d}" for h in hits))
        out.append(CandidateDisease(
            orpha_id=orpha_id,
            label=disease.label,
            overlap=len(hits),
            matched_hpo_ids=matched_strs,
        ))

    out.sort(key=lambda c: (-c.overlap, c.label))
    return out[:top_n]


def get_hpos_for_disease(orpha_id: str) -> set[str]:
    if not orpha_id or not orpha_id.startswith("ORPHA:"):
        return set()
    from generesolver.integrations import orphanet

    targets: set[str] = {orpha_id}
    disease = orphanet.get_by_orpha_id(orpha_id)
    if disease is not None:
        for omim in disease.omim_ids:
            targets.add(f"OMIM:{omim}")

    dz_map = get_disease_hpo_map()
    merged_int_ids: set[int] = set()
    for tid in targets:
        merged_int_ids |= dz_map.get(tid, set())

    return {f"HP:{i:07d}" for i in merged_int_ids}


def augmented_gene_set(gene_symbol: str, pyhpo_set: set[int]) -> set[int]:
    extra = get_gene_hpo_map().get(gene_symbol, set())
    return pyhpo_set | extra
