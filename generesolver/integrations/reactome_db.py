from __future__ import annotations

import logging
from pathlib import Path

from generesolver.config import DATA_DIR

logger = logging.getLogger(__name__)

_REACTOME_PATH = DATA_DIR / "reactome" / "NCBI2Reactome_All_Levels.txt"

_gene_to_pathways: dict[str, set[str]] | None = None
_pathway_to_genes: dict[str, set[str]] | None = None


def _build_entrez_to_symbol() -> dict[int, str]:
    from generesolver.integrations import hpo_a
    annotation_path = hpo_a._ANNOTATION_PATH
    if not annotation_path.exists():
        return {}

    result: dict[int, str] = {}
    with annotation_path.open(encoding="utf-8") as f:
        header_line = f.readline().rstrip("\n").lstrip("#").strip()
        header_cols = [h.strip().lower() for h in header_line.split("\t")]

        entrez_aliases = {"ncbi_gene_id", "entrez-gene-id", "entrez_id", "gene_id"}
        sym_aliases = {"gene_symbol", "entrez-gene-symbol", "gene-symbol", "symbol"}
        entrez_col = next((i for i, h in enumerate(header_cols) if h in entrez_aliases), None)
        sym_col = next((i for i, h in enumerate(header_cols) if h in sym_aliases), None)

        if entrez_col is None or sym_col is None:
            logger.warning(
                "HPO-A header missing entrez/symbol columns (got: %s). "
                "Pathway propagation disabled.",
                header_cols,
            )
            return {}

        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) <= max(entrez_col, sym_col):
                continue
            try:
                eid = int(cols[entrez_col])
            except ValueError:
                continue
            sym = cols[sym_col].strip()
            if sym:
                result[eid] = sym
    return result


def _load() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    if not _REACTOME_PATH.exists():
        logger.warning(
            "Reactome file not found at %s. Run scripts/download_reactome.py. "
            "Pathway propagation will return 0.0 for all genes.",
            _REACTOME_PATH,
        )
        return {}, {}

    entrez_to_symbol = _build_entrez_to_symbol()
    if not entrez_to_symbol:
        logger.warning(
            "Cannot map Entrez IDs to gene symbols (HPO-A not loaded). "
            "Pathway propagation disabled."
        )
        return {}, {}

    gene_to_pw: dict[str, set[str]] = {}
    pw_to_gene: dict[str, set[str]] = {}
    n_kept = 0
    n_skipped_species = 0

    with _REACTOME_PATH.open(encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 6:
                continue
            if cols[5].strip() != "Homo sapiens":
                n_skipped_species += 1
                continue
            try:
                eid = int(cols[0])
            except ValueError:
                continue
            pathway_id = cols[1].strip()
            sym = entrez_to_symbol.get(eid)
            if not sym or not pathway_id:
                continue
            gene_to_pw.setdefault(sym, set()).add(pathway_id)
            pw_to_gene.setdefault(pathway_id, set()).add(sym)
            n_kept += 1

    logger.info(
        "Reactome loaded: %d gene-pathway memberships (%d genes, %d pathways; "
        "skipped %d non-human)",
        n_kept, len(gene_to_pw), len(pw_to_gene), n_skipped_species,
    )
    return gene_to_pw, pw_to_gene


def _ensure_loaded() -> None:
    global _gene_to_pathways, _pathway_to_genes
    if _gene_to_pathways is None:
        _gene_to_pathways, _pathway_to_genes = _load()


def get_pathways_for_gene(gene_symbol: str) -> set[str]:
    _ensure_loaded()
    return _gene_to_pathways.get(gene_symbol, set())


def get_genes_for_pathway(pathway_id: str) -> set[str]:
    _ensure_loaded()
    return _pathway_to_genes.get(pathway_id, set())


def pathway_overlap_score(candidate: str, anchor_pathways: set[str]) -> float:
    candidate_pw = get_pathways_for_gene(candidate)
    if not candidate_pw or not anchor_pathways:
        return 0.0
    intersect = len(candidate_pw & anchor_pathways)
    union = len(candidate_pw | anchor_pathways)
    if union == 0:
        return 0.0
    return intersect / union


def build_anchor_pathway_set(anchor_symbols: list[str]) -> set[str]:
    _ensure_loaded()
    result: set[str] = set()
    for sym in anchor_symbols:
        result.update(_gene_to_pathways.get(sym, set()))
    return result
