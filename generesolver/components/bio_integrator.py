from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from generesolver.schemas.case import CaseState
from generesolver.schemas.evidence import BiologicalEvidence
from generesolver.integrations import open_targets_api, uniprot_api, reactome_api

logger = logging.getLogger(__name__)

_MAX_GENE_WORKERS = 8


def _fetch_for_gene(gene: str) -> BiologicalEvidence:
    try:
        ot_assocs = open_targets_api.get_gene_disease_associations(gene)
    except Exception as exc:
        logger.warning("Open Targets failed for %s: %s", gene, exc)
        ot_assocs = []

    try:
        prot_info = uniprot_api.get_protein_info(gene)
    except Exception as exc:
        logger.warning("UniProt failed for %s: %s", gene, exc)
        prot_info = {}

    uniprot_id = prot_info.get("uniprot_id")

    try:
        if uniprot_id:
            pathways = reactome_api.get_pathways_for_uniprot(uniprot_id)
        else:
            pathways = reactome_api.get_pathways_for_gene(gene)
    except Exception as exc:
        logger.warning("Reactome failed for %s: %s", gene, exc)
        pathways = []

    disease_names = [a["disease_name"] for a in ot_assocs if a.get("disease_name")]
    best_ot_score = max((a.get("score", 0.0) for a in ot_assocs), default=0.0)

    return BiologicalEvidence(
        gene=gene,
        known_function=prot_info.get("function", ""),
        pathways=pathways[:15],
        disease_associations=disease_names[:10],
        uniprot_id=uniprot_id,
        biological_support=_grade_support(best_ot_score, disease_names, pathways, prot_info),
        support_score=float(best_ot_score),
        evidence_sources=_build_sources(ot_assocs, prot_info, pathways),
    )


def run(state: CaseState) -> CaseState:
    genes: list[str] = [gr.gene for gr in state.ranked_genes]

    if state.input.candidate_gene and state.input.candidate_gene not in genes:
        genes.append(state.input.candidate_gene)

    if not genes:
        state.mark_done("bio_integrator")
        return state

    bio_evidence: list[BiologicalEvidence] = list(state.biological_evidence)
    existing_genes = {be.gene for be in bio_evidence}
    pending_genes = [g for g in genes if g not in existing_genes]

    if pending_genes:
        with ThreadPoolExecutor(max_workers=_MAX_GENE_WORKERS) as pool:
            for ev in pool.map(_fetch_for_gene, pending_genes):
                bio_evidence.append(ev)

    state.biological_evidence = bio_evidence
    state.mark_done("bio_integrator")
    return state


def _grade_support(
    ot_score: float,
    diseases: list[str],
    pathways: list[str],
    prot_info: dict,
) -> str:
    if ot_score >= 0.7 or (diseases and ot_score >= 0.5):
        return "strong"
    if ot_score >= 0.3 or diseases or pathways or prot_info.get("function"):
        return "moderate"
    return "weak"


def _build_sources(
    ot_assocs: list[dict],
    prot_info: dict,
    pathways: list[str],
) -> list[str]:
    sources: list[str] = []
    if ot_assocs:
        sources.append("Open Targets Platform")
    if prot_info:
        sources.append("UniProt")
    if pathways:
        sources.append("Reactome")
    return sources
