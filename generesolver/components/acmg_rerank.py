from __future__ import annotations

import logging

from generesolver.config import RANK_WEIGHT_VARIANT
from generesolver.schemas.acmg_assessment import ACMGComparisonResult
from generesolver.schemas.case import CaseState
from generesolver.schemas.report import ProvenanceItem
from generesolver.schemas.variant_disease_context import VariantDiseaseContext

logger = logging.getLogger(__name__)

_CLASSIFICATION_SCORES = {
    "pathogenic": 1.0,
    "likely_pathogenic": 0.8,
    "vus": 0.5,
    "likely_benign": 0.2,
    "benign": 0.0,
}


def _preferred_disease_id(gene: str, contexts: list[VariantDiseaseContext]) -> str | None:
    gene_contexts = [c for c in contexts if c.gene == gene]
    if not gene_contexts:
        return None
    best_score = max((c.phenotype_match_score or 0.0) for c in gene_contexts)
    tied = {c.disease_id for c in gene_contexts if (c.phenotype_match_score or 0.0) == best_score}
    return next(iter(tied)) if len(tied) == 1 else None


def _select_assessment(
    gene: str, assessments: list[ACMGComparisonResult], contexts: list[VariantDiseaseContext],
) -> ACMGComparisonResult | None:
    preferred = _preferred_disease_id(gene, contexts)
    if preferred is None:
        return None
    candidates = [
        a for a in assessments
        if a.computed_acmg_assessment.gene == gene
        and a.computed_acmg_assessment.disease_id == preferred
        and a.computed_acmg_assessment.classification in _CLASSIFICATION_SCORES
        and a.computed_acmg_assessment.counted_criteria
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda a: _CLASSIFICATION_SCORES[a.computed_acmg_assessment.classification])


def run(state: CaseState) -> CaseState:
    if not state.acmg_assessments or not state.ranked_genes:
        state.add_warning("ACMG re-rank skipped: no ACMG assessments or ranked genes.")
        state.mark_done("acmg_rerank")
        return state

    updated = 0
    for gr in state.ranked_genes:
        assessment = _select_assessment(gr.gene, state.acmg_assessments, state.variant_disease_contexts)
        if assessment is None:
            continue
        new_score = _CLASSIFICATION_SCORES[assessment.computed_acmg_assessment.classification]
        delta = (new_score - gr.score_variant) * RANK_WEIGHT_VARIANT
        gr.score_variant = new_score
        gr.final_score = round(min(1.0, max(0.0, gr.final_score + delta)), 4)
        gr.acmg_rerank_interpretation_id = assessment.computed_acmg_assessment.interpretation_id
        updated += 1

    state.ranked_genes.sort(key=lambda g: g.final_score, reverse=True)
    for i, gr in enumerate(state.ranked_genes, start=1):
        gr.rank = i
    state.confidence_level = state.ranked_genes[0].confidence if state.ranked_genes else "insufficient"

    if updated:
        logger.info("ACMG re-rank: updated score_variant/final_score for %d gene(s).", updated)
        state.provenance.append(ProvenanceItem(
            component="acmg_rerank",
            tool_name="GeneResolver ACMG-aware re-rank",
            tool_version="3.0",
        ))

    state.mark_done("acmg_rerank")
    return state
