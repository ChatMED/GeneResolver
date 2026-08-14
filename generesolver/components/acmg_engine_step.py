from __future__ import annotations

import logging

from generesolver.components import acmg_engine
from generesolver.schemas.case import CaseState
from generesolver.schemas.report import ProvenanceItem

logger = logging.getLogger(__name__)


def run(state: CaseState) -> CaseState:
    if not state.variant_disease_contexts:
        state.add_warning("ACMG engine skipped: no variant-disease contexts to assess.")
        state.mark_done("acmg_engine")
        return state

    variant_by_id = {v.variant_id: v for v in state.annotated_variants}

    assessments = []
    n_skipped = 0
    for context in state.variant_disease_contexts:
        variant = variant_by_id.get(context.variant_id)
        if variant is None:
            n_skipped += 1
            continue
        assessments.append(acmg_engine.assess_with_clinvar(variant, context))

    if n_skipped:
        state.add_warning(
            f"{n_skipped} variant-disease context(s) skipped in ACMG assessment: "
            "no matching annotated variant found."
        )

    state.acmg_assessments = assessments
    logger.info("Produced %d ACMG assessment(s) for %d variant-disease context(s).",
                len(assessments), len(state.variant_disease_contexts))

    n_review = sum(1 for a in assessments if a.computed_acmg_assessment.classification
                   in ("pathogenic", "likely_pathogenic"))
    if n_review:
        state.human_review_reasons.append(
            f"{n_review} ACMG assessment(s) computed as Pathogenic/Likely Pathogenic — "
            "requires clinical geneticist review before any clinical action."
        )
        state.requires_human_review = True

    state.provenance.append(ProvenanceItem(
        component="acmg_engine",
        tool_name="GeneResolver ACMG/AMP evidence engine",
        tool_version="2.0",
    ))
    state.mark_done("acmg_engine")
    return state
