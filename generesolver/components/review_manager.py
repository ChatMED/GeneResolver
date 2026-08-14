from __future__ import annotations

from generesolver.schemas.case import CaseState


def run(state: CaseState) -> CaseState:
    seen: set[str] = set()
    deduped: list[str] = []
    for reason in state.human_review_reasons:
        if reason not in seen:
            seen.add(reason)
            deduped.append(reason)
    state.human_review_reasons = deduped

    if state.safety_flags and state.confidence_level == "high":
        state.confidence_level = "moderate"

    if not state.ranked_genes:
        state.confidence_level = "insufficient"

    if state.safety_flags:
        state.requires_human_review = True

    state.mark_done("review_manager")
    return state
