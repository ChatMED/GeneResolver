from __future__ import annotations

from generesolver.schemas.case import CaseState

_HIGH_PENETRANCE_AD_GENES = frozenset({
    "BRCA1", "BRCA2", "TP53", "PTEN", "MLH1", "MSH2", "MSH6", "PMS2",
    "RB1", "VHL", "NF1", "NF2", "APC", "MUTYH", "STK11", "CDH1",
    "PALB2", "ATM", "CHEK2",
})

_PHARMACO_GENES = frozenset({
    "CYP2D6", "CYP2C19", "CYP2C9", "DPYD", "TPMT", "UGT1A1",
    "VKORC1", "G6PD", "SLCO1B1", "NUDT15",
})


def run(state: CaseState) -> CaseState:
    _check_empty_hpo(state)
    _check_pathogenic_variants(state)
    _check_multiple_vus(state)
    _check_high_penetrance_genes(state)
    _check_pharmaco_genes(state)
    _check_low_confidence(state)
    _check_inheritance_conflicts(state)

    state.mark_done("safety_monitor")
    return state


def _flag(state: CaseState, flag: str, reason: str | None = None) -> None:
    state.add_safety_flag(flag)
    if reason:
        if reason not in state.human_review_reasons:
            state.human_review_reasons.append(reason)
        state.requires_human_review = True


def _check_empty_hpo(state: CaseState) -> None:
    if state.has_clinical_note and not state.hpo_profile.present_ids:
        _flag(
            state,
            "HPO profile empty despite clinical note — phenotype prioritization not possible",
        )


def _check_pathogenic_variants(state: CaseState) -> None:
    for comp in state.acmg_assessments:
        a = comp.computed_acmg_assessment
        if a.classification in ("pathogenic", "likely_pathogenic"):
            _flag(
                state,
                f"Pathogenic/LP variant in {a.gene}: {a.variant_id}",
                f"ACMG {a.classification} variant {a.variant_id} "
                f"in {a.gene} requires clinical geneticist review.",
            )


def _check_multiple_vus(state: CaseState) -> None:
    vus = [
        comp for comp in state.acmg_assessments
        if comp.computed_acmg_assessment.classification == "vus"
    ]
    if len(vus) >= 3:
        _flag(
            state,
            f"{len(vus)} VUS variants detected — expert interpretation required",
            f"Multiple VUS ({len(vus)}) detected. Expert review needed to assess joint significance.",
        )


def _check_high_penetrance_genes(state: CaseState) -> None:
    for gr in state.ranked_genes:
        if gr.gene in _HIGH_PENETRANCE_AD_GENES and gr.score_variant > 0.0:
            _flag(
                state,
                f"High-penetrance AD gene {gr.gene} has a variant — genetic counselling strongly recommended",
                f"{gr.gene} is a high-penetrance dominant disease gene. "
                "Genetic counselling should be offered to the patient and family.",
            )


def _check_pharmaco_genes(state: CaseState) -> None:
    for gr in state.ranked_genes:
        if gr.gene in _PHARMACO_GENES:
            state.add_safety_flag(
                f"Pharmacogenomic gene {gr.gene} — medication implications may apply"
            )


def _check_low_confidence(state: CaseState) -> None:
    if not state.ranked_genes:
        return
    top = state.ranked_genes[0]
    if top.confidence == "insufficient":
        state.add_safety_flag(
            "Top-ranked gene has insufficient evidence — results are exploratory only"
        )
    elif top.confidence == "low":
        if "Low-confidence ranking." not in state.human_review_reasons:
            state.human_review_reasons.append(
                "Low-confidence ranking — additional clinical evidence is required before acting on these results."
            )


def _check_inheritance_conflicts(state: CaseState) -> None:
    for ic in state.inheritance_results:
        if ic.compatible is False:
            _flag(
                state,
                f"Inheritance conflict: {ic.gene} ({ic.suspected_model.value}) inconsistent with variant data",
                f"Inheritance pattern for {ic.gene} ({ic.suspected_model.value}) "
                "is incompatible with detected variant(s). Expert review required.",
            )
