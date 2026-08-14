from __future__ import annotations

import logging

from generesolver.schemas.case import CaseState
from generesolver.schemas.report import ProvenanceItem
from generesolver.schemas.variant import ACMGClassification, ACMGResult, AnnotatedVariant

logger = logging.getLogger(__name__)


_LOF_CONSEQUENCES = {
    "transcript_ablation",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "stop_gained",
    "frameshift_variant",
    "start_lost",
}

_MISSENSE_CONSEQUENCES = {"missense_variant", "protein_altering_variant"}

_INFRAME_CONSEQUENCES = {"inframe_insertion", "inframe_deletion"}

_SYNONYMOUS_CONSEQUENCES = {"synonymous_variant", "stop_retained_variant"}

_BA1_AF = 0.05
_BS1_AF = 0.01
_PM2_AF = 0.001


def _pvs1(v: AnnotatedVariant) -> bool:
    return v.consequence in _LOF_CONSEQUENCES and v.impact == "HIGH"


def _pm2(v: AnnotatedVariant) -> bool:
    if v.gnomad_af is None:
        return True
    return v.gnomad_af < _PM2_AF


def _pm4(v: AnnotatedVariant) -> bool:
    return v.consequence in _INFRAME_CONSEQUENCES


def _pp3(v: AnnotatedVariant) -> bool:
    return (
        v.consequence in _MISSENSE_CONSEQUENCES
        and v.impact in ("HIGH", "MODERATE")
    )


def _pp5(v: AnnotatedVariant) -> bool:
    sig = (v.clinvar_significance or "").lower()
    return any(s in sig for s in ("pathogenic", "likely pathogenic"))


def _ps1(v: AnnotatedVariant) -> bool:
    sig = (v.clinvar_significance or "").lower()
    return "pathogenic" in sig and v.hgvs_p is not None


def _ps5(v: AnnotatedVariant) -> bool:
    sig = (v.clinvar_significance or "").lower().strip()
    return sig in ("pathogenic", "likely pathogenic",
                   "pathogenic/likely pathogenic", "likely pathogenic/pathogenic")


def _bs3(v: AnnotatedVariant) -> bool:
    sig = (v.clinvar_significance or "").lower().strip()
    return sig in ("likely benign", "benign", "likely benign/benign", "benign/likely benign")


def _bp5(v: AnnotatedVariant) -> bool:
    sig = (v.clinvar_significance or "").lower()
    return "benign" in sig and "pathogenic" not in sig


def _ba1(v: AnnotatedVariant) -> bool:
    if v.gnomad_af is not None and v.gnomad_af > _BA1_AF:
        return True
    return (v.clinvar_significance or "").lower().strip() == "benign"


def _bs1(v: AnnotatedVariant) -> bool:
    return v.gnomad_af is not None and v.gnomad_af > _BS1_AF


def _bp4(v: AnnotatedVariant) -> bool:
    return (
        v.consequence in _MISSENSE_CONSEQUENCES
        and v.impact in ("LOW", "MODIFIER")
    )


def _bp7(v: AnnotatedVariant) -> bool:
    return v.consequence in _SYNONYMOUS_CONSEQUENCES


def _classify(
    pvs1: bool, ps1: bool, ps5: bool,
    pm2: bool, pm4: bool,
    pp3: bool, pp5: bool,
    ba1: bool, bs1: bool, bs3: bool,
    bp4: bool, bp5: bool, bp7: bool,
) -> ACMGClassification:
    ps = sum([ps1, ps5])
    pm = sum([pm2, pm4])
    pp = sum([pp3, pp5])
    bs = sum([bs1, bs3])
    bp = sum([bp4, bp5, bp7])

    if ba1:
        return ACMGClassification.BENIGN

    if bs >= 2 or (bs >= 1 and bp >= 1):
        return ACMGClassification.LIKELY_BENIGN

    if pvs1 and ps >= 1:
        return ACMGClassification.PATHOGENIC
    if pvs1 and pm >= 2:
        return ACMGClassification.PATHOGENIC
    if ps >= 2:
        return ACMGClassification.PATHOGENIC
    if ps >= 1 and pm >= 3:
        return ACMGClassification.PATHOGENIC

    if pvs1 and pm == 1:
        return ACMGClassification.LIKELY_PATHOGENIC
    if pvs1 and pp >= 2:
        return ACMGClassification.LIKELY_PATHOGENIC
    if ps >= 1 and pm >= 1:
        return ACMGClassification.LIKELY_PATHOGENIC
    if ps >= 1 and pp >= 2:
        return ACMGClassification.LIKELY_PATHOGENIC
    if pm >= 3:
        return ACMGClassification.LIKELY_PATHOGENIC
    if pm >= 2 and pp >= 2:
        return ACMGClassification.LIKELY_PATHOGENIC
    if pm >= 1 and pp >= 4:
        return ACMGClassification.LIKELY_PATHOGENIC

    return ACMGClassification.VUS


def classify_variant(v: AnnotatedVariant) -> ACMGResult:
    criteria_met: list[str] = []
    criteria_failed: list[str] = []

    def check(fn, code: str, met_list: list, failed_list: list) -> bool:
        result = fn(v)
        (met_list if result else failed_list).append(code)
        return result

    pvs1_val = check(_pvs1, "PVS1", criteria_met, criteria_failed)
    ps1_val  = check(_ps1,  "PS1",  criteria_met, criteria_failed)
    ps5_val  = check(_ps5,  "PS5",  criteria_met, criteria_failed)
    pm2_val  = check(_pm2,  "PM2",  criteria_met, criteria_failed)
    pm4_val  = check(_pm4,  "PM4",  criteria_met, criteria_failed)
    pp3_val  = check(_pp3,  "PP3",  criteria_met, criteria_failed)
    pp5_val  = check(_pp5,  "PP5",  criteria_met, criteria_failed)
    ba1_val  = check(_ba1,  "BA1",  criteria_met, criteria_failed)
    bs1_val  = check(_bs1,  "BS1",  criteria_met, criteria_failed)
    bs3_val  = check(_bs3,  "BS3",  criteria_met, criteria_failed)
    bp4_val  = check(_bp4,  "BP4",  criteria_met, criteria_failed)
    bp5_val  = check(_bp5,  "BP5",  criteria_met, criteria_failed)
    bp7_val  = check(_bp7,  "BP7",  criteria_met, criteria_failed)

    classification = _classify(
        pvs1_val, ps1_val, ps5_val,
        pm2_val, pm4_val,
        pp3_val, pp5_val,
        ba1_val, bs1_val, bs3_val,
        bp4_val, bp5_val, bp7_val,
    )

    warnings: list[str] = []
    if classification == ACMGClassification.VUS:
        warnings.append("Variant of uncertain significance — must not be presented as causal.")
    if v.gnomad_af is None:
        warnings.append("PM2 assessed as met because gnomAD frequency is absent — verify manually.")
    if classification in (ACMGClassification.PATHOGENIC, ACMGClassification.LIKELY_PATHOGENIC):
        warnings.append("Pathogenic/LP call requires expert review and functional validation.")

    return ACMGResult(
        variant_id=v.variant_id,
        gene=v.gene or "unknown",
        classification_support=classification,
        criteria_met=criteria_met,
        criteria_failed=criteria_failed,
        requires_expert_review=True,
        warnings=warnings,
        source="acmg_classifier",
    )

def run(state: CaseState) -> CaseState:
    if not state.annotated_variants:
        state.add_warning("ACMG classifier skipped: no annotated variants.")
        state.mark_done("acmg_classifier")
        return state

    results: list[ACMGResult] = []
    vus_count = 0

    for variant in state.annotated_variants:
        result = classify_variant(variant)
        variant.acmg_result = result
        results.append(result)

        if result.classification_support == ACMGClassification.VUS:
            vus_count += 1

    state.acmg_results = results

    if vus_count > 0:
        state.add_safety_flag(
            f"{vus_count} variant(s) classified as VUS — expert review required before reporting."
        )

    pathogenic_count = sum(
        1 for r in results
        if r.classification_support in (ACMGClassification.PATHOGENIC, ACMGClassification.LIKELY_PATHOGENIC)
    )
    if pathogenic_count > 0:
        state.human_review_reasons.append(
            f"{pathogenic_count} variant(s) with Pathogenic/LP classification support — "
            "requires clinical geneticist review before any clinical action."
        )
        state.requires_human_review = True

    state.provenance.append(ProvenanceItem(
        component="acmg_classifier",
        tool_name="GeneResolver ACMG rule engine",
        tool_version="1.0",
    ))
    state.mark_done("acmg_classifier")
    return state
