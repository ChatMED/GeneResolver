from __future__ import annotations

import logging

from generesolver.classification import engine
from generesolver.schemas.biochem import Direction
from generesolver.schemas.case import CaseState
from generesolver.schemas.classification import (
    CaseClass,
    CaseClassification,
    ClassAdvocate,
    ClassificationMode,
)
from generesolver.schemas.report import ProvenanceItem

logger = logging.getLogger(__name__)


def _safe_note(state: CaseState) -> str | None:
    if state.deidentification_applied and state.clinical_note_deidentified:
        return state.clinical_note_deidentified
    return None


def _findings(state: CaseState) -> tuple[str, list[str], list[str]]:
    present_terms = [t for t in state.hpo_profile.terms
                     if getattr(t, "label", None) and t.status.value == "present"]
    hpo_labels = [t.label for t in present_terms]
    hpo_ids = [t.hpo_id for t in present_terms if getattr(t, "hpo_id", None)]

    biochem = [m.summary() for m in state.biochem_panel.measurements
               if m.direction != Direction.UNKNOWN]

    demo: list[str] = []
    if state.age_years is not None:
        demo.append(f"Age at presentation: {state.age_years} years")
    if state.sex:
        demo.append(f"Sex: {state.sex}")
    if state.consanguinity is not None:
        demo.append(f"Consanguinity: {state.consanguinity}")
    if state.family_history:
        demo.append(f"Family history: {state.family_history}")

    variants: list[str] = []
    for v in state.annotated_variants[:15]:
        if v.impact in ("HIGH", "MODERATE"):
            variants.append(f"{v.gene} {v.variant_id}: {v.consequence} ({v.impact})")

    seen: list[str] = []
    narrative = _safe_note(state)
    if narrative:
        block = "CLINICAL NARRATIVE:\n" + narrative
        seen.append("clinical_note")
    else:
        block = engine.build_findings(hpo_labels)
        if hpo_labels:
            seen.append("hpo")
        if biochem:
            block += "\n\nBIOCHEMICAL FINDINGS:\n" + "\n".join(f"- {b}" for b in biochem)
            seen.append("biochem")
        if demo:
            block += "\n\nDEMOGRAPHICS / FAMILY HISTORY:\n" + "\n".join(f"- {d}" for d in demo)
            seen.append("demographics")
        if variants:
            block += "\n\nGENOMIC FINDINGS:\n" + "\n".join(f"- {v}" for v in variants)
            seen.append("variants")

    ev = state.diagnostics.get("class_evidence") or {}
    extra: list[str] = []
    if ev.get("candidate_diseases"):
        extra.append("RETRIEVED CANDIDATE DIAGNOSES (matching this phenotype):\n"
                     + "\n".join(f"- {d}" for d in ev["candidate_diseases"][:8]))
    if ev.get("biochemical"):
        extra.append("BIOCHEMICAL MARKER MATCHES:\n"
                     + "\n".join(f"- {b}" for b in ev["biochemical"][:6]))
    if ev.get("pathways"):
        extra.append("RELEVANT PATHWAYS (for candidate genes):\n"
                     + "\n".join(f"- {p}" for p in ev["pathways"][:8]))
    if ev.get("literature"):
        extra.append("RETRIEVED LITERATURE:\n"
                     + "\n".join(f"- {l}" for l in ev["literature"][:8]))
    if extra:
        block += "\n\n" + "\n\n".join(extra)
        seen.append("retrieval")

    return block, seen, hpo_ids


def _to_schema(result: dict, seen: list[str]) -> CaseClassification:
    advocates = [
        ClassAdvocate(
            case_class=CaseClass(kind),
            score=v["score"],
            evidence=v["evidence"],
            deductions=v["deductions"],
            reasoning=v["reasoning"],
        )
        for kind, v in result["scores"].items()
    ]
    advocates.sort(key=lambda a: -a.score)

    decision = result["decision"]
    return CaseClassification(
        decision=CaseClass(decision) if decision else None,
        mode=ClassificationMode(result["mode"]),
        advocates=advocates,
        margin=result.get("margin", 0.0),
        tiebreak_reasoning=result.get("tiebreak_why", ""),
        conclusion=result.get("conclusion", ""),
        inputs_seen=seen,
        fallback_used=all(a.score == 0.0 for a in advocates),
    )


def run(state: CaseState) -> CaseState:
    findings, seen, hpo_ids = _findings(state)

    if not ({"clinical_note", "hpo", "variants"} & set(seen)):
        logger.warning("case_classifier: no phenotype or genomic evidence — skipping.")
        state.add_warning("Case classification skipped: no phenotype or genomic evidence.")
        state.mark_done("case_classifier")
        return state

    result = engine.classify(findings, hpo_ids=hpo_ids)
    state.case_class = _to_schema(result, seen)

    already_retried = state.diagnostics.get("_class_retried", False)
    if state.case_class.decision is None and not already_retried and "retrieval" not in seen:
        state.diagnostics["_class_retried"] = True
        logger.info("Case uncertain — deferring to class_retrieval then re-classifying.")
        state.trace.add_step(
            step="case_classifier",
            claim="Uncertain on first pass; retrieving evidence to re-classify.",
            grounded_in=seen, confidence="uncertain",
        )
        state.mark_done("case_classifier")
        return state

    decision = state.case_class.decision.value if state.case_class.decision else "uncertain"
    logger.info(
        "Case classified as %s (mode=%s, margin=%.2f) on %s",
        decision, state.case_class.mode.value, state.case_class.margin, "+".join(seen),
    )

    state.trace.add_step(
        step="case_classifier",
        claim=(
            f"Etiology: {decision} ({state.case_class.mode.value}, "
            f"margin {state.case_class.margin:.2f}). "
            + (state.case_class.conclusion or "")
        ),
        grounded_in=seen,
        confidence=state.case_class.mode.value,
    )
    state.provenance.append(ProvenanceItem(
        component="case_classifier",
        tool_name="narrative-grounded etiology classifier",
        tool_version=f"classes={','.join(engine.TYPES)}",
    ))
    state.mark_done("case_classifier")
    return state
