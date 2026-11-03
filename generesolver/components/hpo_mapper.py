from __future__ import annotations

import logging

from generesolver.integrations.fasthpocr_extractor import (
    extract_hpo_terms as fasthpocr_extract,
    is_available as fasthpocr_available,
)
from generesolver.integrations import hpo_db, negation
from generesolver.schemas.case import CaseState
from generesolver.schemas.clinical import NegationStatus
from generesolver.schemas.hpo import HPOProfile, HPOStatus, HPOTerm
from generesolver.schemas.report import ProvenanceItem

logger = logging.getLogger(__name__)


def _negation_to_hpo_status(neg: NegationStatus) -> HPOStatus:
    return {
        NegationStatus.PRESENT:   HPOStatus.PRESENT,
        NegationStatus.ABSENT:    HPOStatus.ABSENT,
        NegationStatus.UNCERTAIN: HPOStatus.UNCERTAIN,
    }[neg]


def _evidence_sentence(text: str, start: int, end: int) -> str:
    sent_start = text.rfind(".", 0, start)
    sent_start = sent_start + 1 if sent_start >= 0 else 0
    sent_end = text.find(".", end)
    sent_end = sent_end + 1 if sent_end >= 0 else len(text)
    return text[sent_start:sent_end].strip()


def run(state: CaseState) -> CaseState:
    if state.input.hpo_terms:
        terms = []
        for hid in state.input.hpo_terms:
            try:
                label = hpo_db._get_ontology().get_hpo_object(hid).name
            except Exception:
                label = ""
            terms.append(HPOTerm(
                hpo_id=hid, label=label, mention=label or hid,
                confidence=1.0, evidence_span="", status=HPOStatus.PRESENT,
            ))
        state.hpo_profile = HPOProfile(terms=terms)
        state.mark_done("hpo_mapper")
        return state

    text = state.clinical_note_deidentified if state.deidentification_applied else ""
    if not text:
        state.add_warning("HPO mapper skipped: no clinical note text.")
        state.mark_done("hpo_mapper")
        return state

    if not fasthpocr_available():
        state.add_warning(
            "FastHPOCR unavailable — install fasthpocr + place hp.obo at "
            "data/hp.obo. See generesolver/integrations/fasthpocr_extractor.py."
        )
        state.add_safety_flag("HPO mapper inactive — phenotype profile empty.")
        state.hpo_profile = HPOProfile(terms=[], warnings=list(state.hpo_profile.warnings))
        state.mark_done("hpo_mapper")
        return state

    raw_annotations = fasthpocr_extract(text)

    seen_ids: set[str] = set()
    hpo_terms: list[HPOTerm] = []

    for ann in raw_annotations:
        hpo_id = ann.get("hpo_id", "")
        if not hpo_id or hpo_id in seen_ids:
            continue
        start = ann.get("start_offset", -1)
        end   = ann.get("end_offset",   -1)
        if start < 0 or end < 0:
            continue

        span_text = ann.get("text_span", "") or ann.get("label", "")
        neg_status = negation.detect_llm(
            text, span_text, start, end,
            deidentification_applied=state.deidentification_applied,
        )
        seen_ids.add(hpo_id)

        hpo_terms.append(HPOTerm(
            mention=span_text,
            hpo_id=hpo_id,
            label=ann.get("label", ""),
            status=_negation_to_hpo_status(neg_status),
            confidence=0.95,
            evidence_span=_evidence_sentence(text, start, end),
        ))

    state.hpo_profile = HPOProfile(
        terms=hpo_terms,
        warnings=list(state.hpo_profile.warnings),
    )

    if not hpo_terms:
        state.add_warning("HPO mapper produced no terms — check clinical note quality.")
        state.add_safety_flag("HPO profile empty — phenotype-based ranking not possible")

    state.provenance.append(ProvenanceItem(
        component="hpo_mapper",
        tool_name="FastHPOCR (Groza et al., Bioinformatics 2024)",
        tool_version=f"{len(hpo_terms)} HPO term(s) extracted",
    ))
    state.mark_done("hpo_mapper")
    return state
