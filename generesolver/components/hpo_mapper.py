from __future__ import annotations

import logging

from pydantic import BaseModel

from generesolver.integrations.fasthpocr_extractor import (
    extract_hpo_terms as fasthpocr_extract,
    is_available as fasthpocr_available,
)
from generesolver.integrations import hpo_db, negation
from generesolver.integrations.llm_client import complete_structured
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


_MISSING_FINDINGS_SYSTEM = (
    "You are given a clinical passage and a list of phenotype terms ALREADY identified in it "
    "by a dictionary-based tool. That tool often misses structural/imaging findings (organ "
    "enlargement, cysts, structural anomalies, etc) that are phrased in ways its fixed "
    "dictionary doesn't match.\n"
    "Identify any ADDITIONAL distinct clinical abnormality described in the passage that is "
    "NOT already covered (even loosely) by the given list. Do not report anything explicitly "
    "stated as normal/absent/negative/ruled out.\n"
    "For each additional finding, return:\n"
    "- verbatim_quote: the EXACT substring copied from the passage that describes it (not "
    "paraphrased -- must appear character-for-character in the passage)\n"
    "- clinical_phrase: the SAME finding as a MINIMAL, atomic clinical/anatomical term (2-3 "
    "words, canonical medical word order, no compound conjunctions like 'with'/'and'/'plus'). "
    "This gets matched against a medical ontology by substring, so it must be a single simple "
    "concept, not a compound description. If one verbatim_quote actually describes MULTIPLE "
    "distinct abnormalities (e.g. both organ enlargement AND cysts), report them as SEPARATE "
    "finding entries (same verbatim_quote is fine to repeat), each with its own minimal "
    "clinical_phrase -- e.g. for 'numerous bilateral cysts...with marked enlargement of both "
    "kidneys', report two findings: clinical_phrase='renal cysts' AND clinical_phrase='enlarged "
    "kidney', not one combined phrase.\n"
    "- modifiers: any severity/laterality/multiplicity words stated for this finding (e.g. "
    "['multiple', 'bilateral'], ['severe'], ['mild']) -- empty list if none. This distinction "
    "matters medically (e.g. a single incidental renal cyst is common and nonspecific, while "
    "multiple bilateral renal cysts is a distinct, much more specific finding) but doesn't "
    "belong in clinical_phrase since it would break the ontology substring search.\n"
    'Return a JSON object: {"findings": [{"verbatim_quote": str, "clinical_phrase": str, '
    '"modifiers": [str, ...]}, ...]}, or {"findings": []} if there is nothing to add.'
)


class _MissingFinding(BaseModel):
    verbatim_quote: str
    clinical_phrase: str
    modifiers: list[str] = []


class _MissingFindingsResult(BaseModel):
    findings: list[_MissingFinding] = []


_MISSING_FINDINGS_MAX_TOKENS = 1200


def _find_additional_findings(
    text: str, known_labels: list[str], backend: str | None,
) -> list[_MissingFinding]:
    user = (
        f"ALREADY IDENTIFIED:\n{', '.join(known_labels) or '(none)'}\n\n"
        f"PASSAGE:\n{text}"
    )
    try:
        result = complete_structured(
            system=_MISSING_FINDINGS_SYSTEM, user=user,
            schema_model=_MissingFindingsResult, max_tokens=_MISSING_FINDINGS_MAX_TOKENS, backend=backend,
        )
    except Exception as exc:
        logger.warning("hpo_mapper: supplementary finding search failed schema, retrying once: %s", exc)
        try:
            result = complete_structured(
                system=_MISSING_FINDINGS_SYSTEM,
                user=user + '\n\n(CORRECTION: your previous answer used the wrong JSON schema. '
                            'Return ONLY {"findings": [{"verbatim_quote": str, "clinical_phrase": '
                            'str, "modifiers": [str, ...]}, ...]}, using {"findings": []} if '
                            'there is nothing to add.)',
                schema_model=_MissingFindingsResult, max_tokens=_MISSING_FINDINGS_MAX_TOKENS, backend=backend,
            )
        except Exception as exc2:
            logger.warning("hpo_mapper: supplementary finding search failed after retry: %s", exc2)
            return []
    return result.findings


def _search_candidates(ontology, phrase: str):
    return list(ontology.search(phrase))


def _lookup_hpo_term(phrase: str, modifiers: list[str] | None = None) -> tuple[str, str] | None:
    try:
        ontology = hpo_db._get_ontology()
        candidates = [
            (str(term.id), term.name) for term in _search_candidates(ontology, phrase)
            if hpo_db.is_phenotypic_abnormality(str(term.id))
        ]
        if not candidates:
            return None
        if modifiers:
            mods = [m.lower() for m in modifiers]
            def _score(candidate: tuple[str, str]) -> int:
                name_lower = candidate[1].lower()
                return sum(1 for m in mods if m in name_lower)
            best = max(candidates, key=_score)
            if _score(best) > 0:
                return best
        return candidates[0]
    except Exception as exc:
        logger.warning("hpo_mapper: HPO ontology search failed for %r: %s", phrase, exc)
        return None


def run(state: CaseState, backend: str | None = None) -> CaseState:
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

    text = (state.clinical_note_deidentified
            if state.deidentification_applied
            else (state.input.clinical_note or ""))
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
            backend=backend,
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

    if backend is not None:
        known_labels = [t.label for t in hpo_terms if t.label]
        for finding in _find_additional_findings(text, known_labels, backend):
            match = _lookup_hpo_term(finding.clinical_phrase, finding.modifiers)
            if match is None:
                logger.info(
                    "hpo_mapper: no HPO term found for supplementary finding %r",
                    finding.clinical_phrase,
                )
                continue
            hpo_id, label = match
            if hpo_id in seen_ids:
                continue

            start = text.find(finding.verbatim_quote)
            if start == -1:
                logger.debug(
                    "hpo_mapper: supplementary finding quote not found verbatim, dropped: %r",
                    finding.verbatim_quote,
                )
                continue
            end = start + len(finding.verbatim_quote)

            neg_status = negation.detect_llm(
                text, finding.verbatim_quote, start, end,
                deidentification_applied=state.deidentification_applied,
                backend=backend,
            )
            seen_ids.add(hpo_id)
            hpo_terms.append(HPOTerm(
                mention=finding.verbatim_quote,
                hpo_id=hpo_id,
                label=label,
                status=_negation_to_hpo_status(neg_status),
                confidence=0.7,
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
        tool_name="FastHPOCR (Groza et al., Bioinformatics 2024) + LLM-supplemented HPO search",
        tool_version=f"{len(hpo_terms)} HPO term(s) extracted",
    ))
    state.mark_done("hpo_mapper")
    return state
