from __future__ import annotations

import logging

from generesolver.integrations import negation, orphanet
from pydantic import BaseModel
from generesolver.integrations.llm_client import complete_structured
from generesolver.schemas.case import CaseState
from generesolver.schemas.clinical import (
    DiseaseMention,
    DiseaseSource,
    DiseaseStatus,
    NegationStatus,
)
from generesolver.schemas.report import ProvenanceItem

logger = logging.getLogger(__name__)

_NORMALIZATION_CONFIDENCE = 0.95

_UNNORMALIZED_CONFIDENCE_PENALTY = 0.5


def _negation_to_disease_status(neg: NegationStatus) -> DiseaseStatus:
    return {
        NegationStatus.PRESENT:   DiseaseStatus.PRESENT,
        NegationStatus.ABSENT:    DiseaseStatus.ABSENT,
        NegationStatus.UNCERTAIN: DiseaseStatus.UNCERTAIN,
    }[neg]


def _evidence_sentence(text: str, start: int, end: int) -> str:
    sent_start = text.rfind(".", 0, start)
    sent_start = sent_start + 1 if sent_start >= 0 else 0
    sent_end = text.find(".", end)
    sent_end = sent_end + 1 if sent_end >= 0 else len(text)
    return text[sent_start:sent_end].strip()


_DISEASE_NER_SYSTEM = (
    "Extract disease, syndrome, not phenotypes, mentions from the provided clinical passage.\n"
    "Return a JSON object: {\"spans\": [{\"word\": <verbatim mention>, \"start\": <char index>, "
    "\"end\": <char index, exclusive>, \"score\": <confidence 0-1>}, ...]}.\n"
    "Do not invent mentions; return only substrings that appear verbatim in the passage. "
    "Many passages describe phenotypes/findings without naming any specific disease or syndrome "
    "at all -- that is a normal, expected outcome, not an error: in that case return "
    '{"spans": []}, nothing else.'
)


class _DiseaseNERSpan(BaseModel):
    word: str
    start: int
    end: int
    score: float


class _DiseaseNERResult(BaseModel):
    spans: list[_DiseaseNERSpan] = []


def _run_llm_disease_ner(text: str, backend: str | None = None) -> list[dict]:
    try:
        result = complete_structured(
            system=_DISEASE_NER_SYSTEM, user=text,
            schema_model=_DiseaseNERResult, max_tokens=400, backend=backend,
        )
    except Exception as exc:
        logger.warning("disease_extractor: LLM disease NER failed schema, retrying once: %s", exc)
        try:
            result = complete_structured(
                system=_DISEASE_NER_SYSTEM,
                user=text + '\n\n(CORRECTION: your previous answer used the wrong JSON schema. '
                            'Return ONLY {"spans": [...]}, using {"spans": []} if there are no '
                            'disease/syndrome mentions -- nothing else.)',
                schema_model=_DiseaseNERResult, max_tokens=400, backend=backend,
            )
        except Exception as exc2:
            logger.warning("disease_extractor: LLM disease NER failed after retry: %s", exc2)
            return []

    return [
        {"word": s.word, "start": int(s.start), "end": int(s.end), "score": float(s.score)}
        for s in result.spans
    ]


def _build_mention_from_span(
    text: str,
    span_text: str,
    start: int,
    end: int,
    ner_confidence: float,
    source: DiseaseSource,
    deidentification_applied: bool,
    backend: str | None = None,
) -> DiseaseMention:
    neg_status = (
        negation.detect_llm(text, span_text, start, end,
                             deidentification_applied=deidentification_applied,
                             backend=backend)
        if start >= 0 else NegationStatus.PRESENT
    )
    status = _negation_to_disease_status(neg_status)

    orpha = orphanet.normalize(span_text)
    if orpha is not None:
        confidence = ner_confidence * _NORMALIZATION_CONFIDENCE
        return DiseaseMention(
            text=span_text,
            label=orpha.label,
            orpha_id=orpha.orpha_id,
            omim_ids=list(orpha.omim_ids),
            icd10_ids=list(orpha.icd10_ids),
            mondo_id=orpha.mondo_ids[0] if orpha.mondo_ids else None,
            status=status,
            confidence=confidence,
            source=source,
            evidence_span=_evidence_sentence(text, start, end) if start >= 0 else "",
            start=start,
            end=end,
        )

    return DiseaseMention(
        text=span_text,
        label=span_text,
        orpha_id=None,
        status=status,
        confidence=ner_confidence * _UNNORMALIZED_CONFIDENCE_PENALTY,
        source=source,
        evidence_span=_evidence_sentence(text, start, end) if start >= 0 else "",
        start=start,
        end=end,
    )


def _dedupe(mentions: list[DiseaseMention]) -> list[DiseaseMention]:
    by_orpha: dict[str, DiseaseMention] = {}
    by_text: dict[tuple[str, DiseaseStatus], DiseaseMention] = {}

    for m in mentions:
        if m.orpha_id:
            existing = by_orpha.get(m.orpha_id)
            if existing is None or m.confidence > existing.confidence:
                by_orpha[m.orpha_id] = m
        else:
            key = (m.text.lower(), m.status)
            existing = by_text.get(key)
            if existing is None or m.confidence > existing.confidence:
                by_text[key] = m

    merged = list(by_orpha.values()) + list(by_text.values())
    merged.sort(key=lambda m: (m.start if m.start >= 0 else 1 << 30, -m.confidence))
    return merged


def run(state: CaseState, backend: str | None = None) -> CaseState:
    text = state.clinical_note_deidentified if state.deidentification_applied else ""

    mentions: list[DiseaseMention] = []

    if text:
        try:
            raw = _run_llm_disease_ner(text, backend=backend)
        except Exception as exc:
            logger.warning("disease_extractor: LLM disease NER failed: %s", exc)
            raw = []
        logger.info("disease_extractor: LLM disease NER returned %d span(s)", len(raw))
        for e in raw:
            mentions.append(_build_mention_from_span(
                text=text,
                span_text=e["word"],
                start=e["start"],
                end=e["end"],
                ner_confidence=float(e["score"]),
                source=DiseaseSource.NER,
                deidentification_applied=state.deidentification_applied,
                backend=backend,
            ))
    else:
        state.add_warning("disease_extractor: no clinical note text available.")

    user_label = (state.input.disease_label or "").strip()
    if user_label:
        orpha = orphanet.normalize(user_label)
        if orpha is not None:
            mentions.append(DiseaseMention(
                text=user_label,
                label=orpha.label,
                orpha_id=orpha.orpha_id,
                omim_ids=list(orpha.omim_ids),
                icd10_ids=list(orpha.icd10_ids),
                mondo_id=orpha.mondo_ids[0] if orpha.mondo_ids else None,
                status=DiseaseStatus.PRESENT,
                confidence=_NORMALIZATION_CONFIDENCE,
                source=DiseaseSource.USER_LABEL,
                evidence_span="",
                start=-1,
                end=-1,
            ))
        else:
            mentions.append(DiseaseMention(
                text=user_label,
                label=user_label,
                orpha_id=None,
                status=DiseaseStatus.PRESENT,
                confidence=_UNNORMALIZED_CONFIDENCE_PENALTY,
                source=DiseaseSource.USER_LABEL,
                evidence_span="",
                start=-1,
                end=-1,
            ))

    state.disease_mentions = _dedupe(mentions)

    if not state.disease_mentions:
        state.add_warning(
            "disease_extractor: no diseases or syndromes detected in the note. "
            "Pipeline continues phenotype-only."
        )
    elif not orphanet.is_available():
        state.add_warning(
            "disease_extractor: Orphanet en_product1.xml missing — mentions "
            "extracted without canonical IDs. Run scripts/download_orphanet.py."
        )

    n_orpha = sum(1 for m in state.disease_mentions if m.orpha_id)
    state.provenance.append(ProvenanceItem(
        component="disease_extractor",
        tool_name=f"LLM disease NER + Orphanet en_product1 normalizer",
        tool_version=(
            f"backend={backend or 'default'}; "
            f"orphanet_available={orphanet.is_available()}; "
            f"mentions={len(state.disease_mentions)} (orpha-normalized={n_orpha})"
        ),
    ))
    state.mark_done("disease_extractor")
    return state
