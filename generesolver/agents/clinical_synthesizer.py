from __future__ import annotations

import logging
from typing import Any

from generesolver.integrations import hpo_a, orphanet
from generesolver.integrations.llm_client import (
    LLMResponseError,
    active_backend,
    complete_json,
)
from generesolver.schemas.case import CaseState
from generesolver.schemas.clinical import (
    DiseaseMention,
    DiseaseSource,
    DiseaseStatus,
)
from generesolver.schemas.hpo import HPOTerm
from generesolver.schemas.report import ProvenanceItem

logger = logging.getLogger(__name__)

_MAX_TOKENS = 1536
_MAX_NOTE_CHARS = 6000
_SHORTLIST_MIN_OVERLAP = 3
_SHORTLIST_TOP_N = 12
_LLM_DISEASE_CONFIDENCE = 0.75

_MAX_DISEASE_SPAN_CHARS = 200
_MIN_SPAN_CHARS = 3


def run(state: CaseState) -> CaseState:
    backend = active_backend()

    if backend != "medgemma" and not state.deidentification_applied:
        msg = (
            f"clinical_synthesizer: refused to run — online backend {backend!r} selected but "
            "deidentification was not applied. Raw clinical notes must not be "
            "sent to online APIs."
        )
        logger.warning(msg)
        state.add_safety_flag(
            "clinical_synthesizer skipped: PII filter required for online LLM backend"
        )
        state.add_warning(msg)
        state.mark_done("clinical_synthesizer")
        return state

    text = state.clinical_note_deidentified if state.deidentification_applied else ""
    if not text.strip():
        logger.info("clinical_synthesizer: no clinical note text — skip")
        state.mark_done("clinical_synthesizer")
        return state

    import generesolver.config as _gcfg
    if not state.hpo_profile.terms and not getattr(_gcfg, "ALLOW_SYNTHESIZER_ON_EMPTY_HPO", False):
        logger.warning(
            "clinical_synthesizer: upstream HPO profile empty (FastHPOCR may "
            "have failed). Skipping — synthesizer is an enricher, not a "
            "replacement for deterministic extraction."
        )
        state.add_warning(
            "clinical_synthesizer skipped: empty upstream HPO profile. Check "
            "FastHPOCR install and data/hp.obo presence."
        )
        state.mark_done("clinical_synthesizer")
        return state
    if not state.hpo_profile.terms:
        logger.info("clinical_synthesizer: empty HPO profile but ALLOW_SYNTHESIZER_ON_EMPTY_HPO "
                    "set — running in evaluation-rescue mode (strict validators apply).")

    note_for_llm = text[:_MAX_NOTE_CHARS]
    truncated = len(text) > _MAX_NOTE_CHARS

    existing_orpha = {m.orpha_id for m in state.disease_mentions if m.orpha_id}

    present_ids = state.hpo_profile.present_ids
    candidate_diseases = hpo_a.candidate_diseases_for_hpo_profile(
        present_ids,
        min_overlap=_SHORTLIST_MIN_OVERLAP,
        top_n=_SHORTLIST_TOP_N,
    )
    shortlist_ids = {c.orpha_id for c in candidate_diseases}

    prompt = _build_prompt(
        note_for_llm,
        existing_hpo_terms=state.hpo_profile.terms,
        existing_disease_mentions=state.disease_mentions,
        candidate_diseases=candidate_diseases,
        truncated=truncated,
    )

    try:
        raw = complete_json(system="", user=prompt, max_tokens=_MAX_TOKENS)
    except LLMResponseError as exc:
        logger.warning("clinical_synthesizer: LLM error (%s) — skip", exc)
        state.add_warning(f"clinical_synthesizer: LLM error — {exc}")
        state.mark_done("clinical_synthesizer")
        return state
    except Exception as exc:
        logger.warning("clinical_synthesizer: unexpected error (%s) — skip", exc)
        state.add_warning(f"clinical_synthesizer: unexpected error — {exc}")
        state.mark_done("clinical_synthesizer")
        return state

    # The synthesizer contributes syndrome/disease hypotheses only; it never changes the HPO profile.
    new_diseases = _pass_a_diseases(raw, text, existing_orpha, shortlist_ids)
    if new_diseases:
        state.disease_mentions.extend(new_diseases)

    selected_orpha = {m.orpha_id for m in new_diseases if m.orpha_id}
    shortlist_records: list[dict] = []
    for c in candidate_diseases:
        try:
            genes = sorted({
                ga.gene_symbol for ga in orphanet.get_genes_for_disease(c.orpha_id)
            })
        except Exception:
            genes = []
        shortlist_records.append({
            "orpha_id": c.orpha_id,
            "label": c.label,
            "overlap": c.overlap,
            "selected": c.orpha_id in selected_orpha,
            "genes": genes,
        })
    state.diagnostics["synthesizer"] = {
        "n_present_hpo": len(present_ids),
        "shortlist_min_overlap": _SHORTLIST_MIN_OVERLAP,
        "shortlist": shortlist_records,
        "selected_orpha": sorted(selected_orpha),
    }

    state.provenance.append(ProvenanceItem(
        component="clinical_synthesizer",
        tool_name=(
            f"LLM syndrome hypotheses ({backend}), grounded in text spans and "
            "resolved against Orphanet"
        ),
        tool_version=(
            f"backend={backend}; "
            f"deidentified={state.deidentification_applied}; "
            f"diseases_added={len(new_diseases)}"
        ),
    ))
    state.mark_done("clinical_synthesizer")
    return state

def _build_prompt(
    note: str,
    *,
    existing_hpo_terms: list[HPOTerm],
    existing_disease_mentions: list[DiseaseMention],
    candidate_diseases: list[Any],
    truncated: bool,
) -> str:
    existing_hpo_lines = [
        f"  - {t.label} ({t.status.value})"
        for t in existing_hpo_terms
    ] or ["  (none extracted)"]
    existing_disease_lines = [
        f"  - {m.label} ({m.status.value})"
        for m in existing_disease_mentions
    ] or ["  (none extracted)"]

    if candidate_diseases:
        candidate_lines = [
            f"  - {c.orpha_id}  {c.label}  (matches {c.overlap} of patient's HPO terms)"
            for c in candidate_diseases
        ]
    else:
        candidate_lines = ["  (no diseases overlap ≥3 of patient's HPO terms)"]

    trunc_note = "\n[NOTE TRUNCATED]" if truncated else ""

    return (
        "You are a clinical geneticist assistant reviewing a clinical note "
        "after deterministic dictionary-based extractors have already run. "
        "Your job is to propose syndrome or disease hypotheses that those "
        "extractors MISSED.\n\n"
        "OUTPUT:\n"
        "  inferred_disease_hypotheses — propose the SYNDROME NAME with the "
        "symptom cluster as text_span.\n\n"
        "CANDIDATE_DISEASES — a shortlist of rare diseases whose annotated "
        "HPO terms overlap the patient's profile. PREFER picking from this "
        "list (use the orpha_id field for an exact match). If the list does "
        "NOT contain the right syndrome, you MAY propose a different one by "
        "name (set orpha_id to null and provide a canonical label) — but only "
        "when you have a strong clinical reason. The validator will lookup "
        "your label via Orphanet's name index, so spell the canonical name.\n\n"
        "HARD RULES (failing any drops the item silently):\n"
        "1. Use ONLY information in NOTE. Do NOT invent symptoms, syndromes, "
        "lab values, family history, or demographics.\n"
        "2. Every item MUST cite a text_span — a VERBATIM substring of NOTE.\n"
        "3. text_span 3–200 chars; SHOULD be the multi-symptom phrase. status "
        "MUST be 'uncertain' unless the NOTE literally names the syndrome.\n"
        "4. Do NOT duplicate diseases already in EXISTING_DISEASES.\n"
        "5. Prefer FEWER, HIGHER-QUALITY items. 0 outputs is acceptable.\n\n"
        "EXAMPLE — picking from the shortlist:\n"
        "  CANDIDATE_DISEASES included 'ORPHA:3156  Senior-Loken syndrome  "
        "(matches 4 of patient's HPO terms)'\n"
        "  Patient note mentions polyuria + polydipsia + nyctalopia + growth delay.\n"
        "  → inferred_disease_hypotheses:\n"
        "      orpha_id:  'ORPHA:3156'\n"
        "      label:     'Senior-Loken syndrome'\n"
        "      status:    'uncertain'\n"
        "      text_span: 'excessive thirst, frequent urination, growth delay, night blindness'\n"
        "      reasoning: 'Classic nephronophthisis + retinal dystrophy ciliopathy pattern'\n\n"
        "EXAMPLE — free-recall when shortlist is incomplete:\n"
        "  CANDIDATE_DISEASES did NOT include CAH but cluster is 'ambiguous "
        "genitalia + salt-wasting + affected sibling' in a neonate.\n"
        "  → inferred_disease_hypotheses:\n"
        "      orpha_id:  null\n"
        "      label:     'Congenital adrenal hyperplasia'\n"
        "      status:    'uncertain'\n"
        "      text_span: 'ambiguous genitalia, salt-wasting, affected sibling'\n"
        "      reasoning: 'Virilization + salt-wasting + familial → CAH'\n\n"
        f"NOTE:\n{note}{trunc_note}\n\n"
        "CANDIDATE_DISEASES (orpha_id / label / how many patient HPO terms match):\n"
        + "\n".join(candidate_lines) + "\n\n"
        "EXISTING_HPO (phenotypes already extracted, for context):\n"
        + "\n".join(existing_hpo_lines) + "\n\n"
        "EXISTING_DISEASES (already extracted — do not repeat):\n"
        + "\n".join(existing_disease_lines) + "\n\n"
        "Respond JSON only, no prose, no markdown. Schema:\n"
        '{"inferred_disease_hypotheses":['
        '{"orpha_id":"ORPHA:N or null","label":"","status":"uncertain","text_span":"","reasoning":""}'
        "]}"
    )


def _pass_a_diseases(
    raw: dict | None,
    note: str,
    existing_orpha: set[str],
    shortlist_ids: set[str],
) -> list[DiseaseMention]:

    if not isinstance(raw, dict):
        return []
    items = raw.get("inferred_disease_hypotheses") or []
    if not isinstance(items, list):
        return []

    seen_in_this_pass: set[str] = set()
    out: list[DiseaseMention] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        raw_orpha = str(item.get("orpha_id") or "").strip()
        text_span = str(item.get("text_span") or "").strip()

        if not _text_span_in_note(text_span, note, max_chars=_MAX_DISEASE_SPAN_CHARS):
            logger.debug(
                "clinical_synthesizer/A: disease %r dropped — text_span %r fails length/substring",
                label or raw_orpha, text_span[:120],
            )
            continue

        disease = None
        resolved_via = ""
        if raw_orpha and raw_orpha.startswith("ORPHA:") and raw_orpha in shortlist_ids:
            disease = orphanet.get_by_orpha_id(raw_orpha)
            if disease is not None:
                resolved_via = "shortlist"

        if disease is None and label:
            disease = orphanet.normalize(label)
            if disease is not None:
                resolved_via = "normalize"

        if disease is None:
            logger.debug(
                "clinical_synthesizer/A: disease (orpha=%r, label=%r) dropped — "
                "shortlist miss + normalize miss",
                raw_orpha, label,
            )
            continue

        if disease.orpha_id in existing_orpha or disease.orpha_id in seen_in_this_pass:
            continue
        seen_in_this_pass.add(disease.orpha_id)

        logger.debug(
            "clinical_synthesizer/A: disease %s (%s) accepted via %s",
            disease.orpha_id, disease.label, resolved_via,
        )
        out.append(DiseaseMention(
            text=text_span,
            label=disease.label,
            orpha_id=disease.orpha_id,
            omim_ids=list(disease.omim_ids),
            icd10_ids=list(disease.icd10_ids),
            mondo_id=(disease.mondo_ids[0] if disease.mondo_ids else None),
            status=_safe_status_disease(item.get("status")),
            confidence=_LLM_DISEASE_CONFIDENCE,
            source=DiseaseSource.LLM_SYNTHESIS,
            evidence_span=text_span,
            start=-1,
            end=-1,
        ))
    return out

def _safe_status_disease(raw_status: Any) -> DiseaseStatus:
    val = str(raw_status or "").strip().lower()
    if val == "absent":
        return DiseaseStatus.ABSENT
    if val == "present":
        return DiseaseStatus.PRESENT
    return DiseaseStatus.UNCERTAIN


def _text_span_in_note(span: str, note: str, *, max_chars: int = _MAX_DISEASE_SPAN_CHARS) -> bool:
    if not span:
        return False
    s = span.strip()
    if len(s) < _MIN_SPAN_CHARS or len(s) > max_chars:
        return False
    return s.lower() in note.lower()
