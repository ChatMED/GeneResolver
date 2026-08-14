from __future__ import annotations

import logging

from pydantic import BaseModel

from generesolver.integrations.llm_client import active_backend, complete_structured
from generesolver.schemas.case import CaseState
from generesolver.schemas.clinical import ClinicalEntity, EntityType
from generesolver.schemas.report import ProvenanceItem

logger = logging.getLogger(__name__)

_LLM_MAX_TOKENS = 500

_LLM_SYSTEM = (
    "Extract demographic and family-history facts from clinical text. Return ONLY what is "
    "explicitly stated — never infer or guess.\n"
    "age_years: the patient's age in years, converting months/weeks/days to a fractional year "
    "(e.g. 6 months old = 0.5, 3 weeks old = 0.06, 2 days old = 0.005); 'newborn'/'neonate' with "
    "no other age given = 0.0; null if age is not stated at all.\n"
    "age_months: ADDITIONALLY (fill this IN ADDITION TO age_years above, not instead of it), "
    "the RAW number of months, copied as stated, ONLY when the text states the patient's age in "
    "months (e.g. '6-month-old' = 6.0). Do NOT convert this to years yourself -- age_years above "
    "already does that. null if age is not stated in months.\n"
    "sex: 'male' or 'female' if explicitly indicated by the text (words like male/boy/he/his/son "
    "vs female/girl/she/her/daughter); null if ambiguous or not stated.\n"
    "family_history_statements: a list of COMPLETE sentences copied VERBATIM from the input text "
    "that describe a POSITIVE family history fact (parents, siblings, relatives, pedigree, "
    "consanguinity, inheritance pattern actually present) — do not paraphrase or summarize, and "
    "do NOT trim a sentence down to only the family-history clause: if the family-history fact "
    "shares a sentence with other information (e.g. the patient's own name or age), copy the "
    "WHOLE sentence exactly as written, start to end, not just the relevant fragment. Do NOT "
    "include a sentence that states family history is ABSENT/unremarkable/negative/not "
    "contributory (e.g. 'no family history of note') — for those, leave this list empty unless "
    "another sentence has a real positive fact. A sentence mentioning consanguinity belongs here "
    "VERBATIM (whole sentence) in addition to (not instead of) the consanguinity field below — "
    "populate both, they are not mutually exclusive. Empty list if none.\n"
    "consanguinity: true if the parents are explicitly stated to be consanguineous/related "
    "(e.g. first cousins), false if explicitly stated as non-consanguineous, null if not "
    "mentioned.\n"
    "de_novo_stated: true if the variant is described as de novo / new / not inherited, WITHOUT "
    "the text also stating that parental (or trio) genetic testing actually confirmed this; "
    "null if not mentioned.\n"
    "de_novo_confirmed: true ONLY if the text explicitly states parental or trio genetic testing "
    "confirmed the variant is absent in both parents (a de novo claim with no testing mentioned "
    "is de_novo_stated, NOT this); null if not mentioned or not confirmed by testing.\n"
    "segregation_observed: true if the text states the variant or phenotype was found to track "
    "with (co-occur with) the disease across multiple family members who were tested/examined; "
    "false if the text states it does NOT track with the disease in tested relatives; null if "
    "not mentioned or no relatives were tested.\n"
    "n_affected_segregating: the number of affected relatives explicitly stated to carry the "
    "same variant or show the same phenotype tracking with it, if a specific count is given; "
    "null if not stated as a number.\n"
    "n_unaffected_nonsegregating: the number of unaffected relatives explicitly stated to NOT "
    "carry the variant (or not show the phenotype), if a specific count is given; null if not "
    "stated as a number."
)


class _DemographicsExtraction(BaseModel):
    age_years: float | None = None
    age_months: float | None = None
    sex: str | None = None
    family_history_statements: list[str] = []
    consanguinity: bool | None = None
    de_novo_stated: bool | None = None
    de_novo_confirmed: bool | None = None
    segregation_observed: bool | None = None
    n_affected_segregating: int | None = None
    n_unaffected_nonsegregating: int | None = None



def _llm_extract(note: str, backend: str | None = None) -> _DemographicsExtraction | None:
    try:
        return complete_structured(
            system=_LLM_SYSTEM, user=note,
            schema_model=_DemographicsExtraction, max_tokens=_LLM_MAX_TOKENS,
            backend=backend,
        )
    except Exception as exc:
        logger.warning("demographics: LLM extraction failed schema, retrying once: %s", exc)
    try:
        return complete_structured(
            system=_LLM_SYSTEM,
            user=note + '\n\n(CORRECTION: your previous answer used the wrong JSON schema. '
                        'Return ONLY {"age_years": <float|null>, "age_months": <float|null>, '
                        '"sex": "male"|"female"|null, '
                        '"family_history_statements": [...], "consanguinity": <bool|null>, '
                        '"de_novo_stated": <bool|null>, "de_novo_confirmed": <bool|null>, '
                        '"segregation_observed": <bool|null>, "n_affected_segregating": <int|null>, '
                        '"n_unaffected_nonsegregating": <int|null>}, nothing else.)',
            schema_model=_DemographicsExtraction, max_tokens=_LLM_MAX_TOKENS,
            backend=backend,
        )
    except Exception as exc:
        logger.warning("demographics: LLM extraction failed after retry: %s", exc)
        return None


def run(state: CaseState, backend: str | None = None) -> CaseState:
    note = state.input.clinical_note
    if not note:
        state.mark_done("demographics")
        return state

    inp = state.input
    parsed: list[str] = []

    if active_backend() != "medgemma":
        state.add_warning(
            "Demographics extraction skipped: requires the local MedGemma backend "
            "(runs on the original note before de-identification)."
        )
        llm_result = None
    else:
        llm_result = _llm_extract(note, backend=backend)

    if llm_result is not None:
        age = llm_result.age_years
        age_months = llm_result.age_months
        sex = llm_result.sex if llm_result.sex in ("male", "female") else None
        consang = llm_result.consanguinity
        fh_statements = llm_result.family_history_statements
        de_novo_stated = llm_result.de_novo_stated
        de_novo_confirmed = llm_result.de_novo_confirmed
        segregation_observed = llm_result.segregation_observed
        n_affected_segregating = llm_result.n_affected_segregating
        n_unaffected_nonsegregating = llm_result.n_unaffected_nonsegregating
    else:
        age = None
        age_months = None
        sex = None
        consang = None
        fh_statements = []
        de_novo_stated = None
        de_novo_confirmed = None
        segregation_observed = None
        n_affected_segregating = None
        n_unaffected_nonsegregating = None

    state.age_months = age_months
    if age_months is not None:
        age = round(age_months / 12.0, 4)

    state.age_years = inp.age_years if inp.age_years is not None else age
    if inp.age_years is None and age is not None:
        source = "months, converted" if age_months is not None else "llm-extracted"
        parsed.append(f"age={age}y ({source})")

    state.sex = inp.sex or sex
    if not inp.sex and sex:
        parsed.append(f"sex={sex}")

    state.family_history = inp.family_history or ("; ".join(fh_statements) if fh_statements else None)
    if not inp.family_history and fh_statements:
        parsed.append(f"family_history={len(fh_statements)} statement(s)")

    state.consanguinity = inp.consanguinity if inp.consanguinity is not None else consang
    if inp.consanguinity is None and consang is not None:
        parsed.append(f"consanguinity={consang}")

    state.de_novo_stated = de_novo_stated
    state.de_novo_confirmed = de_novo_confirmed
    state.segregation_observed = segregation_observed
    state.n_affected_segregating = n_affected_segregating
    state.n_unaffected_nonsegregating = n_unaffected_nonsegregating
    if de_novo_confirmed is not None:
        parsed.append(f"de_novo_confirmed={de_novo_confirmed}")
    elif de_novo_stated is not None:
        parsed.append(f"de_novo_stated={de_novo_stated}")
    if segregation_observed is not None:
        parsed.append(f"segregation_observed={segregation_observed}")

    for statement in fh_statements:
        start = note.lower().find(statement.lower())
        if start != -1:
            end = start + len(statement)
            state.clinical_entities.append(ClinicalEntity(
                text=statement,
                entity_type=EntityType.FAMILY_HISTORY,
                confidence=0.85,
                start=start,
                end=end,
                source_model="demographics (llm)",
            ))

    if state.family_history:
        state.has_family_history = True

    if parsed:
        logger.info("Demographics parsed (llm): %s", ", ".join(parsed))

    state.trace.add_step(
        step="demographics",
        claim=("Parsed " + ", ".join(parsed)) if parsed else "No demographics found in note",
        grounded_in=["clinical_note (raw, pre-deidentification)"],
    )
    state.provenance.append(ProvenanceItem(
        component="demographics",
        tool_name="llm demographic extractor",
        tool_version="medgemma (local backend only; fails gracefully on error)",
    ))
    state.mark_done("demographics")
    return state
