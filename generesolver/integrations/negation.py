from __future__ import annotations

import logging

from pydantic import BaseModel

from generesolver.integrations.llm_client import complete_structured
from generesolver.schemas.clinical import NegationStatus

logger = logging.getLogger(__name__)



class _NegationResult(BaseModel):
    status: str


_NEG_LLM_SYSTEM = (
    "You are classifying whether a specific clinical finding is stated as present, "
    "absent (negated), or uncertain in a passage of clinical text.\n"
    "present: the finding is stated as occurring in / observed in the patient.\n"
    "absent: the finding is explicitly negated (e.g. 'no seizures', 'denies X', "
    "'ruled out X', 'negative for X', a header like 'Absent features: X').\n"
    "uncertain: the finding is hedged (e.g. 'possible X', 'suspected X', 'cannot "
    "exclude X', 'query X') — neither clearly present nor clearly absent.\n"
    'Respond with a JSON object: {"status": "present"|"absent"|"uncertain"}.'
)

_STATUS_MAP = {
    "present": NegationStatus.PRESENT,
    "absent": NegationStatus.ABSENT,
    "uncertain": NegationStatus.UNCERTAIN,
}

def detect_llm(
    text: str,
    span_text: str,
    span_start: int,
    span_end: int,
    *,
    deidentification_applied: bool = True,
    backend: str | None = None,
) -> NegationStatus:
    window_start = max(0, span_start - 150)
    window_end = min(len(text), span_end + 100)
    window = text[window_start:window_end]
    
    user = f"Passage: {window!r}\nFinding to classify: {span_text!r}"
    try:
        try:
            result = complete_structured(
                system=_NEG_LLM_SYSTEM, user=user,
                schema_model=_NegationResult, max_tokens=50, backend=backend,
            )
        except Exception as exc:
            logger.warning("negation: LLM classification failed schema, retrying once: %s", exc)
            result = complete_structured(
                system=_NEG_LLM_SYSTEM,
                user=user + '\n\n(CORRECTION: return ONLY {"status": "present"|"absent"|"uncertain"}'
                             ' for the single finding above, nothing else.)',
                schema_model=_NegationResult, max_tokens=50, backend=backend,
            )
        return _STATUS_MAP.get((result.status or "").strip().lower(), NegationStatus.PRESENT)
    except Exception as exc:
        logger.warning("negation: LLM classification failed: %s", exc)
        return NegationStatus.PRESENT
