from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from generesolver.integrations import llm_client

logger = logging.getLogger(__name__)

_SYSTEM = "You are a clinical geneticist producing a ranked differential diagnosis."


class _RankerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ranked: list[str] = Field(default_factory=list)


def rank(case_block: str, reasoning: str, mechanism: str, evidence_block: str,
         candidate_names: list[str]) -> list[str]:
    if not candidate_names:
        return []
    n_total = len(candidate_names)
    candidates = "\n".join(f"- {n}" for n in candidate_names)
    user = (
        f"Patient findings:\n{case_block}\n\n"
        f"Per-candidate evidence retrieved from databases:\n{evidence_block}\n\n"
        f"Reasoning:\n{reasoning or '(none)'}\n\n"
        f"Mechanism:\n{mechanism or '(none)'}\n\n"
        f"Candidate diseases ({n_total} total):\n{candidates}\n\n"
        "Rank ALL of the above candidate diseases from MOST to least likely for THIS patient, "
        "based on how well the retrieved evidence above explains the case (matched findings, "
        "biochemical specificity, literature). Weigh the patient's SPECIFIC findings (those "
        "characterising few diseases) far above generic findings (those characterising many); "
        "do NOT rank a disease highly only because it shares generic or severe features. "
        "If ABSENT / excluded findings are listed, do not drop the contradicted candidate — "
        "still include it, but place it near the bottom (rejected candidates must remain "
        "visible, not silently disappear). "
        f"Your \"ranked\" list MUST contain exactly the same {n_total} disease names given above, "
        "each exactly once, reordered — do not omit any, do not add any, and do not stop early "
        "even if you are unsure about the ones near the bottom. Put the single most likely "
        "diagnosis first.\n"
        'Return JSON: {"ranked": ["<disease name>", ...]}'
    )
    try:
        r = llm_client.complete_structured(
            _SYSTEM, user, schema_model=_RankerResponse, max_tokens=4096)
    except Exception as exc:
        logger.warning("dx.ranker failed: %s", exc)
        return []
    if r is None:
        return []
    return [n.strip() for n in (r.ranked or []) if n and n.strip()]
