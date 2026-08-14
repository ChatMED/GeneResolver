from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from generesolver.integrations import llm_client

logger = logging.getLogger(__name__)

_SYSTEM = "You are a clinical geneticist reasoning."


class _ReasonerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reasoning: str = ""
    mechanism: str = ""
    proposed_diseases: list[str] = Field(default_factory=list)


def reason(case_block: str, findings_block: str) -> tuple[str, str, list[str]]:
    user = (
        f"Patient findings:\n{case_block}\n\n"
        f"Findings retrieved from databases:\n{findings_block}\n\n"
        "Each finding is annotated with how many diseases it characterises. Weigh SPECIFIC "
        "findings (few diseases) far above generic ones (many diseases) — a finding that "
        "characterises thousands of diseases barely narrows the diagnosis.\n"
        "If ABSENT / excluded findings are listed, ELIMINATE any candidate disease that would "
        "cause them.\n"
        "Reason about which diseases best explain this patient. Then, in a MECHANISM section, "
        "explain the biological mechanism linking the findings to the most likely disease(s) "
        "— the enzyme or pathway defect and why it produces these findings. You may also name "
        "additional likely diseases that are not listed above.\n"
        'Return JSON: {"reasoning": "<your reasoning>", "mechanism": "<mechanism explanation>", '
        '"proposed_diseases": ["<disease name>", ...]}'
    )
    try:
        r = llm_client.complete_structured(
            _SYSTEM, user, schema_model=_ReasonerResponse, max_tokens=4096)
    except Exception as exc:
        logger.warning("dx.reasoner failed: %s", exc)
        return "", "", []
    if r is None:
        return "", "", []
    proposed = [n.strip() for n in (r.proposed_diseases or []) if n and n.strip()]
    return (r.reasoning or "").strip(), (r.mechanism or "").strip(), proposed
