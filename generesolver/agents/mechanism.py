from __future__ import annotations

import json
import logging

from generesolver.integrations.llm_client import LLMResponseError, complete_json
from generesolver.schemas.case import CaseState

logger = logging.getLogger(__name__)
_TOP_GENES = 5


def run(state: CaseState) -> CaseState:
    top_genes = [gr.gene for gr in state.ranked_genes[:_TOP_GENES]]
    if state.input.candidate_gene and state.input.candidate_gene not in top_genes:
        top_genes.insert(0, state.input.candidate_gene)

    bio_index = {be.gene: be for be in state.biological_evidence}

    for gene in top_genes:
        bio = bio_index.get(gene)
        if bio is None or bio.mechanism_summary_doctor:
            continue
        try:
            doctor_txt, expert_txt = _generate(gene, bio)
            bio.mechanism_summary_doctor = doctor_txt
            bio.mechanism_summary_expert = expert_txt
        except Exception as exc:
            logger.warning("Mechanism generation failed for %s: %s", gene, exc)

    state.mark_done("mechanism")
    return state


def _generate(gene: str, bio: object) -> tuple[str, str]:
    function_text = getattr(bio, "known_function", "") or "Unknown"
    pathways = ", ".join(getattr(bio, "pathways", [])[:5]) or "None identified"
    diseases = ", ".join(getattr(bio, "disease_associations", [])[:5]) or "None identified"

    prompt = (
        f"You are a clinical genetics expert.\n\n"
        f"Gene: {gene}\n"
        f"Protein function: {function_text}\n"
        f"Key pathways: {pathways}\n"
        f"Known disease associations: {diseases}\n\n"
        f"Provide two mechanism summaries as JSON (no markdown fences):\n"
        f'{{"doctor": "<2-3 plain-English sentences for a non-geneticist clinician>", '
        f'"expert": "<3-5 detailed mechanistic sentences with pathway context for a clinical geneticist>"}}'
    )

    try:
        data = complete_json(system="", user=prompt, max_tokens=1024)
    except LLMResponseError as exc:
        # Some OpenAI-compatible models return several otherwise valid
        # alternatives as a JSON array despite the request for one object.
        try:
            alternatives = json.loads(exc.raw_payload)
        except (TypeError, ValueError):
            raise exc
        if not isinstance(alternatives, list) or not alternatives:
            raise exc
        data = alternatives[0]
        if not isinstance(data, dict):
            raise exc
        logger.info("Mechanism model returned %d alternatives for %s; using the first.",
                    len(alternatives), gene)
    doctor, expert = data.get("doctor"), data.get("expert")
    if not isinstance(doctor, str) or not doctor.strip() \
            or not isinstance(expert, str) or not expert.strip():
        raise LLMResponseError("Mechanism response must contain nonempty doctor and expert summaries")
    return doctor.strip(), expert.strip()
