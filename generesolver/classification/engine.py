
from __future__ import annotations

import logging
from typing import Any

from generesolver.integrations import llm_client

logger = logging.getLogger(__name__)

TYPES = ["single_gene", "chromosomal", "multifactorial", "phenocopy"]

MAX_TOKENS = 900
TEMPERATURE = 0.0

_LABEL_MAP = {
    "single_gene": "single_gene",
    "chromosomal": "chromosomal",
    "multifactorial": "multifactorial",
    "non_genetic": "phenocopy",
}

CLASSIFY_PROMPT = """You are a clinical-genetics etiology classifier.

Classify the PRIMARY cause and mechanism, of the main disease for the patient described in the clinical
vignette into exactly ONE of these four labels:

1. "single_gene"
   One gene is sufficient to cause the disorder. Possible molecular mechanisms are:
   -Loss of function: too little or no functional protein.
   -Gain of function: the protein gains abnormal activity.
   -Dominant negative: the abnormal protein interferes with the normal protein.
   -Haploinsufficiency: one functional copy does not produce enough protein.
   -Toxic protein effect: the altered protein accumulates or damages cells.
   When seeing the case narrative, phenotypes, test, lab results, assume the molecular mechanism and detect
   whether the combination of the manifestations are connect in some of these mechanisms. If Yes than is single gene
   label if not do not assume single_gene. Same for mentioned diseases.

2. "chromosomal"
   The cause is a large-scale genomic or chromosome abnormality,
   usually affecting multiple genes or chromosome regulation.
   Common mechanisms include:
    -Aneuploidy: an extra or missing chromosome.
    -Large copy-number variant: deletion or duplication of a chromosome segment.
    -Translocation: chromosome segments exchange positions.
    -Inversion: a chromosome segment is reversed.
    -Imprinting defect: the active parental copy is missing or incorrectly regulated.
    -Uniparental disomy: both chromosome copies come from the same parent.
    When assessing the case, search for confirmation of these mechanisms. If the underlying cause and manifestations are
    explainable by these mechanisms the case is chromosomal

3. "multifactorial"
   No single variant, and no chromosome-level abnormality, is sufficient to
   explain the disease. Possible mechanisms are:
   -Polygenic risk: many variants of small individual effect combine to cross
    a liability threshold; no one variant is sufficient on its own.
   -Gene-environment interaction: a genetic predisposition only produces
    disease together with a specific environmental, developmental, or
    lifestyle factor -- neither alone is sufficient.
   -Postzygotic/somatic process without one driver: a somatic-origin
    condition where no single confirmed clone or variant explains it.
   -Multifactorial developmental process: the phenotype arises from the
    combined effect of several contributing physiological or developmental
    processes, with no one sufficient genetic or chromosomal lesion identified.
   When assessing the case, search for confirmation of these mechanisms the
   same way you would for single_gene or chromosomal -- do not use this label
   as a fallback merely because no mechanism was confirmed; only use it when
   the evidence itself points to combined/distributed causation.

4. "non_genetic"
   The primary cause is acquired, with no pathogenic genetic or chromosomal
   alteration as the primary driver. Possible mechanisms are:
   -Infectious: an active or past infection is the direct cause.
   -Toxic/iatrogenic: a drug, toxin, or medical treatment/procedure is the
    direct cause.
   -Acquired nutritional/metabolic: a dietary deficiency or excess (not an
    inherited metabolic gene defect) is the direct cause.
   -Traumatic/mechanical: physical injury or mechanical disruption is the
    direct cause.
   -Vascular/ischemic: acquired vascular compromise (clot, hemorrhage,
    occlusion) is the direct cause.
   -Acquired autoimmune/inflammatory: an acquired immune-mediated process is
    the direct cause, not an inherited immune-gene defect.
   -Phenocopy: an acquired condition mimics a genetic disorder's phenotype
    without a primary genetic cause.
   A genetic susceptibility combined with one of these factors belongs under
   multifactorial's gene-environment-interaction mechanism, not here.

Return only valid JSON on one line, without Markdown or additional text.
Work out "mechanism_explained" FIRST, then "reasoning", THEN commit to
"label" last -- the label must follow from the mechanism you already stated,
not the other way around:
{"mechanism_explained":"one sentence naming the single specific mechanism from above that best matches the case",
"reasoning":"one sentence on why that mechanism implies the label below, do not overexplain",
"label":"single_gene|chromosomal|multifactorial|non_genetic"}
"""

_SCHEMA_CORRECTION = (
    "\n\n(CORRECTION: your previous answer used the wrong JSON schema. Return ONLY: "
    '{"mechanism_explained": str, "reasoning": str, '
    '"label": "single_gene|chromosomal|multifactorial|non_genetic"})'
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def build_findings(
    hpo_labels: list[str],
    biochem: list[str] | None = None,
    demographics: list[str] | None = None,
    variants: list[str] | None = None,
) -> str:
    block = "PATIENT PHENOTYPE (HPO findings):\n" + "\n".join(f"- {l}" for l in hpo_labels)
    for title, lines in (
        ("BIOCHEMICAL FINDINGS", biochem),
        ("DEMOGRAPHICS / FAMILY HISTORY", demographics),
        ("GENOMIC FINDINGS", variants),
    ):
        if lines:
            block += f"\n\n{title}:\n" + "\n".join(f"- {l}" for l in lines)
    return block


def _call_classifier(findings: str) -> dict:
    try:
        response = llm_client.complete_json(
            CLASSIFY_PROMPT, findings, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
        )
    except Exception as exc:
        logger.warning("Etiology classification failed schema, retrying once: %s", exc)
        try:
            response = llm_client.complete_json(
                CLASSIFY_PROMPT, findings + _SCHEMA_CORRECTION,
                max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            )
        except Exception as exc2:
            raise RuntimeError(f"Etiology classification model failed after retry: {exc2}") from exc2
    if not isinstance(response, dict) or not isinstance(response.get("label"), str):
        raise RuntimeError("Etiology classification returned invalid JSON (missing label)")
    return response


def classify(findings: str, with_llm: bool = True, max_tokens: int = 700, **_ignored) -> dict:
    response = _call_classifier(findings)

    raw_label = _clean_text(response.get("label"))
    normalized = raw_label.lower().replace("-", "_").replace(" ", "_").strip()
    decision = _LABEL_MAP.get(normalized)
    if decision is None and raw_label:
        logger.warning("Unrecognized label %r; treating as uncertain.", raw_label)

    mechanism = _clean_text(response.get("mechanism_explained"))
    reasoning = _clean_text(response.get("reasoning"))
    conclusion = f"{mechanism}. {reasoning}".strip(". ") if mechanism else reasoning

    scores: dict[str, dict] = {
        t: {
            "score": 1.0 if t == decision else 0.0,
            "evidence": [mechanism] if (t == decision and mechanism) else [],
            "deductions": [],
            "reasoning": reasoning if t == decision else "",
        }
        for t in TYPES
    }

    mode = "dominant" if decision else "uncertain"

    return {
        "scores": scores,
        "decision": decision,
        "mode": mode,
        "runner_up": None,
        "margin": 1.0 if decision else 0.0,
        "conclusion": conclusion,
        "tiebreak_why": "",
    }
