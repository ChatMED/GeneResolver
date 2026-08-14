from __future__ import annotations

import logging

from pydantic import BaseModel

from generesolver.integrations.llm_client import complete_structured
from generesolver.schemas.case import CaseState
from generesolver.schemas.gene import InheritanceCompatibility, InheritanceModel
from generesolver.schemas.report import ProvenanceItem
from generesolver.schemas.variant import AnnotatedVariant

logger = logging.getLogger(__name__)



_INHERITANCE_LLM_SYSTEM = (
    "You are given a verbatim family-history text extracted from a clinical case. "
    "Infer the single Mendelian inheritance pattern this family history is most "
    "consistent with, or 'unknown' if it doesn't clearly indicate one.\n"
    "Read the text literally and do not over-interpret it:\n"
    "- 'not consanguineous' / 'non-consanguineous' means the parents are NOT "
    "related -- never treat this as evidence FOR consanguinity or recessive "
    "inheritance.\n"
    "- A statement that relatives were TESTED does not by itself mean they were "
    "AFFECTED -- only count relatives explicitly described as affected/diagnosed.\n"
    "- A negative or sporadic family history is weak evidence, consistent with "
    "autosomal-dominant-by-de-novo-mutation, autosomal recessive, X-linked, or a "
    "non-genetic cause -- it does not, by itself, indicate dominant inheritance "
    "specifically. Prefer 'de_novo' or 'unknown' over guessing a specific "
    "Mendelian pattern from absence of family history alone.\n"
    "- An apparent multi-generation/parent-to-child pattern in the pedigree can "
    "still be autosomal recessive if the visibly-affected relatives could plausibly "
    "be heterozygous carriers with a mild/subclinical phenotype (e.g. thin "
    "basement membrane changes) rather than the same disease as the proband -- "
    "do not assume dominant inheritance from pedigree shape alone if a recessive "
    "explanation is equally plausible.\n"
    "Return one of: autosomal_dominant, autosomal_recessive, x_linked, "
    "x_linked_recessive, x_linked_dominant, mitochondrial, de_novo, unknown.\n"
    'Respond with a JSON object: {"inheritance_model": "<one of the above>"}'
)


class _InheritanceExtraction(BaseModel):
    inheritance_model: str


_RETRY_SUFFIX = (
    "\n\n(CORRECTION: your previous answer used the wrong JSON schema. Return ONLY "
    '{"inheritance_model": "<autosomal_dominant|autosomal_recessive|x_linked|'
    'x_linked_recessive|x_linked_dominant|mitochondrial|de_novo|unknown>"}, nothing else.)'
)


def _llm_infer_model(family_history: str, backend: str | None) -> InheritanceModel | None:
    user = family_history
    for attempt in range(2):
        try:
            result = complete_structured(
                system=_INHERITANCE_LLM_SYSTEM, user=user,
                schema_model=_InheritanceExtraction, max_tokens=50, backend=backend,
            )
            raw = (result.inheritance_model or "").strip().lower()
            return InheritanceModel(raw)
        except Exception as exc:
            if attempt == 0:
                logger.warning("inheritance: LLM inference failed schema, retrying once: %s", exc)
                user = family_history + _RETRY_SUFFIX
                continue
            logger.warning("inheritance: LLM inference failed after retry: %s", exc)
            return None


def infer_inheritance_model(family_history: str, backend: str | None = None) -> InheritanceModel:
    if not family_history:
        return InheritanceModel.UNKNOWN
    llm_result = _llm_infer_model(family_history, backend)
    return llm_result if llm_result is not None else InheritanceModel.UNKNOWN


def _check_compatibility(
    gene: str,
    suspected_model: InheritanceModel,
    variants_for_gene: list[AnnotatedVariant],
    sex: str | None,
) -> InheritanceCompatibility:
    limitations: list[str] = []
    compatible: bool | None = None


    if not variants_for_gene:
        limitations.append("No variants available for zygosity check.")
        return InheritanceCompatibility(
            gene=gene,
            suspected_model=suspected_model,
            compatible=None,
            limitations=limitations,
        )

    pathogenic_variants = [
        v for v in variants_for_gene if v.impact in ("HIGH", "MODERATE")
    ]

    if not pathogenic_variants:
        pathogenic_variants = variants_for_gene

    if suspected_model == InheritanceModel.AUTOSOMAL_DOMINANT:
        compatible = len(pathogenic_variants) >= 1
        limitations.append("Parental segregation data not available — cannot confirm de novo vs inherited.")

    elif suspected_model == InheritanceModel.AUTOSOMAL_RECESSIVE:
        if len(pathogenic_variants) >= 2:
            compatible = True
        elif len(pathogenic_variants) == 1:
            compatible = None
            limitations.append(
                "Only one heterozygous variant found — second hit required for AR compatibility."
            )
        else:
            compatible = False
        limitations.append("Parental testing required to confirm compound het vs trans configuration.")

    elif suspected_model in (InheritanceModel.X_LINKED, InheritanceModel.X_LINKED_RECESSIVE):
        if sex == "male":
            compatible = len(pathogenic_variants) >= 1
        elif sex == "female":
            compatible = None
            limitations.append(
                "X-linked in female requires carrier assessment — "
                "two hits or X-inactivation data needed."
            )
        else:
            compatible = None
            limitations.append("Patient sex unknown — X-linked compatibility cannot be determined.")

    elif suspected_model == InheritanceModel.DE_NOVO:
        compatible = len(pathogenic_variants) >= 1
        limitations.append(
            "De novo status requires trio sequencing (parental) — not confirmed from VCF alone."
        )

    else:
        compatible = None
        limitations.append("Inheritance model unknown — compatibility cannot be assessed.")

    return InheritanceCompatibility(
        gene=gene,
        suspected_model=suspected_model,
        compatible=compatible,
        limitations=limitations,
    )


def run(state: CaseState) -> CaseState:
    family_history = state.input.family_history or ""
    sex = state.input.sex
    consanguinity = state.input.consanguinity

    suspected_model = infer_inheritance_model(family_history)

    if suspected_model == InheritanceModel.UNKNOWN:
        state.add_warning(
            "Inheritance model could not be inferred from family history. "
            "Compatibility assessment will be marked uncertain."
        )

    if consanguinity:
        state.add_warning("Consanguinity reported — autosomal recessive inheritance more likely.")
        if suspected_model == InheritanceModel.UNKNOWN:
            suspected_model = InheritanceModel.AUTOSOMAL_RECESSIVE

    variants_by_gene: dict[str, list[AnnotatedVariant]] = {}
    for v in state.annotated_variants:
        gene = v.gene or "unknown"
        variants_by_gene.setdefault(gene, []).append(v)

    candidate_genes = list(variants_by_gene.keys())
    if state.input.candidate_gene:
        if state.input.candidate_gene not in candidate_genes:
            candidate_genes.append(state.input.candidate_gene)

    compatibilities: list[InheritanceCompatibility] = []
    for gene in candidate_genes:
        if gene == "unknown":
            continue
        compat = _check_compatibility(
            gene,
            suspected_model,
            variants_by_gene.get(gene, []),
            sex,
        )
        compatibilities.append(compat)

    state.inheritance_results = compatibilities

    if not family_history:
        state.add_safety_flag(
            "No family history provided — inheritance compatibility is uncertain for all genes."
        )

    state.provenance.append(ProvenanceItem(
        component="inheritance",
        tool_name="GeneResolver inheritance rule engine",
        tool_version="1.0",
    ))
    state.mark_done("inheritance")
    return state
