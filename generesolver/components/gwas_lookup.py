from __future__ import annotations

import logging

from generesolver.schemas.case import CaseState
from generesolver.schemas.variant import GWASEvidence
from generesolver.integrations import gwas_catalog

logger = logging.getLogger(__name__)


def run(state: CaseState) -> CaseState:
    genes: list[str] = [gr.gene for gr in state.ranked_genes]

    if state.input.candidate_gene and state.input.candidate_gene not in genes:
        genes.append(state.input.candidate_gene)

    if not genes:
        state.mark_done("gwas_lookup")
        return state

    all_evidence: list[GWASEvidence] = list(state.gwas_evidence)

    for gene in genes:
        try:
            raw = gwas_catalog.lookup(gene=gene)
            for item in raw:
                all_evidence.append(
                    GWASEvidence(
                        query=item.get("query", gene),
                        trait=item.get("trait"),
                        p_value=_safe_float(item.get("p_value")),
                        odds_ratio=_safe_float(item.get("odds_ratio")),
                        support_level=item.get("support_level", "supportive_context_only"),
                        source_db=item.get("source_db"),
                        warning=item.get(
                            "warning",
                            GWASEvidence.model_fields["warning"].default,
                        ),
                    )
                )
        except Exception as exc:
            logger.warning("GWAS lookup failed for gene %s: %s", gene, exc)

    state.gwas_evidence = all_evidence
    state.mark_done("gwas_lookup")
    return state


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
