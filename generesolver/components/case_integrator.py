from __future__ import annotations

import logging

from generesolver.schemas.case import CaseState

logger = logging.getLogger(__name__)

_TOP_DISEASES = 3


def _gene_rank(state: CaseState) -> dict[str, int]:
    return {g.gene: g.rank for g in (state.ranked_genes or [])}


def _variants_by_gene(state: CaseState) -> dict[str, list]:
    out: dict[str, list] = {}
    for v in (getattr(state, "annotated_variants", None) or []):
        gene = getattr(v, "gene", None)
        if gene:
            out.setdefault(gene, []).append(v)
    return out


def _confidence(disease: dict, gene_supported: bool, variant_supported: bool,
                tied: bool) -> str:
    pp = float(disease.get("posttest_prob", 0.0) or 0.0)
    if variant_supported and not tied:
        return "high"
    if gene_supported and pp >= 0.4 and not tied:
        return "moderate"
    if tied or pp < 0.25:
        return "low"
    return "moderate"


def run(state: CaseState) -> CaseState:
    diseases = (getattr(state, "ranked_diseases", None) or [])[:_TOP_DISEASES]
    grank = _gene_rank(state)
    vbg = _variants_by_gene(state)

    hypotheses: list[dict] = []
    for d in diseases:
        if not isinstance(d, dict):
            continue
        pairs = d.get("candidate_disease_gene_pairs") or [{"disease_id": d.get("disease_id"),
                                                           "name": d.get("name"),
                                                           "genes": d.get("genes") or []}]
        cand_genes = [g for p in pairs for g in (p.get("genes") or [])]
        best_gene = min((g for g in cand_genes if g in grank),
                        key=lambda g: grank[g], default=(cand_genes[0] if cand_genes else None))
        variant = None
        if best_gene and best_gene in vbg:
            variant = getattr(vbg[best_gene][0], "variant_id", None)

        tied = bool(d.get("phenotype_tie_ids"))
        evidence_for, evidence_against = [], []
        st = d.get("supporting_terms") or []
        if st:
            evidence_for.append(f"{len(st)} matching HPO term(s); top log-LR "
                                f"{max((t.get('log_lr', 0) for t in st), default=0):.1f}")
        if d.get("evidence", {}).get("biochemical"):
            evidence_for.append("specific biochemical marker")
        if best_gene and best_gene in grank:
            evidence_for.append(f"causal gene {best_gene} ranked #{grank[best_gene]} by the gene pipeline")
        if variant:
            evidence_for.append(f"candidate variant {variant} in {best_gene}")
        if tied:
            evidence_against.append(
                f"phenotype does not separate this from {len(d['phenotype_tie_ids'])} "
                "sibling subtype(s)")
        if best_gene and best_gene not in grank:
            evidence_against.append(f"causal gene {best_gene} not corroborated by variant/gene evidence")

        missing = []
        if not variant:
            missing.append("molecular/variant confirmation")
        if tied:
            missing.append("biochemical test to resolve subtype")

        hypotheses.append({
            "disease_id": d.get("disease_id"),
            "disease": d.get("name"),
            "gene": best_gene,
            "variant": variant,
            "phenotype_tie_ids": d.get("phenotype_tie_ids") or [],
            "evidence_for": evidence_for,
            "evidence_against": evidence_against,
            "missing_decisive_tests": missing,
            "confidence": _confidence(d, bool(best_gene), bool(variant), tied),
            "posttest_prob": d.get("posttest_prob"),
        })

    diag = getattr(state, "diagnostics", None)
    if isinstance(diag, dict):
        diag["final_integrated"] = {"final_diagnostic_hypotheses": hypotheses}
    logger.info("case_integrator: produced %d integrated hypothesis(es)", len(hypotheses))
    state.mark_done("case_integrator")
    return state
