from __future__ import annotations

import logging

from generesolver.schemas.case import CaseState
from generesolver.schemas.gene import GeneRank, InheritanceModel
from generesolver.config import (
    RANK_WEIGHT_PHENOTYPE,
    RANK_WEIGHT_VARIANT,
    RANK_WEIGHT_GENE_DISEASE,
    RANK_WEIGHT_LITERATURE,
    RANK_WEIGHT_GWAS,
    RANK_WEIGHT_CONSTRAINT,
    RANK_WEIGHT_PATHWAY,
    PATHWAY_ANCHOR_TOP_K,
    DISEASE_MATCH_BOOST_MAX,
    LR_RANKER_ENABLED,
)
from generesolver.integrations import gnomad_constraint, hpo_db, reactome_db

_HPO_Y_LINKED = 1450
_HPO_X_LINKED_RECESSIVE = 1419

_NEGATED_HPO_PENALTY_MAX = 0.05
_SEX_PENALTY_Y_LINKED_FEMALE = 0.20
_SEX_PENALTY_X_RECESSIVE_FEMALE = 0.05


def _negated_hpo_penalty(gene_hpo_ids: set[int], absent_int_ids: set[int]) -> float:
    if not absent_int_ids or not gene_hpo_ids:
        return 0.0
    matches = len(gene_hpo_ids & absent_int_ids)
    if matches == 0:
        return 0.0
    return -_NEGATED_HPO_PENALTY_MAX * (matches / len(absent_int_ids))


def _sex_penalty(gene_hpo_ids: set[int], patient_sex: str | None) -> float:
    if not patient_sex or not gene_hpo_ids:
        return 0.0
    sex = patient_sex.strip().lower()
    if sex != "female":
        return 0.0
    if _HPO_Y_LINKED in gene_hpo_ids:
        return -_SEX_PENALTY_Y_LINKED_FEMALE
    if _HPO_X_LINKED_RECESSIVE in gene_hpo_ids:
        return -_SEX_PENALTY_X_RECESSIVE_FEMALE
    return 0.0


def _constraint_weight(inheritance, base_weight: float) -> float:
    if inheritance is not None and inheritance.suspected_model == InheritanceModel.AUTOSOMAL_RECESSIVE:
        return 0.0
    return base_weight


def _disease_match_boost(score_disease_match: float) -> float:
    if score_disease_match <= 0.0 or DISEASE_MATCH_BOOST_MAX <= 0.0:
        return 0.0
    return DISEASE_MATCH_BOOST_MAX * max(0.0, min(1.0, score_disease_match))

logger = logging.getLogger(__name__)

_LITERATURE_SUPPORT_SCORES = {
    "strong": 1.0,
    "moderate": 0.6,
    "weak": 0.3,
    "none": 0.0,
    "unknown": 0.0,
}


def run(state: CaseState) -> CaseState:
    if not state.ranked_genes:
        logger.info("gene_ranker: no candidate genes to rank")
        state.mark_done("gene_ranker")
        return state

    bio_index = {be.gene: be for be in state.biological_evidence}
    lit_index = {le.gene: le for le in state.literature_evidence}
    inheritance_index = {ic.gene: ic for ic in state.inheritance_results}

    has_variant_signal = any(gr.score_variant > 0 for gr in state.ranked_genes)
    if has_variant_signal:
        w_phenotype = RANK_WEIGHT_PHENOTYPE
        w_variant = RANK_WEIGHT_VARIANT
        w_gene_disease = RANK_WEIGHT_GENE_DISEASE
        w_literature = RANK_WEIGHT_LITERATURE
        w_gwas = RANK_WEIGHT_GWAS
        w_constraint = RANK_WEIGHT_CONSTRAINT
        w_pathway = RANK_WEIGHT_PATHWAY
    elif LR_RANKER_ENABLED:
        w_phenotype = 0.80
        w_variant = 0.0
        w_gene_disease = 0.05
        w_literature = 0.0
        w_gwas = 0.0
        w_constraint = 0.10
        w_pathway = 0.05
    else:
        w_phenotype = RANK_WEIGHT_PHENOTYPE + 0.15
        w_variant = 0.0
        w_gene_disease = RANK_WEIGHT_GENE_DISEASE
        w_literature = RANK_WEIGHT_LITERATURE
        w_gwas = RANK_WEIGHT_GWAS
        w_constraint = RANK_WEIGHT_CONSTRAINT + 0.04
        w_pathway = RANK_WEIGHT_PATHWAY + 0.07

    raw_pheno = [gr.score_phenotype for gr in state.ranked_genes]
    pmin, pmax = min(raw_pheno), max(raw_pheno)
    pspan = pmax - pmin
    norm_pheno = (
        [(p - pmin) / pspan for p in raw_pheno] if pspan > 0
        else [0.0] * len(raw_pheno)
    )

    anchors = sorted(
        state.ranked_genes, key=lambda g: g.score_phenotype, reverse=True
    )[:PATHWAY_ANCHOR_TOP_K]
    anchor_symbols = [a.gene for a in anchors]
    anchor_pathways = reactome_db.build_anchor_pathway_set(anchor_symbols)

    absent_int_ids: set[int] = set()
    for hpo_id in state.hpo_profile.absent_ids:
        try:
            absent_int_ids.add(int(hpo_id.split(":")[1]))
        except (ValueError, IndexError):
            continue
    patient_sex = getattr(state.input, "sex", None)

    completed: list[GeneRank] = []
    for idx, gr in enumerate(state.ranked_genes):
        gene = gr.gene
        gene_hpo_ids = hpo_db.get_merged_gene_hpo_ids(gene)

        score_phenotype_normalized = norm_pheno[idx]

        bio = bio_index.get(gene)
        score_gene_disease = bio.support_score if bio else 0.0

        lit = lit_index.get(gene)
        score_literature = (
            _LITERATURE_SUPPORT_SCORES.get(lit.support_level, 0.0) if lit else 0.0
        )

        score_gwas = _compute_gwas_score(state, gene)
        score_constraint = gnomad_constraint.constraint_score(gene)
        score_pathway = reactome_db.pathway_overlap_score(gene, anchor_pathways)
        inheritance = inheritance_index.get(gene)
        gated_w_constraint = _constraint_weight(inheritance, w_constraint)

        final_score = (
            score_phenotype_normalized * w_phenotype
            + gr.score_variant * w_variant
            + score_gene_disease * w_gene_disease
            + score_literature * w_literature
            + score_gwas * w_gwas
            + score_constraint * gated_w_constraint
            + score_pathway * w_pathway
        )
        final_score += _negated_hpo_penalty(gene_hpo_ids, absent_int_ids)
        final_score += _sex_penalty(gene_hpo_ids, patient_sex)
        final_score += _disease_match_boost(gr.score_disease_match)
        final_score = round(min(1.0, max(0.0, final_score)), 4)

        known_diseases = (bio.disease_associations[:5] if bio else [])
        warnings = list(gr.warnings)
        if bio and bio.biological_support == "weak":
            warnings.append("Limited biological evidence for this gene.")

        completed.append(
            GeneRank(
                rank=0,
                gene=gene,
                hgnc_id=gr.hgnc_id,
                final_score=final_score,
                confidence=_confidence(final_score),
                score_phenotype=gr.score_phenotype,
                score_variant=gr.score_variant,
                score_gene_disease=score_gene_disease,
                score_literature=score_literature,
                score_gwas=score_gwas,
                score_disease_match=gr.score_disease_match,
                best_variant_id=gr.best_variant_id,
                seeded_by_diseases=list(gr.seeded_by_diseases),
                inheritance=inheritance,
                known_disease_associations=known_diseases,
                prioritization_reason=_reason(gr, score_gene_disease, score_gwas, bio),
                warnings=warnings,
                requires_expert_review=True,
            )
        )

    completed.sort(key=lambda g: g.final_score, reverse=True)
    for i, gr in enumerate(completed, start=1):
        gr.rank = i

    state.ranked_genes = completed
    state.confidence_level = completed[0].confidence if completed else "insufficient"
    state.mark_done("gene_ranker")
    return state


def _compute_gwas_score(state: CaseState, gene: str) -> float:
    gene_evidence = [e for e in state.gwas_evidence if e.query == gene]
    if not gene_evidence:
        return 0.0
    min_p = min(
        (e.p_value for e in gene_evidence if e.p_value is not None),
        default=None,
    )
    if min_p is None:
        return 0.5
    if min_p <= 5e-8:
        return 1.0
    if min_p <= 1e-5:
        return 0.7
    if min_p <= 1e-3:
        return 0.4
    return 0.2


def _confidence(score: float) -> str:
    if score >= 0.70:
        return "high"
    if score >= 0.50:
        return "moderate"
    if score >= 0.30:
        return "low"
    return "insufficient"


def _reason(
    gr: GeneRank,
    score_gene_disease: float,
    score_gwas: float,
    bio: object,
) -> str:
    parts: list[str] = []
    if gr.score_phenotype >= 0.5:
        parts.append("strong phenotype overlap")
    elif gr.score_phenotype >= 0.2:
        parts.append("partial phenotype overlap")
    if gr.score_variant >= 0.8:
        parts.append("pathogenic variant detected")
    elif gr.score_variant >= 0.5:
        parts.append("VUS or moderate-impact variant")
    if score_gene_disease >= 0.5:
        parts.append("known disease association in Open Targets")
    if score_gwas >= 0.7:
        parts.append("genome-wide significant GWAS hit")
    if gr.score_disease_match >= 0.7 and gr.seeded_by_diseases:
        first = gr.seeded_by_diseases[0]
        n_more = len(gr.seeded_by_diseases) - 1
        suffix = f" (+{n_more} more)" if n_more else ""
        parts.append(f"linked to clinician-named disease {first}{suffix} (Orphanet)")
    elif gr.score_disease_match > 0.0 and gr.seeded_by_diseases:
        parts.append("candidate-gene link to a clinician-named disease (Orphanet)")
    return "; ".join(parts) if parts else "low overall evidence"
