from __future__ import annotations

import logging

from generesolver.integrations import (
    clingen_gene_validity, clingen_haploinsufficiency, hpo_db, lr_score, orphanet, vep_api,
)
from generesolver.schemas.case import CaseState
from generesolver.schemas.variant import AnnotatedVariant
from generesolver.schemas.variant_disease_context import (
    AllelicVariantEvidence,
    FamilyEvidence,
    VariantDiseaseContext,
)

logger = logging.getLogger(__name__)

_RULESET_VERSION = "ACMG-AMP-2015+ClinGen-SVI"


_MECHANISM_BY_ASSOCIATION_TYPE = {
    "Disease-causing germline mutation(s) (loss of function) in": "loss_of_function",
    "Disease-causing germline mutation(s) (gain of function) in": "gain_of_function",
}


def _gene_disease_pairs(gene: str) -> list[tuple[str, str | None, str | None]]:
    pairs: list[tuple[str, str | None, str | None]] = []
    seen: set[str] = set()
    for orpha_id, assoc in orphanet.get_diseases_for_gene(gene):
        if orpha_id in seen:
            continue
        seen.add(orpha_id)
        disease = orphanet.get_by_orpha_id(orpha_id)
        mechanism = _MECHANISM_BY_ASSOCIATION_TYPE.get(assoc.association_type)
        if mechanism is None:
            mechanism = clingen_haploinsufficiency.get_lof_mechanism(gene)
        pairs.append((orpha_id, disease.label if disease else None, mechanism))
    return pairs


def _matched_hpo_terms(gene: str, patient_hpo_ids: list[str]) -> list[str]:
    gene_hpo_ids = hpo_db.get_merged_gene_hpo_ids(gene)
    if not gene_hpo_ids:
        return []
    patient_int_ids: set[int] = set()
    for hpo_id in patient_hpo_ids:
        try:
            patient_int_ids.add(int(hpo_id.split(":")[1]))
        except (ValueError, IndexError):
            continue
    shared = sorted(gene_hpo_ids & patient_int_ids)
    return [f"HP:{tid:07d}" for tid in shared]


def _allele_fraction(v: AnnotatedVariant) -> float | None:
    if v.allele_depth_ref is None or v.allele_depth_alt is None:
        return None
    total = v.allele_depth_ref + v.allele_depth_alt
    if total <= 0:
        return None
    return round(v.allele_depth_alt / total, 4)


def _source_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    cache_version = getattr(vep_api, "_VEP_CACHE_VERSION", None)
    if cache_version:
        versions["vep_cache_version"] = str(cache_version)
    return versions


def run(state: CaseState) -> CaseState:
    if not state.annotated_variants or not state.ranked_genes:
        state.add_warning("Variant-disease context builder skipped: no annotated variants or ranked genes.")
        state.mark_done("variant_disease_context")
        return state

    genome_build = state.vcf_context.genome_build.value if state.vcf_context else "unknown"
    patient_hpo_ids = state.hpo_profile.present_ids
    patient_sex = getattr(state.input, "sex", None)

    absent_hpo_ids = state.hpo_profile.absent_ids
    _raw_disease_scores = dict(lr_score.score_diseases(patient_hpo_ids, absent_hpo_ids))
    if _raw_disease_scores:
        _dmin, _dmax = min(_raw_disease_scores.values()), max(_raw_disease_scores.values())
        _dspan = _dmax - _dmin
        disease_phenotype_scores = (
            {d: (s - _dmin) / _dspan for d, s in _raw_disease_scores.items()} if _dspan > 0
            else {d: 0.0 for d in _raw_disease_scores}
        )
    else:
        disease_phenotype_scores = {}

    family_evidence = None
    _family_fields = (
        state.de_novo_stated, state.de_novo_confirmed, state.segregation_observed,
        state.n_affected_segregating, state.n_unaffected_nonsegregating, state.consanguinity,
        state.unaffected_carrier_observed, state.unaffected_carrier_adult,
    )
    if any(f is not None for f in _family_fields):
        family_evidence = FamilyEvidence(
            segregation_observed=state.segregation_observed,
            n_affected_segregating=state.n_affected_segregating,
            n_unaffected_nonsegregating=state.n_unaffected_nonsegregating,
            de_novo_confirmed=state.de_novo_confirmed,
            de_novo_stated=state.de_novo_stated,
            consanguinity=state.consanguinity,
            unaffected_carrier_observed=state.unaffected_carrier_observed,
            unaffected_carrier_adult=state.unaffected_carrier_adult,
            source="demographics (narrative, llm-extracted)",
        )

    variants_by_gene: dict[str, list[AnnotatedVariant]] = {}
    for v in state.annotated_variants:
        if v.gene:
            variants_by_gene.setdefault(v.gene, []).append(v)
    if not any(gr.gene in variants_by_gene for gr in state.ranked_genes):
        state.add_warning(
            "No annotated variant overlaps a ranked gene. Variant-disease and ACMG "
            "assessment cannot proceed for this VCF sample; increase the variant cap "
            "or use full VCF annotation."
        )

    inheritance_by_gene = {ic.gene: ic for ic in state.inheritance_results}
    source_versions = _source_versions()

    contexts: list[VariantDiseaseContext] = []
    n_skipped_no_transcript = 0
    n_skipped_no_disease = 0

    for gr in state.ranked_genes:
        gene_variants = variants_by_gene.get(gr.gene, [])
        if not gene_variants:
            continue

        disease_pairs = _gene_disease_pairs(gr.gene)
        if not disease_pairs:
            n_skipped_no_disease += len(gene_variants)
            continue

        matched_hpo = _matched_hpo_terms(gr.gene, patient_hpo_ids)
        inheritance = inheritance_by_gene.get(gr.gene)

        for v in gene_variants:
            if not v.transcript:
                n_skipped_no_transcript += 1
                continue

            other_alleles = [
                AllelicVariantEvidence(
                    variant_id=other.variant_id,
                    zygosity=other.zygosity,
                    classification_support=(
                        other.acmg_result.classification_support.value
                        if other.acmg_result else None
                    ),
                )
                for other in gene_variants
                if other.variant_id != v.variant_id
            ]

            for disease_id, disease_name, disease_mechanism in disease_pairs:
                validity = clingen_gene_validity.get_validity(gr.gene, disease_name)
                contexts.append(VariantDiseaseContext(
                    variant_id=v.variant_id,
                    gene=gr.gene,
                    disease_id=disease_id,
                    disease_name=disease_name,
                    transcript=v.transcript,
                    genome_build=genome_build,
                    ruleset_version=_RULESET_VERSION,
                    inheritance=(inheritance.suspected_model.value if inheritance else None),
                    disease_mechanism=disease_mechanism,
                    gene_disease_validity=(validity["classification"] if validity else None),
                    patient_hpo_terms=list(patient_hpo_ids),
                    matched_hpo_terms=matched_hpo,
                    phenotype_match_score=disease_phenotype_scores.get(disease_id, 0.0),
                    zygosity=v.zygosity,
                    patient_sex=patient_sex,
                    allele_fraction=_allele_fraction(v),
                    other_alleles=other_alleles,
                    family_evidence=family_evidence,
                    source_versions=dict(source_versions),
                ))

    if n_skipped_no_transcript:
        state.add_warning(
            f"{n_skipped_no_transcript} variant(s) skipped for variant-disease "
            "context building: no transcript annotated."
        )
    if n_skipped_no_disease:
        state.add_warning(
            f"{n_skipped_no_disease} variant(s) skipped for variant-disease "
            "context building: no known gene-disease association (Orphanet)."
        )

    state.variant_disease_contexts = contexts
    logger.info("Built %d variant-disease context(s) for %d ranked gene(s).",
                len(contexts), len(state.ranked_genes))

    state.mark_done("variant_disease_context")
    return state
