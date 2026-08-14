from __future__ import annotations

import logging

from generesolver import config
from generesolver.components.vcf_input import load_normalized_variants, NormalizedVariant
from generesolver.integrations import vep_api, gnomad_api, clinvar_local, gnomad_popmax_local, revel_local, conservation_local, mane_local
from generesolver.schemas.case import CaseState
from generesolver.schemas.genomic import GenomeBuild, QCStatus
from generesolver.schemas.report import ProvenanceItem
from generesolver.schemas.variant import AnnotatedVariant

logger = logging.getLogger(__name__)

_CANDIDATE_RARITY_CEILING = 0.01

_HIGH_MODERATE_CONSEQUENCES = {
    "transcript_ablation",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "stop_gained",
    "frameshift_variant",
    "stop_lost",
    "start_lost",
    "transcript_amplification",
    "inframe_insertion",
    "inframe_deletion",
    "missense_variant",
    "protein_altering_variant",
    "splice_region_variant",
}

_SYNONYMOUS_CONSEQUENCES = {"synonymous_variant", "stop_retained_variant"}



def _normalize_variant_key(chrom: str, pos: int, ref: str, alt: str) -> str:
    return f"{chrom.lstrip('chr')}:{pos}:{ref}:{alt}"


def _build_variant_id(chrom: str, pos: int, ref: str, alt: str) -> str:
    return f"{chrom}:{pos}:{ref}:{alt}"


def _get_vep_result(raw_variant: NormalizedVariant, vep_results: dict) -> dict:
    chrom, pos, ref, alt = raw_variant.chrom, raw_variant.pos, raw_variant.ref, raw_variant.alt
    normalized_key = _normalize_variant_key(chrom, pos, ref, alt)
    display_id = _build_variant_id(chrom, pos, ref, alt)
    
    return vep_results.get(normalized_key) or vep_results.get(display_id) or {}


def _has_relevant_consequence(consequence: str, impact: str) -> bool:
    return (
        consequence in _HIGH_MODERATE_CONSEQUENCES or 
        impact in ("HIGH", "MODERATE")
    )


def _resolve_clinvar_fields(
    vep: dict,
    genome_build: str,
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
) -> tuple[str | None, str | None, str | None, str | None, str | None, str | None]:
    local_clinvar = clinvar_local.get_variant(genome_build, chrom, pos, ref, alt)
    if local_clinvar:
        return (
            local_clinvar["clinvar_significance"],
            local_clinvar["clinvar_variation_id"],
            local_clinvar["clinvar_review_status"],
            local_clinvar["clinvar_condition"],
            local_clinvar["clinvar_allele_id"],
            "local_clinvar_vcf",
        )
    
    clinvar_sig = vep.get("clinvar_significance")
    return (
        clinvar_sig,
        vep.get("clinvar_id"),
        None,
        None,
        None,
        "vep" if clinvar_sig else None,
    )


def _resolve_popmax_allele_number(
    vep: dict,
    slivar: dict,
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    is_relevant_consequence: bool,
) -> tuple[float | None, int | None, str]:
    gnomad_af_popmax = vep.get("gnomad_af_popmax")
    gnomad_af_popmax_an = vep.get("gnomad_af_popmax_allele_number")
    gnomad_frequency_status = vep.get("gnomad_frequency_status") or "not_evaluated"
    
    if gnomad_af_popmax is None and slivar.get("gnomad_popmax_af") is not None:
        gnomad_af_popmax = slivar["gnomad_popmax_af"]
        if gnomad_frequency_status != "observed":
            gnomad_frequency_status = "observed"
    
    if gnomad_af_popmax_an is None and slivar.get("gnomad_popmax_an") is not None:
        gnomad_af_popmax_an = slivar["gnomad_popmax_an"]
        if gnomad_af_popmax is None:
            gnomad_af_popmax = slivar.get("gnomad_popmax_af")
        if gnomad_frequency_status != "observed":
            gnomad_frequency_status = "observed"
    
    if (
        gnomad_af_popmax_an is None
        and is_relevant_consequence
        and not config.LOCAL_MODE
    ):
        gn = gnomad_api.get_allele_frequency(chrom, pos, ref, alt, vep.get("genome_build") or "")
        if gn.get("af_popmax_allele_number") is not None:
            gnomad_af_popmax = gn.get("af_popmax")
            gnomad_af_popmax_an = gn.get("af_popmax_allele_number")
    
    return gnomad_af_popmax, gnomad_af_popmax_an, gnomad_frequency_status


def _resolve_gnomad_overall_af(
    vep: dict,
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    is_relevant_consequence: bool,
) -> tuple[float | None, str]:
    gnomad_af = vep.get("gnomad_af")
    gnomad_frequency_status = vep.get("gnomad_frequency_status") or "not_evaluated"
    
    if gnomad_af is None and is_relevant_consequence and not config.LOCAL_MODE:
        gn = gnomad_api.get_allele_frequency(chrom, pos, ref, alt, vep.get("genome_build") or "")
        if gn.get("af") is not None:
            gnomad_af = gn.get("af")
            gnomad_frequency_status = "observed"
    
    return gnomad_af, gnomad_frequency_status



def _check_prerequisites(state: CaseState) -> tuple[bool, str, GenomeBuild]:
    if not state.vcf_context:
        state.add_warning("Variant annotator skipped: no VCF context.")
        state.mark_done("variant_annotator")
        return False, "", GenomeBuild.UNKNOWN

    if state.vcf_context.qc_status == QCStatus.FAIL:
        state.add_warning("Variant annotator skipped: VCF QC failed.")
        state.mark_done("variant_annotator")
        return False, "", GenomeBuild.UNKNOWN

    vcf_path = state.vcf_context.file_path
    if not vcf_path:
        state.add_warning("Variant annotator skipped: no VCF file path in context.")
        state.mark_done("variant_annotator")
        return False, "", GenomeBuild.UNKNOWN

    genome_build_enum = state.vcf_context.genome_build
    genome_build = genome_build_enum.value
    if genome_build == GenomeBuild.UNKNOWN.value:
        state.add_warning("Annotating variants without confirmed genome build — results may be unreliable.")

    return True, genome_build, genome_build_enum


def _load_and_validate_variants(vcf_path: str, genome_build_enum: GenomeBuild, state: CaseState) -> list[NormalizedVariant] | None:
    raw_variants, n_skipped_structural = load_normalized_variants(vcf_path, genome_build_enum)
    if n_skipped_structural:
        state.add_warning(
            f"Skipped {n_skipped_structural} structural/symbolic-ALT variant(s) "
            "(e.g. <DEL>, <DUP>, breakends) — not supported by GeneResolver's "
            "point-variant annotation pipeline."
        )
    if not raw_variants:
        state.add_warning("No PASS variants loaded from VCF.")
        state.mark_done("variant_annotator")
        return None
    return raw_variants


def _warn_about_missing_resources(state: CaseState, genome_build: str, slivar_results: dict) -> None:
    if not gnomad_popmax_local.is_available(genome_build):
        state.add_warning(
            f"Local gnomAD-popmax/SpliceAI reference files not available for "
            f"{genome_build} (GRCh37 only) — PM2/BA1 rely solely on VEP's own "
            "popmax fallback and PP3/BP4's splice half is unavailable."
        )
    if not gnomad_popmax_local.an_index_available():
        state.add_warning(
            "Local gnomAD popmax allele-number index not available — PM2/BA1 "
            "cannot confirm adequate population coverage (frequency alone is "
            "not enough per ClinGen's guidance), so BA1's stand-alone benign "
            "call stays unavailable regardless of how common a variant is. "
            "Run scripts/download_gnomad_exomes_sites.py then "
            "scripts/build_gnomad_an_index.py to fetch it."
        )
    if not revel_local.is_available():
        state.add_warning(
            "Local REVEL score file not available — PP3/BP4's missense half "
            "is unavailable. Run scripts/download_revel.py to fetch."
        )
    if not mane_local.is_available():
        state.add_warning(
            "Local MANE summary file not available — PVS1's biological-relevance "
            "gate (mane_status) relies solely on VEP's own MANE annotation, which "
            "is empty for GRCh37 cases (MANE is a GRCh38 resource). Run "
            "scripts/download_mane.py to fetch."
        )
    if not conservation_local.is_available():
        state.add_warning(
            "Local phyloP conservation track not available — BP7's "
            "conservation half is unavailable. Run scripts/download_ucsc_phylop.py to fetch."
        )
    if not clinvar_local.is_available(genome_build):
        state.add_warning(
            f"Local ClinVar VCF not available for {genome_build} — ClinVar "
            "significance falls back to whatever VEP's own cache carries "
            "(if any). Run scripts/download_clinvar.py for the authoritative "
            "local source."
        )


def _extract_synonymous_positions(
    raw_variants: list[NormalizedVariant],
    vep_results: dict,
) -> list[tuple[str, int]]:
    synonymous_positions = []
    for v in raw_variants:
        vep = _get_vep_result(v, vep_results)
        if vep.get("consequence") in _SYNONYMOUS_CONSEQUENCES:
            synonymous_positions.append((v.chrom, v.pos))
    return synonymous_positions


def _resolve_gnomad_homozygote_count(vep: dict, slivar: dict) -> int | None:
    if vep.get("gnomad_homozygote_count") is not None:
        return vep.get("gnomad_homozygote_count")
    if slivar.get("gnomad_nhomalt") is not None:
        return int(slivar["gnomad_nhomalt"])
    return None


def _resolve_revel_score(vep: dict, consequence: str, chrom: str, pos: int, ref: str, alt: str) -> float | None:
    if vep.get("revel_score") is not None:
        return vep.get("revel_score")
    if consequence == "missense_variant":
        return revel_local.get_revel_score(chrom, pos, ref, alt)
    return None


def _resolve_spliceai_score(vep: dict, slivar: dict) -> float | None:
    if vep.get("spliceai_score") is not None:
        return vep.get("spliceai_score")
    return slivar.get("spliceai_score")


def _create_annotated_variant(
    rv: NormalizedVariant,
    vep: dict,
    slivar: dict,
    conservation_results: dict,
    genome_build: str,
) -> AnnotatedVariant:
    chrom, pos, ref, alt = rv.chrom, rv.pos, rv.ref, rv.alt
    variant_id = _build_variant_id(chrom, pos, ref, alt)
    
    consequence = vep.get("consequence", "") or vep.get("most_severe_consequence", "")
    impact = vep.get("impact", "")
    
    clinvar_sig, clinvar_id, clinvar_review, clinvar_cond, clinvar_allele_id, clinvar_src = (
        _resolve_clinvar_fields(vep, genome_build, chrom, pos, ref, alt)
    )
    
    is_relevant = _has_relevant_consequence(consequence, impact)
    gnomad_af, gnomad_freq_status = _resolve_gnomad_overall_af(
        vep, chrom, pos, ref, alt, is_relevant
    )
    gnomad_af_popmax, gnomad_af_popmax_an, gnomad_freq_status_popmax = _resolve_popmax_allele_number(
        vep, slivar, chrom, pos, ref, alt, is_relevant
    )
    
    return AnnotatedVariant(
        variant_id=variant_id,
        gene=vep.get("gene"),
        transcript=vep.get("transcript"),
        hgvs_c=vep.get("hgvs_c"),
        hgvs_p=vep.get("hgvs_p"),
        consequence=consequence,
        all_consequence_terms=vep.get("all_consequence_terms") or [],
        is_splice_relevant=bool(vep.get("is_splice_relevant")),
        mane_status=vep.get("mane_status") or mane_local.get_mane_status(vep.get("transcript")),
        protein_position=vep.get("protein_position"),
        exon=vep.get("exon"),
        intron=vep.get("intron"),
        impact=impact,
        clinvar_significance=clinvar_sig,
        clinvar_id=clinvar_id,
        clinvar_review_status=clinvar_review,
        clinvar_condition=clinvar_cond,
        clinvar_allele_id=clinvar_allele_id,
        clinvar_source=clinvar_src,
        gnomad_af=gnomad_af,
        gnomad_af_popmax=gnomad_af_popmax,
        gnomad_af_by_population=vep.get("gnomad_af_by_population") or {},
        gnomad_allele_count=vep.get("gnomad_allele_count"),
        gnomad_allele_number=vep.get("gnomad_allele_number"),
        gnomad_af_popmax_allele_number=gnomad_af_popmax_an,
        gnomad_homozygote_count=_resolve_gnomad_homozygote_count(vep, slivar),
        gnomad_filter_status=vep.get("gnomad_filter_status"),
        gnomad_coverage=vep.get("gnomad_coverage"),
        gnomad_frequency_status=gnomad_freq_status_popmax or gnomad_freq_status,
        sift_prediction=vep.get("sift"),
        sift_score=vep.get("sift_score"),
        polyphen_prediction=vep.get("polyphen"),
        polyphen_score=vep.get("polyphen_score"),
        revel_score=_resolve_revel_score(vep, consequence, chrom, pos, ref, alt),
        spliceai_score=_resolve_spliceai_score(vep, slivar),
        domains=vep.get("domains") or [],
        conservation_score=conservation_results.get(_normalize_variant_key(chrom, pos, ref, alt)),
        genotype=rv.genotype,
        zygosity=rv.zygosity,
        allele_depth_ref=rv.allele_depth_ref,
        allele_depth_alt=rv.allele_depth_alt,
        read_depth=rv.read_depth,
        quality=rv.quality,
        filter_status=rv.filter_status,
        was_multiallelic=rv.was_multiallelic,
        was_indel_normalized=rv.was_indel_normalized,
        ref_matches_reference=rv.ref_matches_reference,
    )


def _add_variant_warnings(variant: AnnotatedVariant, raw_variant: NormalizedVariant) -> None:
    if variant.gnomad_af is None:
        variant.warnings.append("gnomAD frequency not available — variant may be absent or lookup failed.")
    if not variant.gene:
        variant.warnings.append("Gene could not be determined from VEP annotation.")
    if raw_variant.ref_matches_reference is False:
        variant.warnings.append(
            "REF allele does not match the reference FASTA at this position — "
            "possible strand/build mismatch; verify before use."
        )


def run(state: CaseState) -> CaseState:
    should_continue, genome_build, genome_build_enum = _check_prerequisites(state)
    if not should_continue:
        return state

    vcf_path = state.vcf_context.file_path

    raw_variants = _load_and_validate_variants(vcf_path, genome_build_enum, state)
    if raw_variants is None:
        return state

    logger.info("Annotating %d PASS variants via VEP...", len(raw_variants))

    vep_results = vep_api.annotate_variants(
        [(v.chrom, v.pos, v.ref, v.alt) for v in raw_variants]
    )

    slivar_results = gnomad_popmax_local.annotate_batch(
        [(v.chrom, v.pos, v.ref, v.alt) for v in raw_variants], genome_build
    )

    synonymous_positions = _extract_synonymous_positions(raw_variants, vep_results)
    conservation_results = conservation_local.get_conservation_batch(synonymous_positions)

    _warn_about_missing_resources(state, genome_build, slivar_results)

    annotated: list[AnnotatedVariant] = []

    for rv in raw_variants:
        vep = _get_vep_result(rv, vep_results)
        
        consequence = vep.get("consequence", "") or vep.get("most_severe_consequence", "")
        impact = vep.get("impact", "")
        gnomad_af = vep.get("gnomad_af")

        is_relevant_consequence = _has_relevant_consequence(consequence, impact)
        is_rare = gnomad_af is None or gnomad_af < _CANDIDATE_RARITY_CEILING

        if not is_relevant_consequence and not is_rare:
            continue

        slivar = slivar_results.get(_normalize_variant_key(rv.chrom, rv.pos, rv.ref, rv.alt)) or {}

        variant = _create_annotated_variant(
            rv, vep, slivar, conservation_results, genome_build
        )

        _add_variant_warnings(variant, rv)

        annotated.append(variant)

    state.annotated_variants = annotated
    logger.info("Retained %d candidate variants after filtering.", len(annotated))

    if not annotated:
        state.add_warning("No candidate variants remained after annotation filtering.")

    state.provenance.append(ProvenanceItem(
        component="variant_annotator",
        tool_name="Ensembl VEP REST API + gnomAD GraphQL",
        tool_version="Ensembl VEP REST v12",
    ))
    if any(v.clinvar_source == "local_clinvar_vcf" for v in annotated):
        state.provenance.append(ProvenanceItem(
            component="variant_annotator",
            tool_name="Local ClinVar VCF (NCBI)",
            tool_version=genome_build,
        ))
    if slivar_results:
        state.provenance.append(ProvenanceItem(
            component="variant_annotator",
            tool_name="slivar (local Docker) + gnomAD v2.1.1 popmax/SpliceAI gnotate files",
            tool_version="slivar_hg37",
        ))
    if mane_local.is_available() and any(v.mane_status for v in annotated):
        state.provenance.append(ProvenanceItem(
            component="variant_annotator",
            tool_name="Local MANE summary file (NCBI/EBI)",
            tool_version="MANE_local",
        ))
    state.mark_done("variant_annotator")
    return state
