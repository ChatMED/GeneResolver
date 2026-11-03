from enum import Enum
from pydantic import BaseModel, Field


class ACMGClassification(str, Enum):
    PATHOGENIC = "pathogenic"
    LIKELY_PATHOGENIC = "likely_pathogenic"
    VUS = "variant_of_uncertain_significance"
    LIKELY_BENIGN = "likely_benign"
    BENIGN = "benign"
    UNCLASSIFIED = "unclassified"


class ACMGResult(BaseModel):
    variant_id: str
    gene: str
    classification_support: ACMGClassification
    criteria_met: list[str] = Field(default_factory=list)
    criteria_failed: list[str] = Field(default_factory=list)
    requires_expert_review: bool = True
    warnings: list[str] = Field(default_factory=list)
    source: str = "acmg_classifier"


class GWASEvidence(BaseModel):
    query: str
    trait: str | None = None
    p_value: float | None = None
    odds_ratio: float | None = None
    evidence_type: str = "population_association"
    support_level: str = "supportive_context_only"
    source_db: str | None = None
    warning: str = (
        "GWAS evidence is population-level and does not establish "
        "Mendelian causality for this individual patient."
    )


class AnnotatedVariant(BaseModel):
    variant_id: str
    gene: str | None = None
    transcript: str | None = None
    hgvs_c: str | None = None
    hgvs_p: str | None = None
    consequence: str | None = None
    all_consequence_terms: list[str] = Field(default_factory=list)
    is_splice_relevant: bool = False
    mane_status: str | None = None
    protein_position: str | None = None
    exon: str | None = None
    intron: str | None = None
    impact: str | None = None
    clinvar_significance: str | None = None
    clinvar_id: str | None = None
    clinvar_review_status: str | None = None
    clinvar_condition: str | None = None
    clinvar_allele_id: str | None = None
    clinvar_source: str | None = None
    gnomad_af: float | None = None
    gnomad_af_popmax: float | None = None
    gnomad_af_by_population: dict[str, float] = Field(default_factory=dict)
    gnomad_allele_count: int | None = None
    gnomad_allele_number: int | None = None
    gnomad_af_popmax_allele_number: int | None = None
    gnomad_homozygote_count: int | None = None
    gnomad_filter_status: str | None = None
    gnomad_coverage: float | None = None
    gnomad_frequency_status: str = "not_evaluated"
    # The exact frequency the candidate filter used (see config.VARIANT_FILTER_PROFILE).
    filter_af: float | None = None
    filter_af_source: str | None = None
    filter_af_status: str | None = None
    filter_profile: str | None = None

    sift_prediction: str | None = None
    sift_score: float | None = None
    polyphen_prediction: str | None = None
    polyphen_score: float | None = None
    revel_score: float | None = None
    spliceai_score: float | None = None

    domains: list[str] = Field(default_factory=list)

    conservation_score: float | None = None

    dbsnp_id: str | None = None
    acmg_result: ACMGResult | None = None
    gwas_evidence: list[GWASEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    genotype: str | None = None
    zygosity: str | None = None
    allele_depth_ref: int | None = None
    allele_depth_alt: int | None = None
    read_depth: int | None = None
    quality: float | None = None
    filter_status: str | None = None
    was_multiallelic: bool = False
    was_indel_normalized: bool = False
    ref_matches_reference: bool | None = None
