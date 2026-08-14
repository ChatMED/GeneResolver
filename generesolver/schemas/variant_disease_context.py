from __future__ import annotations

from pydantic import BaseModel, Field


class AllelicVariantEvidence(BaseModel):
    variant_id: str
    zygosity: str | None = None
    classification_support: str | None = None
    phase: str | None = None


class FamilyEvidence(BaseModel):
    segregation_observed: bool | None = None
    n_affected_segregating: int | None = None
    n_unaffected_nonsegregating: int | None = None
    de_novo_confirmed: bool | None = None
    de_novo_stated: bool | None = None
    consanguinity: bool | None = None
    source: str | None = None


class ClinicalEvidenceItem(BaseModel):
    evidence_type: str
    description: str
    source: str | None = None
    supports_pathogenic: bool | None = None
    strength: str | None = None


class AlternativeDiagnosis(BaseModel):
    disease_id: str | None = None
    disease_name: str | None = None
    gene: str | None = None
    reason: str | None = None


class VariantDiseaseContext(BaseModel):
    variant_id: str
    gene: str
    disease_id: str
    disease_name: str | None = None
    transcript: str
    genome_build: str
    ruleset_version: str

    inheritance: str | None = None
    disease_mechanism: str | None = None
    gene_disease_validity: str | None = None
    vcep_specification: str | None = None

    patient_hpo_terms: list[str] = Field(default_factory=list)
    matched_hpo_terms: list[str] = Field(default_factory=list)
    phenotype_match_score: float | None = None
    phenotype_specificity_supported: bool | None = None

    zygosity: str | None = None
    patient_sex: str | None = None
    affected_status: str | None = None
    parental_origin: str | None = None
    phase: str | None = None
    allele_fraction: float | None = None

    other_alleles: list[AllelicVariantEvidence] = Field(default_factory=list)

    family_evidence: FamilyEvidence | None = None
    functional_evidence: list[ClinicalEvidenceItem] = Field(default_factory=list)
    case_control_evidence: list[ClinicalEvidenceItem] = Field(default_factory=list)
    alternative_diagnoses: list[AlternativeDiagnosis] = Field(default_factory=list)

    source_versions: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
