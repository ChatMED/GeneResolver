from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CriterionStatus(str, Enum):
    MET = "met"
    NOT_MET = "not_met"
    NOT_APPLICABLE = "not_applicable"
    NOT_EVALUATED = "not_evaluated"
    CONFLICTING = "conflicting"


class EvidenceStrength(str, Enum):
    SUPPORTING = "supporting"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"
    STAND_ALONE = "stand_alone"


class EvidenceDirection(str, Enum):
    PATHOGENIC = "pathogenic"
    BENIGN = "benign"


class EvidenceItem(BaseModel):
    evidence_id: str
    evidence_type: str
    source: str
    value: dict = Field(default_factory=dict)
    provenance: dict = Field(default_factory=dict)


class CriterionResult(BaseModel):
    code: str
    status: CriterionStatus
    direction: EvidenceDirection
    strength: EvidenceStrength | None = None
    points: int | None = None

    rationale: str
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    evaluator_version: str
    source_versions: dict[str, str] = Field(default_factory=dict)


class Conflict(BaseModel):
    criteria: list[str]
    reason: str


class CombinationResult(BaseModel):
    classification: str
    total_points: int | None
    basis: str
    conflicts: list[Conflict] = Field(default_factory=list)
    conflict_status: str | None = None
    counted_criteria: list[str] = Field(default_factory=list)
    excluded_criteria: list[str] = Field(default_factory=list)


class ACMGAssessment(BaseModel):
    interpretation_id: str
    variant_id: str
    gene: str
    disease_id: str
    transcript: str
    inheritance: str | None = None
    disease_mechanism: str | None = None

    classification: str
    total_points: int | None
    classification_status: str
    criteria: list[CriterionResult] = Field(default_factory=list)

    pathogenic_criteria: list[str] = Field(default_factory=list)
    benign_criteria: list[str] = Field(default_factory=list)
    criteria_not_met: list[str] = Field(default_factory=list)
    criteria_not_applicable: list[str] = Field(default_factory=list)
    criteria_not_evaluated: list[str] = Field(default_factory=list)
    criteria_conflicting: list[str] = Field(default_factory=list)
    counted_criteria: list[str] = Field(default_factory=list)
    excluded_dependent_criteria: list[str] = Field(default_factory=list)

    combination_model: str = "Tavtigian_2020"
    ruleset: dict = Field(default_factory=dict)
    resource_versions: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    requires_expert_review: bool = True


class ClinVarExternalAnnotation(BaseModel):
    significance: str | None = None
    review_status: str | None = None
    condition_name: str | None = None
    variation_id: str | None = None
    snapshot: str | None = None


class ACMGComparisonResult(BaseModel):
    computed_acmg_assessment: ACMGAssessment
    clinvar_external_annotation: ClinVarExternalAnnotation | None = None
    concordance: str
    concordance_basis: str | None = None
