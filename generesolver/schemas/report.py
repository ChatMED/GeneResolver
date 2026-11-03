from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field


class ConfidenceBand(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNCERTAIN = "uncertain"


class AlternativeHypothesis(BaseModel):
    gene: str
    orpha_id: str | None = None
    disease_label: str = ""
    score_in_ranker: float = 0.0
    brief_reasoning: str = ""


class AppliedRerank(BaseModel):
    gene: str
    adjustment: float
    reasoning: str = ""
    grounded_in: list[str] = Field(default_factory=list)


class DifferentialDecision(BaseModel):
    most_likely_gene: str = ""
    most_likely_orpha: str | None = None
    most_likely_disease_label: str = ""
    confidence_band: ConfidenceBand = ConfidenceBand.UNCERTAIN
    primary_reasoning: str = ""
    alternatives: list[AlternativeHypothesis] = Field(default_factory=list)
    decisive_tests: list[str] = Field(default_factory=list)
    flag_human_review: bool = True
    review_reasons: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    grounded_in: list[str] = Field(default_factory=list)
    rerank_applied: list[AppliedRerank] = Field(default_factory=list)


class ProvenanceItem(BaseModel):
    component: str
    tool_name: str
    tool_version: str | None = None
    database_version: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    input_hash: str | None = None
    output_hash: str | None = None
    details: dict = Field(default_factory=dict)


class DoctorReport(BaseModel):
    case_id: str
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    case_summary: str = ""
    classification: str = ""
    classification_reasoning: str = ""
    classification_scores: dict[str, float] = Field(default_factory=dict)
    possible_conditions: list[dict] = Field(default_factory=list)

    top_candidate_genes: list[str] = Field(default_factory=list)
    top_candidate_variants: list[str] = Field(default_factory=list)
    evidence_strength: str = ""
    mechanism_explanation: str = ""
    uncertainty_statement: str = ""
    suggested_review_points: list[str] = Field(default_factory=list)
    suggested_next_steps: list[dict] = Field(default_factory=list)
    next_steps_reasoning: str = ""
    reasoning_trace: list[dict] = Field(default_factory=list)
    retrieval_trace: list[dict] = Field(default_factory=list)
    sources_consulted: list[str] = Field(default_factory=list)
    investigation_limitations: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "This is a preliminary decision-support output. "
        "It must be reviewed by a qualified clinician or clinical geneticist "
        "before any diagnostic or clinical action is taken."
    )


class GeneticistAppendix(BaseModel):
    case_id: str
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hpo_terms_used: list[str] = Field(default_factory=list)
    vcf_context_summary: str = ""
    candidate_variants: list[dict] = Field(default_factory=list)
    acmg_evidence: list[dict] = Field(default_factory=list)
    inheritance_assessment: list[dict] = Field(default_factory=list)
    gwas_evidence: list[dict] = Field(default_factory=list)
    gene_disease_evidence: list[dict] = Field(default_factory=list)
    literature_evidence: list[dict] = Field(default_factory=list)
    mechanism_evidence: list[dict] = Field(default_factory=list)
    tool_versions: list[ProvenanceItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
