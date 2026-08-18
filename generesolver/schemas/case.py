from enum import Enum
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from .clinical import ClinicalEntity, DiseaseMention
from .hpo import HPOProfile
from .genomic import VCFContext
from .variant import AnnotatedVariant, ACMGResult, GWASEvidence
from .variant_disease_context import VariantDiseaseContext
from .acmg_assessment import ACMGComparisonResult
from .gene import GeneRank, InheritanceCompatibility
from .evidence import LiteratureEvidence, BiologicalEvidence
from .report import (
    DoctorReport,
    GeneticistAppendix,
    ProvenanceItem,
    DifferentialDecision,
)
from .biochem import BiochemPanel
from .classification import CaseClassification
from .trace import NextSteps, ReasoningTrace


class TaskMode(str, Enum):
    PHENOTYPE_ONLY = "phenotype_only"
    PHENOTYPE_VCF = "phenotype_vcf_prioritization"
    VARIANT_INTERPRETATION = "variant_interpretation"
    GENE_DISEASE_QUERY = "gene_disease_query"
    LOCUS_LOOKUP = "locus_lookup"


class CaseInput(BaseModel):
    case_id: str
    clinical_note: str | None = None
    vcf_path: str | None = None
    dragen_dir: str | None = None
    candidate_gene: str | None = None
    candidate_variant: str | None = None
    rsid: str | None = None
    disease_label: str | None = None
    family_history: str | None = None
    hpo_terms: list[str] | None = None
    sex: str | None = None
    consanguinity: bool | None = None
    age_years: float | None = None


class CaseState(BaseModel):
    case_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    task_mode: TaskMode | None = None
    input: CaseInput
    clinical_note_deidentified: str | None = None
    deidentification_applied: bool = False
    has_clinical_note: bool = False
    has_vcf: bool = False
    has_candidate_variant: bool = False
    has_rsid: bool = False
    has_family_history: bool = False
    completed_steps: list[str] = Field(default_factory=list)
    pending_steps: list[str] = Field(default_factory=list)
    stop_after_step: str | None = None
    clinical_entities: list[ClinicalEntity] = Field(default_factory=list)
    disease_mentions: list[DiseaseMention] = Field(default_factory=list)
    hpo_profile: HPOProfile = Field(default_factory=HPOProfile)
    vcf_context: VCFContext | None = None
    annotated_variants: list[AnnotatedVariant] = Field(default_factory=list)
    acmg_results: list[ACMGResult] = Field(default_factory=list)
    gwas_evidence: list[GWASEvidence] = Field(default_factory=list)
    inheritance_results: list[InheritanceCompatibility] = Field(default_factory=list)
    ranked_genes: list[GeneRank] = Field(default_factory=list)
    ranked_diseases: list = Field(default_factory=list)
    variant_disease_contexts: list[VariantDiseaseContext] = Field(default_factory=list)
    acmg_assessments: list[ACMGComparisonResult] = Field(default_factory=list)
    literature_evidence: list[LiteratureEvidence] = Field(default_factory=list)
    biological_evidence: list[BiologicalEvidence] = Field(default_factory=list)

    biochem_panel: BiochemPanel = Field(default_factory=BiochemPanel)
    age_years: float | None = None
    age_months: float | None = None
    sex: str | None = None
    family_history: str | None = None
    consanguinity: bool | None = None
    de_novo_stated: bool | None = None
    de_novo_confirmed: bool | None = None
    segregation_observed: bool | None = None
    n_affected_segregating: int | None = None
    n_unaffected_nonsegregating: int | None = None
    unaffected_carrier_observed: bool | None = None
    unaffected_carrier_adult: bool | None = None
    case_class: CaseClassification | None = None
    branch_result: dict = Field(default_factory=dict)
    safety_flags: list[str] = Field(default_factory=list)
    requires_human_review: bool = False
    human_review_reasons: list[str] = Field(default_factory=list)
    confidence_level: str = "unknown"

    differential_decision: DifferentialDecision | None = None
    next_steps: NextSteps = Field(default_factory=NextSteps)
    trace: ReasoningTrace = Field(default_factory=ReasoningTrace)

    doctor_report: DoctorReport | None = None
    geneticist_appendix: GeneticistAppendix | None = None

    provenance: list[ProvenanceItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    diagnostics: dict = Field(default_factory=dict)

    def mark_done(self, step: str) -> None:
        if step not in self.completed_steps:
            self.completed_steps.append(step)
        if step in self.pending_steps:
            self.pending_steps.remove(step)

    def add_warning(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    def add_safety_flag(self, flag: str) -> None:
        if flag not in self.safety_flags:
            self.safety_flags.append(flag)
            self.requires_human_review = True
