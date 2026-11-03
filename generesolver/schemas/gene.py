from enum import Enum
from pydantic import BaseModel, Field


class InheritanceModel(str, Enum):
    AUTOSOMAL_DOMINANT = "autosomal_dominant"
    AUTOSOMAL_RECESSIVE = "autosomal_recessive"
    X_LINKED = "x_linked"
    X_LINKED_RECESSIVE = "x_linked_recessive"
    X_LINKED_DOMINANT = "x_linked_dominant"
    MITOCHONDRIAL = "mitochondrial"
    DE_NOVO = "de_novo"
    UNKNOWN = "unknown"


class InheritanceCompatibility(BaseModel):
    gene: str
    suspected_model: InheritanceModel
    compatible: bool | None = None
    limitations: list[str] = Field(default_factory=list)


class GeneRank(BaseModel):
    rank: int
    gene: str
    hgnc_id: str | None = None
    final_score: float = Field(ge=0.0, le=1.0)
    confidence: str

    score_phenotype: float = 0.0
    score_variant: float = 0.0
    score_gene_disease: float = 0.0
    score_literature: float = 0.0
    score_gwas: float = 0.0
    score_disease_match: float = 0.0
    score_phenotype_normalized: float = 0.0
    score_constraint: float = 0.0
    score_pathway: float = 0.0
    constraint_weight_gated: bool = False
    score_adjustment: float = 0.0

    best_variant_id: str | None = None
    seeded_by_diseases: list[str] = Field(default_factory=list)
    rerank_adjustment: float = 0.0
    rerank_reasoning: str = ""
    rerank_grounded_in: list[str] = Field(default_factory=list)
    inheritance: InheritanceCompatibility | None = None
    known_disease_associations: list[str] = Field(default_factory=list)
    prioritization_reason: str = ""
    warnings: list[str] = Field(default_factory=list)

    acmg_rerank_interpretation_id: str | None = None
    requires_expert_review: bool = True
