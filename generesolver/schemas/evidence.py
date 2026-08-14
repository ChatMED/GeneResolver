from pydantic import BaseModel, Field


class LiteratureEvidence(BaseModel):
    gene: str
    query_terms: list[str] = Field(default_factory=list)
    pmids: list[str] = Field(default_factory=list)
    support_level: str = "unknown"
    summary: str = ""
    warnings: list[str] = Field(default_factory=list)


class BiologicalEvidence(BaseModel):
    gene: str
    known_function: str = ""
    pathways: list[str] = Field(default_factory=list)
    disease_associations: list[str] = Field(default_factory=list)
    uniprot_id: str | None = None
    biological_support: str = "unknown"
    support_score: float = 0.0
    mechanism_summary_doctor: str = ""
    mechanism_summary_expert: str = ""
    evidence_sources: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
