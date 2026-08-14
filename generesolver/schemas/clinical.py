from enum import Enum
from pydantic import BaseModel, Field


class NegationStatus(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNCERTAIN = "uncertain"


class EntityType(str, Enum):
    DISEASE = "disease"
    GENE = "gene"
    PROTEIN = "protein"
    CELL_LINE = "cell_line"
    SYMPTOM = "symptom"
    VARIANT = "variant"
    INHERITANCE = "inheritance"
    FAMILY_HISTORY = "family_history"
    LAB_FINDING = "lab_finding"


class ClinicalEntity(BaseModel):
    text: str
    entity_type: EntityType
    negation_status: NegationStatus = NegationStatus.PRESENT
    confidence: float = Field(ge=0.0, le=1.0)
    start: int
    end: int
    source_model: str


class DiseaseStatus(str, Enum):
    PRESENT = "present"
    UNCERTAIN = "uncertain"
    ABSENT = "absent"


class DiseaseSource(str, Enum):
    NER = "ner"
    USER_LABEL = "user_label"
    LLM_SYNTHESIS = "llm_synthesis"


class DiseaseMention(BaseModel):
    text: str
    label: str = ""
    orpha_id: str | None = None
    omim_ids: list[str] = Field(default_factory=list)
    icd10_ids: list[str] = Field(default_factory=list)
    mondo_id: str | None = None
    status: DiseaseStatus = DiseaseStatus.PRESENT
    confidence: float = Field(ge=0.0, le=1.0)
    source: DiseaseSource = DiseaseSource.NER
    evidence_span: str = ""
    start: int = -1
    end: int = -1
