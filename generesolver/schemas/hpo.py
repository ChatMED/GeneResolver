from enum import Enum
from pydantic import BaseModel, Field


class HPOStatus(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNCERTAIN = "uncertain"


class HPOTerm(BaseModel):
    mention: str
    hpo_id: str
    label: str
    status: HPOStatus
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_span: str


class HPOProfile(BaseModel):
    terms: list[HPOTerm] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def present_ids(self) -> list[str]:
        return [t.hpo_id for t in self.terms if t.status == HPOStatus.PRESENT]

    @property
    def absent_ids(self) -> list[str]:
        return [t.hpo_id for t in self.terms if t.status == HPOStatus.ABSENT]
