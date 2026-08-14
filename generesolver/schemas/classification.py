from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CaseClass(str, Enum):
    SINGLE_GENE = "single_gene"
    CHROMOSOMAL = "chromosomal"
    MULTIFACTORIAL = "multifactorial"
    PHENOCOPY = "phenocopy"


class ClassificationMode(str, Enum):
    DOMINANT = "dominant"
    OVERLAP = "overlap"
    UNCERTAIN = "uncertain"


class ClassAdvocate(BaseModel):
    case_class: CaseClass
    score: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    deductions: list[str] = Field(default_factory=list)
    reasoning: str = ""


class CaseClassification(BaseModel):
    decision: CaseClass | None = None
    mode: ClassificationMode = ClassificationMode.UNCERTAIN
    advocates: list[ClassAdvocate] = Field(default_factory=list)
    margin: float = 0.0
    tiebreak_reasoning: str = ""
    conclusion: str = ""
    inputs_seen: list[str] = Field(default_factory=list)
    fallback_used: bool = False

    @property
    def is_single_gene(self) -> bool:
        return self.decision == CaseClass.SINGLE_GENE

    def top_score(self) -> float:
        return max((a.score for a in self.advocates), default=0.0)
