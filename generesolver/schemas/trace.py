from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class TestPower(str, Enum):
    DECISIVE = "decisive"
    SUPPORTIVE = "supportive"
    EXCLUSIONARY = "exclusionary"


class ConfirmatoryTest(BaseModel):
    test: str
    rationale: str = ""
    confirms: list[str] = Field(default_factory=list)
    refutes: list[str] = Field(default_factory=list)
    power: TestPower = TestPower.SUPPORTIVE
    source: str = ""
    loinc_code: str | None = None


class NextSteps(BaseModel):
    tests: list[ConfirmatoryTest] = Field(default_factory=list)
    reasoning: str = ""

    def decisive(self) -> list[ConfirmatoryTest]:
        return [t for t in self.tests if t.power == TestPower.DECISIVE]


class RetrievalRecord(BaseModel):
    source: str
    query: str = ""
    n_results: int = 0
    results: list[str] = Field(default_factory=list)
    took_ms: int | None = None
    failed: bool = False
    error: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ReasoningStep(BaseModel):
    step: str
    claim: str = ""
    grounded_in: list[str] = Field(default_factory=list)
    confidence: str = ""


class ReasoningTrace(BaseModel):
    retrievals: list[RetrievalRecord] = Field(default_factory=list)
    steps: list[ReasoningStep] = Field(default_factory=list)

    def add_retrieval(self, **kw) -> RetrievalRecord:
        rec = RetrievalRecord(**kw)
        self.retrievals.append(rec)
        return rec

    def add_step(self, **kw) -> ReasoningStep:
        rec = ReasoningStep(**kw)
        self.steps.append(rec)
        return rec

    def sources_used(self) -> list[str]:
        seen: list[str] = []
        for r in self.retrievals:
            if not r.failed and r.source not in seen:
                seen.append(r.source)
        return seen
