from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Direction(str, Enum):
    INCREASED = "increased"
    DECREASED = "decreased"
    NORMAL = "normal"
    UNKNOWN = "unknown"


class BiochemMeasurement(BaseModel):
    analyte: str
    value: float | None = None
    unit: str | None = None
    ref_low: float | None = None
    ref_high: float | None = None
    direction: Direction = Direction.UNKNOWN
    direction_source: str = "unknown"
    specimen: str | None = None
    verbatim: str = ""
    hpo_id: str | None = None

    @property
    def is_abnormal(self) -> bool:
        return self.direction in (Direction.INCREASED, Direction.DECREASED)

    def summary(self) -> str:
        bits = [self.analyte]
        if self.value is not None:
            bits.append(f"{self.value}{f' {self.unit}' if self.unit else ''}")
        if self.direction != Direction.UNKNOWN:
            bits.append(f"({self.direction.value})")
        if self.specimen:
            bits.append(f"[{self.specimen}]")
        return " ".join(bits)


class BiochemPanel(BaseModel):
    measurements: list[BiochemMeasurement] = Field(default_factory=list)
    extraction_failed: bool = False

    def abnormal(self) -> list[BiochemMeasurement]:
        return [m for m in self.measurements if m.is_abnormal]
