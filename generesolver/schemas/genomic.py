from enum import Enum
from pydantic import BaseModel, Field


class GenomeBuild(str, Enum):
    GRCH37 = "GRCh37"
    GRCH38 = "GRCh38"
    UNKNOWN = "unknown"


class QCStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    UNKNOWN = "unknown"


class VCFContext(BaseModel):
    file_path: str | None = None
    file_type: str | None = None
    sample_id: str | None = None
    genome_build: GenomeBuild = GenomeBuild.UNKNOWN
    variant_count: int | None = None
    pass_variant_count: int | None = None
    qc_status: QCStatus = QCStatus.UNKNOWN
    pipeline_version: str | None = None
    warnings: list[str] = Field(default_factory=list)
