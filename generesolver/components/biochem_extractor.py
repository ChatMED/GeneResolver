from __future__ import annotations

import logging

from pydantic import BaseModel

from generesolver.integrations.llm_client import complete_structured, active_backend
from generesolver.schemas.biochem import BiochemMeasurement, Direction
from generesolver.schemas.case import CaseState
from generesolver.schemas.report import ProvenanceItem

logger = logging.getLogger(__name__)



_BIOCHEM_LLM_SYSTEM = (
    "Extract biochemical/laboratory test results from clinical text. Return ONLY what is "
    "explicitly stated — never infer or guess.\n"
    "For each genuine lab result, extract:\n"
    "- analyte: the name of the substance/test measured (e.g., 'glucose', 'potassium', 'ammonia')\n"
    "- value: the numeric value if stated\n"
    "- unit: the unit of measurement (e.g., 'mg/dL', 'mL/min', '%')\n"
    "- direction: 'increased', 'decreased', or 'normal' based on context or reference range\n"
    "- ref_low / ref_high: lower and upper bounds of reference range if stated\n"
    "- specimen: sample type (plasma, serum, urine, CSF, etc.) if specified\n"
    "Return a JSON object: {\"measurements\": [{\"analyte\": str, \"value\": float|null, "
    "\"unit\": str|null, \"direction\": \"increased\"|\"decreased\"|\"normal\"|\"unknown\", "
    "\"ref_low\": float|null, \"ref_high\": float|null, \"specimen\": str|null}, ...]}\n"
    'Use {"measurements": []} if no values are stated anywhere.'
)


class _BiochemMeasurementExtraction(BaseModel):
    analyte: str
    value: float | None = None
    unit: str | None = None
    direction: str = "unknown"
    ref_low: float | None = None
    ref_high: float | None = None
    specimen: str | None = None


class _BiochemExtractionResult(BaseModel):
    measurements: list[_BiochemMeasurementExtraction] = []


def _llm_extract(note: str, backend: str | None = None) -> list[BiochemMeasurement] | None:
    try:
        try:
            result = complete_structured(
                system=_BIOCHEM_LLM_SYSTEM, user=note,
                schema_model=_BiochemExtractionResult, max_tokens=1000, backend=backend,
            )
        except Exception as exc:
            logger.warning("biochem_extractor: LLM extraction failed schema, retrying once: %s", exc)
            try:
                result = complete_structured(
                    system=_BIOCHEM_LLM_SYSTEM,
                    user=note + '\n\n(CORRECTION: your previous answer used the wrong JSON schema. '
                                'Return ONLY {"measurements": [...]} -- a JSON OBJECT with a '
                                '"measurements" key, not a bare array -- using {"measurements": []} '
                                'if none are found.)',
                    schema_model=_BiochemExtractionResult, max_tokens=1000, backend=backend,
                )
            except Exception as exc2:
                logger.warning("biochem_extractor: LLM extraction retry failed: %s", exc2)
                raise
        
        measurements: list[BiochemMeasurement] = []
        direction_map = {
            "increased": Direction.INCREASED,
            "decreased": Direction.DECREASED,
            "normal": Direction.NORMAL,
            "unknown": Direction.UNKNOWN,
        }
        
        for m in result.measurements:
            direction = direction_map.get((m.direction or "").strip().lower(), Direction.UNKNOWN)
            measurements.append(BiochemMeasurement(
                analyte=m.analyte,
                value=m.value,
                unit=m.unit,
                direction=direction,
                direction_source="llm",
                ref_low=m.ref_low,
                ref_high=m.ref_high,
                specimen=m.specimen,
                verbatim=f"{m.analyte}={m.value} {m.unit or ''}".strip(),
            ))
        return measurements
    except Exception as exc:
        logger.warning("biochem_extractor: LLM extraction failed: %s", exc)
        return None


def extract(note: str, backend: str | None = None) -> list[BiochemMeasurement]:
    result = _llm_extract(note, backend)
    if result is None:
        raise RuntimeError("Biochemical extraction model failed")
    return result


def run(state: CaseState, backend: str | None = None) -> CaseState:
    note = state.clinical_note_deidentified if state.deidentification_applied else ""
    if not note:
        state.mark_done("biochem_extractor")
        return state

    try:
        measurements = extract(note, backend=backend)
    except Exception as exc:
        logger.warning("Biochemical extraction failed: %s", exc)
        state.biochem_panel.extraction_failed = True
        state.add_warning(f"Biochemical extraction failed: {exc}")
        state.mark_done("biochem_extractor")
        return state

    state.biochem_panel.measurements = measurements
    abnormal = state.biochem_panel.abnormal()

    if measurements:
        logger.info(
            "Biochemical extractor: %d measurement(s), %d abnormal.",
            len(measurements), len(abnormal),
        )

    state.trace.add_step(
        step="biochem_extractor",
        claim=(
            f"{len(measurements)} measurement(s) parsed, {len(abnormal)} abnormal: "
            + "; ".join(m.summary() for m in abnormal[:5])
        ) if measurements else "No biochemical measurements found in note",
        grounded_in=["clinical_note_deidentified"],
    )
    state.provenance.append(ProvenanceItem(
        component="biochem_extractor",
        tool_name="llm biochemical measurement extractor",
        tool_version=f"backend={backend or active_backend()}",
    ))
    state.mark_done("biochem_extractor")
    return state
