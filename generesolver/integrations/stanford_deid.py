from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MODEL = "StanfordAIMI/stanford-deidentifier-base"

_pipe = None
_load_failed = False

_PLACEHOLDERS = {
    "PATIENT": "[PATIENT]",
    "HCW": "[CLINICIAN]",
    "HOSPITAL": "[HOSPITAL]",
    "DATE": "[DATE]",
    "ID": "[ID]",
    "PHONE": "[PHONE]",
    "AGE": "[AGE]",
    "LOCATION": "[LOCATION]",
    "VENDOR": "[VENDOR]",
}


def _load():
    global _pipe, _load_failed
    if _pipe is None and not _load_failed:
        try:
            from transformers import pipeline
            _pipe = pipeline(
                "token-classification",
                model=_MODEL,
            )
            logger.info("Loaded Stanford de-identifier: %s", _MODEL)
        except Exception as exc:
            _load_failed = True
            logger.warning("Stanford de-identifier unavailable: %s", exc)
    return _pipe


def _placeholder(label: str) -> str:
    return _PLACEHOLDERS.get(label.upper().removeprefix("B-").removeprefix("I-"), "[PHI]")


def deidentify_text(text: str) -> tuple[str, bool]:
    if not text:
        return text, True

    pipe = _load()
    if pipe is None:
        return text, False

    try:
        spans = pipe(text)
    except Exception as exc:
        logger.warning("Stanford de-identification failed: %s", exc)
        return text, False

    if not spans:
        return text, True
    out = text
    for span in sorted(spans, key=lambda s: s["start"], reverse=True):
        out = out[:span["start"]] + _placeholder(span["entity_group"]) + out[span["end"]:]

    return out, True


def is_available() -> bool:
    return _load() is not None
