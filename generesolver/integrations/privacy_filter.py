from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MODEL = "OpenMed/privacy-filter-multilingual"
_deidentify_fn = None


def _load() -> object | None:
    global _deidentify_fn
    if _deidentify_fn is None:
        try:
            from openmed import deidentify
            _deidentify_fn = deidentify
        except Exception as exc:
            logger.warning("openmed privacy filter unavailable: %s", exc)
    return _deidentify_fn


def deidentify_text(text: str) -> tuple[str, bool]:
    if not text:
        return text, True

    fn = _load()
    if fn is None:
        return text, False

    try:
        result = fn(text, method="mask", model_name=_MODEL)
        deidentified = getattr(result, "deidentified_text", result)
        return str(deidentified), True
    except Exception as exc:
        logger.warning("Privacy filter failed: %s — original text retained", exc)
        return text, False


def is_available() -> bool:
    return _load() is not None
