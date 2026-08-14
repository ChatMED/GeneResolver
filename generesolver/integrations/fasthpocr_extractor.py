from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from generesolver.config import DATA_DIR, OUTPUTS_DIR

logger = logging.getLogger(__name__)

_HPO_OBO_FILE  = DATA_DIR / "hp.obo"
_INDEX_DIR     = OUTPUTS_DIR / "_fasthpocr_index"
_CR_INDEX_FILE = _INDEX_DIR / "hp.index"

_annotator: Any | None = None
_unavailable_reason: str | None = None


def _ensure_index() -> Path | None:
    _INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if _CR_INDEX_FILE.exists():
        return _CR_INDEX_FILE

    if not _HPO_OBO_FILE.exists():
        logger.error(
            "FastHPOCR needs hp.obo at %s. Download it from "
            "https://github.com/obophenotype/human-phenotype-ontology/"
            "releases and place there. (Tested release: v2026-02-16.)",
            _HPO_OBO_FILE,
        )
        return None

    try:
        from FastHPOCR.IndexHPO import IndexHPO
        logger.info("Building FastHPOCR cr-index from %s (one-time, ~1-3 min)...",
                    _HPO_OBO_FILE)
        idx = IndexHPO(str(_HPO_OBO_FILE), str(_INDEX_DIR))
        idx.index()
        if not _CR_INDEX_FILE.exists():
            logger.error("FastHPOCR index build completed but %s not present",
                         _CR_INDEX_FILE)
            return None
        return _CR_INDEX_FILE
    except Exception as exc:
        logger.error("FastHPOCR index build failed: %s", exc)
        return None


def _get_annotator():
    global _annotator, _unavailable_reason
    if _annotator is not None:
        return _annotator
    if _unavailable_reason is not None:
        return None
    try:
        from FastHPOCR.HPOAnnotator import HPOAnnotator
    except ImportError as exc:
        _unavailable_reason = f"FastHPOCR not installed: {exc}"
        logger.warning(_unavailable_reason)
        return None

    index_path = _ensure_index()
    if index_path is None:
        _unavailable_reason = "FastHPOCR index unavailable"
        return None

    try:
        _annotator = HPOAnnotator(str(index_path))
    except Exception as exc:
        _unavailable_reason = f"FastHPOCR HPOAnnotator init failed: {exc}"
        logger.error(_unavailable_reason)
        return None
    return _annotator


def is_available() -> bool:
    return _get_annotator() is not None


def _hpo_uri_to_id(uri: str) -> str:
    if not uri:
        return ""
    tail = uri.rsplit("/", 1)[-1]
    return tail.replace("_", ":") if tail.startswith("HP") else uri


def extract_hpo_terms(text: str) -> list[dict]:
    if not text or not text.strip():
        return []
    ann = _get_annotator()
    if ann is None:
        return []

    try:
        raw = ann.annotate(text)
    except Exception as exc:
        logger.warning("FastHPOCR annotate raised: %s", exc)
        return []

    out: list[dict] = []
    for a in raw:
        hpo_id = _hpo_uri_to_id(a.getHPOUri())
        if not hpo_id:
            continue
        out.append({
            "hpo_id":       hpo_id,
            "label":        a.getHPOLabel(),
            "text_span":    a.getTextSpan(),
            "start_offset": a.getStartOffset(),
            "end_offset":   a.getEndOffset(),
        })
    return out
