from __future__ import annotations

import logging
from pathlib import Path

from generesolver.config import DATA_DIR

logger = logging.getLogger(__name__)

_HI_BED = DATA_DIR / "clingen" / "hi_check.bed"
_SUFFICIENT_EVIDENCE_SCORE = "3"

_hi_cache: dict[str, str] | None = None


def is_available() -> bool:
    return _HI_BED.exists()


def _load() -> dict[str, str]:
    global _hi_cache
    if _hi_cache is not None:
        return _hi_cache
    if not _HI_BED.exists():
        logger.warning(
            "Local ClinGen haploinsufficiency BED not found at %s -- "
            "disease_mechanism fallback unavailable; PVS1 stays NOT_EVALUATED "
            "wherever Orphanet's own association_type doesn't specify a "
            "mechanism.", _HI_BED,
        )
        _hi_cache = {}
        return _hi_cache

    result: dict[str, str] = {}
    with _HI_BED.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("track") or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            gene, score = parts[3], parts[4]
            result[gene.upper()] = score
    logger.info("Local ClinGen haploinsufficiency track loaded: %d gene(s) from %s",
                len(result), _HI_BED.name)
    _hi_cache = result
    return _hi_cache


def get_lof_mechanism(gene: str) -> str | None:
    if not gene:
        return None
    score = _load().get(gene.upper())
    return "loss_of_function" if score == _SUFFICIENT_EVIDENCE_SCORE else None
