from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

from generesolver.config import DATA_DIR

logger = logging.getLogger(__name__)

_VALIDITY_PATH = DATA_DIR / "clingen" / "gene_validity.csv"

_CONDITION_STOPWORDS = {"disease", "disorder", "syndrome", "deficiency", "the", "of", "type", "and"}


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower())
            if w not in _CONDITION_STOPWORDS and len(w) > 2}


_index: dict[str, list[dict]] | None = None


def is_available() -> bool:
    return _VALIDITY_PATH.exists()


def _load() -> dict[str, list[dict]]:
    if not _VALIDITY_PATH.exists():
        logger.warning(
            "ClinGen gene-validity file not found at %s. Run "
            "scripts/download_clingen_gene_validity.py to fetch. "
            "gene_disease_validity will stay unpopulated.",
            _VALIDITY_PATH,
        )
        return {}

    result: dict[str, list[dict]] = {}
    n = 0
    with _VALIDITY_PATH.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 7 or not row[0] or row[0].startswith("+") or row[0] == "GENE SYMBOL":
                continue
            gene, _hgnc, disease_label, _mondo, moi, _sop, classification = row[:7]
            if not gene or not classification:
                continue
            result.setdefault(gene, []).append({
                "disease_label": disease_label, "moi": moi, "classification": classification,
            })
            n += 1

    logger.info("ClinGen gene-disease validity loaded: %d entr(y/ies) across %d gene(s)",
                n, len(result))
    return result


def get_validity(gene: str, disease_name: str | None) -> dict | None:
    global _index
    if _index is None:
        _index = _load()
    entries = _index.get(gene)
    if not entries:
        return None
    if len(entries) == 1:
        return entries[0]
    if not disease_name:
        return None
    disease_tokens = _tokens(disease_name)
    if not disease_tokens:
        return None
    for entry in entries:
        entry_tokens = _tokens(entry["disease_label"])
        if entry_tokens and len(disease_tokens & entry_tokens) >= min(2, len(disease_tokens)):
            return entry
    return None
