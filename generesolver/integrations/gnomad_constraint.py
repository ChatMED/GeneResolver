from __future__ import annotations

import csv
import logging
from pathlib import Path

from generesolver.config import DATA_DIR

logger = logging.getLogger(__name__)

_CONSTRAINT_PATH = DATA_DIR / "gnomad" / "gnomad.v4.1.constraint_metrics.tsv"
_constraint_map: dict[str, dict[str, float]] | None = None


def _parse_float(s: str) -> float | None:
    if s in ("", "NA", "NaN", "."):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _load() -> dict[str, dict[str, float]]:
    if not _CONSTRAINT_PATH.exists():
        logger.warning(
            "gnomAD constraint file not found at %s. Run "
            "scripts/download_gnomad_constraint.py to fetch. "
            "constraint_score() will return 0.0 for all genes.",
            _CONSTRAINT_PATH,
        )
        return {}

    result: dict[str, dict[str, float]] = {}
    with _CONSTRAINT_PATH.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sym = (row.get("gene") or row.get("gene_symbol") or "").strip()
            if not sym:
                continue
            if sym in result:
                continue
            mane_select = (row.get("mane_select") or "").strip().lower()
            canonical = (row.get("canonical") or "").strip().lower()
            if mane_select not in ("true", "1") and canonical not in ("true", "1"):
                pass

            entry: dict[str, float] = {}
            pli = _parse_float(row.get("lof.pLI", ""))
            loeuf = _parse_float(row.get("lof.oe_ci.upper", ""))
            mis_z = _parse_float(row.get("mis.z_score", ""))
            if pli is not None:
                entry["pli"] = pli
            if loeuf is not None:
                entry["loeuf"] = loeuf
            if mis_z is not None:
                entry["mis_z"] = mis_z
            if entry:
                result[sym] = entry

    logger.info(
        "gnomAD constraint loaded: %d genes from %s",
        len(result), _CONSTRAINT_PATH.name,
    )
    return result


def get_constraint(gene_symbol: str) -> dict[str, float] | None:
    global _constraint_map
    if _constraint_map is None:
        _constraint_map = _load()
    return _constraint_map.get(gene_symbol)


def constraint_score(gene_symbol: str) -> float:
    c = get_constraint(gene_symbol)
    if c is None:
        return 0.0
    if "pli" in c:
        return max(0.0, min(1.0, c["pli"]))
    if "loeuf" in c:
        return max(0.0, min(1.0, (2.0 - c["loeuf"]) / 2.0))
    return 0.0
