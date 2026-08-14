from __future__ import annotations

import gzip
import logging
from pathlib import Path

from generesolver.config import DATA_DIR

logger = logging.getLogger(__name__)

_MANE_DIR = DATA_DIR / "mane"

_STATUS_MAP = {
    "MANE Select": "MANE_Select",
    "MANE Plus Clinical": "MANE_Plus_Clinical",
}

_mane_map: dict[str, str] | None = None


def _mane_file() -> Path | None:
    if not _MANE_DIR.is_dir():
        return None
    candidates = sorted(_MANE_DIR.glob("MANE.*.summary.txt.gz"))
    return candidates[-1] if candidates else None


def is_available() -> bool:
    return _mane_file() is not None


def _strip_version(transcript_id: str) -> str:
    return transcript_id.split(".", 1)[0]


def _load() -> dict[str, str]:
    path = _mane_file()
    if path is None:
        logger.warning(
            "Local MANE summary file not found under %s. Run scripts/download_mane.py "
            "to fetch it. PVS1's biological-relevance gate (mane_status) will rely "
            "solely on VEP's own MANE annotation, which is empty for GRCh37 cases.",
            _MANE_DIR,
        )
        return {}

    result: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        required = ("Ensembl_nuc", "MANE_status")
        missing = [r for r in required if r not in col]
        if missing:
            logger.warning(
                "MANE summary file %s is missing expected column(s) %s -- format may "
                "have changed. mane_local will return no results.", path.name, missing,
            )
            return {}
        idx_nuc, idx_status = col["Ensembl_nuc"], col["MANE_status"]

        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) <= max(idx_nuc, idx_status):
                continue
            raw_status = cols[idx_status]
            status = _STATUS_MAP.get(raw_status)
            if status is None:
                continue
            transcript = _strip_version(cols[idx_nuc])
            if not transcript:
                continue
            result.setdefault(transcript, status)

    logger.info("MANE local lookup loaded: %d transcript(s) from %s", len(result), path.name)
    return result


def get_mane_status(transcript_id: str | None) -> str | None:
    global _mane_map
    if not transcript_id:
        return None
    if _mane_map is None:
        _mane_map = _load()
    return _mane_map.get(_strip_version(transcript_id))
