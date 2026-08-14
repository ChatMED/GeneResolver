from __future__ import annotations

import logging
import sqlite3
import zipfile
from pathlib import Path

from generesolver.config import DATA_DIR

logger = logging.getLogger(__name__)

_REVEL_ZIP_PATH = DATA_DIR / "revel" / "revel-v1.3_all_chromosomes.zip"
_REVEL_MEMBER_NAME = "revel_with_transcript_ids"
_INDEX_PATH = DATA_DIR / "revel" / "revel_grch37.sqlite3"

_BUILD_BATCH_SIZE = 200_000

_conn: sqlite3.Connection | None = None


def is_available() -> bool:
    return _REVEL_ZIP_PATH.exists()


def _build_index() -> None:
    logger.info("Building REVEL GRCh37 SQLite index at %s (one-time, several minutes) ...", _INDEX_PATH)
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _INDEX_PATH.with_suffix(".building.sqlite3")
    tmp_path.unlink(missing_ok=True)
    conn = sqlite3.connect(str(tmp_path))
    conn.execute("CREATE TABLE revel (chrom TEXT, pos INTEGER, ref TEXT, alt TEXT, score REAL)")

    batch: list[tuple[str, int, str, str, float]] = []
    n = 0
    with zipfile.ZipFile(_REVEL_ZIP_PATH) as z, z.open(_REVEL_MEMBER_NAME) as raw:
        header = raw.readline().decode("utf-8").rstrip("\n").split(",")
        col = {name: i for i, name in enumerate(header)}
        required = ("chr", "hg19_pos", "ref", "alt", "REVEL")
        missing = [r for r in required if r not in col]
        if missing:
            conn.close()
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"REVEL file is missing expected column(s) {missing} -- format may have changed.")
        idx_chr, idx_pos, idx_ref, idx_alt, idx_score = (col[r] for r in required)

        for raw_line in raw:
            cols = raw_line.decode("utf-8", "replace").rstrip("\n").split(",")
            if len(cols) <= idx_score:
                continue
            pos_str, score_str = cols[idx_pos], cols[idx_score]
            if pos_str == "." or score_str in (".", ""):
                continue
            try:
                batch.append((cols[idx_chr], int(pos_str), cols[idx_ref], cols[idx_alt], float(score_str)))
            except ValueError:
                continue
            n += 1
            if len(batch) >= _BUILD_BATCH_SIZE:
                conn.executemany("INSERT INTO revel VALUES (?, ?, ?, ?, ?)", batch)
                conn.commit()
                batch.clear()
                if n % (_BUILD_BATCH_SIZE * 10) == 0:
                    logger.info("  ... %d REVEL row(s) indexed so far", n)
        if batch:
            conn.executemany("INSERT INTO revel VALUES (?, ?, ?, ?, ?)", batch)
            conn.commit()

    logger.info("REVEL index: %d row(s) loaded, building lookup index ...", n)
    conn.execute("CREATE INDEX idx_revel_pos ON revel (chrom, pos, ref, alt)")
    conn.commit()
    conn.close()
    tmp_path.replace(_INDEX_PATH)
    logger.info("REVEL GRCh37 SQLite index ready at %s (%d rows)", _INDEX_PATH, n)


def _connection() -> sqlite3.Connection | None:
    global _conn
    if _conn is not None:
        return _conn
    if not _INDEX_PATH.exists():
        if not is_available():
            return None
        _build_index()
    _conn = sqlite3.connect(str(_INDEX_PATH), check_same_thread=False)
    return _conn


def get_revel_score(chrom: str, pos: int, ref: str, alt: str) -> float | None:
    conn = _connection()
    if conn is None:
        return None
    row = conn.execute(
        "SELECT score FROM revel WHERE chrom=? AND pos=? AND ref=? AND alt=? LIMIT 1",
        (str(chrom).lstrip("chr"), pos, ref, alt),
    ).fetchone()
    return row[0] if row else None
