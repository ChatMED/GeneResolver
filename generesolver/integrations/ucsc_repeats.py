from __future__ import annotations

import bisect
import gzip
import logging
from pathlib import Path

from generesolver.config import DATA_DIR

logger = logging.getLogger(__name__)

_REPEATS_PATH = DATA_DIR / "ucsc" / "simpleRepeat_hg19.txt.gz"

_CHROM_COL, _START_COL, _END_COL = 1, 2, 3

_intervals: dict[str, tuple[list[int], list[int]]] | None = None


def is_available() -> bool:
    return _REPEATS_PATH.exists()


def _load() -> dict[str, tuple[list[int], list[int]]]:
    if not _REPEATS_PATH.exists():
        logger.warning(
            "UCSC simpleRepeat track not found at %s. Run "
            "scripts/download_ucsc_simple_repeat.py to fetch. Repeat-region "
            "lookups will return None (not checked) for every variant.",
            _REPEATS_PATH,
        )
        return {}

    by_chrom: dict[str, list[tuple[int, int]]] = {}
    n = 0
    with gzip.open(_REPEATS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) <= _END_COL:
                continue
            chrom = cols[_CHROM_COL].lstrip("chr")
            try:
                start, end = int(cols[_START_COL]), int(cols[_END_COL])
            except ValueError:
                continue
            by_chrom.setdefault(chrom, []).append((start, end))
            n += 1

    result: dict[str, tuple[list[int], list[int]]] = {}
    for chrom, pairs in by_chrom.items():
        pairs.sort()
        result[chrom] = ([p[0] for p in pairs], [p[1] for p in pairs])

    logger.info("UCSC simpleRepeat loaded: %d interval(s) across %d chromosome(s) from %s",
                n, len(result), _REPEATS_PATH.name)
    return result


def in_repeat_region(chrom: str, pos: int) -> bool | None:
    global _intervals
    if _intervals is None:
        _intervals = _load()
    if not _intervals:
        return None
    starts, ends = _intervals.get(str(chrom).lstrip("chr"), ([], []))
    if not starts:
        return False
    idx = bisect.bisect_right(starts, pos) - 1
    lo = max(0, idx - 2)
    for i in range(lo, min(len(starts), idx + 3)):
        if starts[i] < pos <= ends[i]:
            return True
    return False
