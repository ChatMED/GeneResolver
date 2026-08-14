from __future__ import annotations

import csv
import re
from functools import lru_cache

from generesolver.config import ROOT

_PATH = ROOT / "data" / "mondo.sssom.tsv"
_PREF = ("ORPHA", "OMIM", "MONDO")


def _norm_label(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


def _norm_id(raw: str) -> str | None:
    m = re.match(r"\s*(OMIM|ORPHA|ORPHANET|MONDO|OMIMPS)\s*:\s*([A-Za-z0-9]+)", raw or "", re.I)
    if not m:
        return None
    prefix = m.group(1).upper()
    if prefix == "ORPHANET":
        prefix = "ORPHA"
    if prefix == "OMIMPS":
        prefix = "OMIM"
    return f"{prefix}:{m.group(2)}"


@lru_cache(maxsize=1)
def _index() -> dict[str, set[str]]:
    idx: dict[str, set[str]] = {}
    if not _PATH.exists():
        return idx
    with _PATH.open(encoding="utf-8") as fh:
        data = [ln for ln in fh if ln.strip() and not ln.startswith("#")]
    if not data:
        return idx
    delim = "\t" if "\t" in data[0] else ","
    reader = csv.reader(data, delimiter=delim)
    header = next(reader, None)
    cols = {c.strip().lower(): i for i, c in enumerate(header or [])}
    pairs = [(cols.get("subject_id"), cols.get("subject_label")),
             (cols.get("object_id"), cols.get("object_label"))]
    for row in reader:
        for id_i, lab_i in pairs:
            if id_i is None or lab_i is None or len(row) <= max(id_i, lab_i):
                continue
            did = _norm_id(row[id_i])
            lab = _norm_label(row[lab_i])
            if did and lab:
                idx.setdefault(lab, set()).add(did)
    return idx


def resolve(name: str) -> tuple[str, str] | None:
    ids = _index().get(_norm_label(name))
    if not ids:
        return None
    for pref in _PREF:
        for did in ids:
            if did.startswith(pref):
                return did, name
    return next(iter(sorted(ids))), name


def is_available() -> bool:
    return _PATH.exists()
