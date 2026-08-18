from __future__ import annotations

import csv
import re
from pathlib import Path

_DISEASE_TOKEN = re.compile(
    r"\b(OMIM|ORPHA|ORPHANET|MONDO|CCRD|DECIPHER|OMIMPS):\s*([A-Za-z0-9]+)\b",
    re.IGNORECASE,
)
_HPO_TOKEN = re.compile(r"\bHP:?\s*(\d{1,7})\b", re.IGNORECASE)


def normalize_disease_id(raw: str) -> str | None:
    if not raw:
        return None
    m = _DISEASE_TOKEN.search(str(raw))
    if not m:
        return None
    prefix, num = m.group(1).upper(), m.group(2)
    if prefix in ("ORPHANET", "ORPHA"):
        prefix = "ORPHA"
    if prefix == "OMIMPS":
        prefix = "OMIM"
    return f"{prefix}:{num}"


def normalize_hpo_id(raw: str) -> str | None:
    if not raw:
        return None
    m = _HPO_TOKEN.search(str(raw))
    if not m:
        return None
    return f"HP:{int(m.group(1)):07d}"


def extract_ids(value, kind: str = "disease") -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    text = " ".join(str(x) for x in items)
    if kind == "hpo":
        return [f"HP:{int(n):07d}" for n in _HPO_TOKEN.findall(text)]
    out, seen = [], set()
    for m in _DISEASE_TOKEN.finditer(text):
        nid = normalize_disease_id(m.group(0))
        if nid and nid not in seen:
            seen.add(nid)
            out.append(nid)
    return out


def normalize_gene(symbol: str, alias_map: dict[str, str] | None = None) -> str | None:
    if not symbol:
        return None
    s = str(symbol).strip().upper()
    if alias_map and s in alias_map:
        return alias_map[s]
    return s or None



def load_crosswalk(path: Path | None) -> dict[str, set[str]]:
    cross: dict[str, set[str]] = {}
    if not path:
        return cross
    path = Path(path)
    if not path.exists():
        return cross

    def link(a: str | None, b: str | None) -> None:
        if not a or not b or a == b:
            return
        cross.setdefault(a, set()).add(b)
        cross.setdefault(b, set()).add(a)

    with path.open(encoding="utf-8") as fh:
        data_lines = [ln for ln in fh if ln.strip() and not ln.startswith("#")]
        if not data_lines:
            return cross
        delim = "\t" if "\t" in data_lines[0] else ","
        reader = csv.reader(data_lines, delimiter=delim)
        header = next(reader, None)
        cols = {c.strip().lower(): i for i, c in enumerate(header or [])}
        si = cols.get("subject_id", 0)
        oi = cols.get("object_id", 1)
        for row in reader:
            if len(row) <= max(si, oi):
                continue
            link(normalize_disease_id(row[si]), normalize_disease_id(row[oi]))
    return cross


def load_label_map(path: Path | None) -> dict[str, str]:
    labels: dict[str, str] = {}
    if not path:
        return labels
    path = Path(path)
    if not path.exists():
        return labels
    with path.open(encoding="utf-8") as fh:
        data_lines = [ln for ln in fh if ln.strip() and not ln.startswith("#")]
        if not data_lines:
            return labels
        delim = "\t" if "\t" in data_lines[0] else ","
        reader = csv.reader(data_lines, delimiter=delim)
        header = next(reader, None)
        cols = {c.strip().lower(): i for i, c in enumerate(header or [])}
        pairs = [(cols.get("subject_id"), cols.get("subject_label")),
                 (cols.get("object_id"), cols.get("object_label"))]
        for row in reader:
            for id_i, lab_i in pairs:
                if id_i is None or lab_i is None:
                    continue
                if len(row) <= max(id_i, lab_i):
                    continue
                did = normalize_disease_id(row[id_i])
                lab = (row[lab_i] or "").strip()
                if did and lab and did not in labels:
                    labels[did] = lab
    return labels


def resolve_names(ids: list[str], label_map: dict[str, str]) -> list[str]:
    out, seen = [], set()
    for did in ids:
        name = label_map.get(did, did)
        key = name.lower()
        if key not in seen:
            seen.add(key)
            out.append(name)
    return out


def expand_gold(gold_ids: list[str], cross: dict[str, set[str]]) -> set[str]:
    out: set[str] = set()
    for g in gold_ids:
        out.add(g)
        out |= cross.get(g, set())
    return out


def load_gene_alias(path: Path | None) -> dict[str, str]:
    amap: dict[str, str] = {}
    if not path:
        return amap
    path = Path(path)
    if not path.exists():
        return amap
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = re.split(r"[\t,]", line.strip())
            if len(parts) >= 2:
                amap[parts[0].strip().upper()] = parts[1].strip().upper()
    return amap
