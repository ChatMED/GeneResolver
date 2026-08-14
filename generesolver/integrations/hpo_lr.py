from __future__ import annotations

import logging
from functools import lru_cache

from generesolver.config import ROOT

logger = logging.getLogger(__name__)

_HPOA_PATH = ROOT / "data" / "hpo_a" / "phenotype.hpoa"
_P2G_PATH = ROOT / "data" / "hpo_a" / "phenotype_to_genes.txt"
_FREQ_CLASS = {
    "HP:0040280": 1.00,
    "HP:0040281": 0.90,
    "HP:0040282": 0.55,
    "HP:0040283": 0.17,
    "HP:0040284": 0.025,
    "HP:0040285": 0.00,
}

_DEFAULT_FREQUENCY = 1.0


def _parse_frequency(raw: str) -> float:
    s = (raw or "").strip()
    if not s:
        return _DEFAULT_FREQUENCY
    if s in _FREQ_CLASS:
        return _FREQ_CLASS[s]
    if s.endswith("%"):
        try:
            return max(0.0, min(1.0, float(s[:-1]) / 100.0))
        except ValueError:
            return _DEFAULT_FREQUENCY
    if "/" in s:
        num, _, den = s.partition("/")
        try:
            n, d = float(num), float(den)
            if d > 0:
                return max(0.0, min(1.0, n / d))
        except ValueError:
            pass
        return _DEFAULT_FREQUENCY
    return _DEFAULT_FREQUENCY


def _hpo_to_int(hpo_id: str) -> int | None:
    try:
        return int(hpo_id.split(":")[1])
    except (ValueError, IndexError):
        return None


@lru_cache(maxsize=1)
def _load_hpoa() -> tuple[
    dict[str, dict[int, float]],
    dict[str, set[int]],
    dict[str, str],
    dict[str, dict[int, list[str]]],
]:
    present: dict[str, dict[int, float]] = {}
    excluded: dict[str, set[int]] = {}
    names: dict[str, str] = {}
    references: dict[str, dict[int, list[str]]] = {}

    if not _HPOA_PATH.exists():
        logger.warning(
            "hpo_lr: %s missing — LR ranker cannot run. "
            "Download via the HPO release (purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa).",
            _HPOA_PATH,
        )
        return present, excluded, names, references

    with _HPOA_PATH.open(encoding="utf-8") as fh:
        header_cols: list[str] | None = None
        idx: dict[str, int] = {}
        for line in fh:
            if line.startswith("#"):
                continue
            row = line.rstrip("\n").split("\t")
            if header_cols is None:
                header_cols = row
                idx = {c: i for i, c in enumerate(header_cols)}
                continue
            try:
                disease_id = row[idx["database_id"]].strip()
                hpo_id = row[idx["hpo_id"]].strip()
                aspect = row[idx["aspect"]].strip()
                qualifier = row[idx["qualifier"]].strip()
                freq_raw = row[idx["frequency"]] if idx["frequency"] < len(row) else ""
                name = row[idx["disease_name"]].strip() if idx.get("disease_name", 99) < len(row) else ""
                ref_raw = row[idx["reference"]].strip() if idx.get("reference", 99) < len(row) else ""
            except (KeyError, IndexError):
                continue
            if aspect != "P":
                continue
            term_int = _hpo_to_int(hpo_id)
            if term_int is None or not disease_id:
                continue
            if name and disease_id not in names:
                names[disease_id] = name
            if qualifier == "NOT":
                excluded.setdefault(disease_id, set()).add(term_int)
                continue
            freq = _parse_frequency(freq_raw)
            d = present.setdefault(disease_id, {})
            if freq > d.get(term_int, 0.0):
                d[term_int] = freq
            if ref_raw:
                refs = [r for r in ref_raw.split(";") if r and r.startswith(("PMID", "OMIM", "ORPHA"))]
                if refs:
                    references.setdefault(disease_id, {}).setdefault(term_int, [])
                    for r in refs:
                        if r not in references[disease_id][term_int]:
                            references[disease_id][term_int].append(r)

    logger.info(
        "hpo_lr: loaded %d diseases (%d with excluded terms) from phenotype.hpoa",
        len(present), len(excluded),
    )
    return present, excluded, names, references


def disease_hpo_frequencies() -> dict[str, dict[int, float]]:
    return _load_hpoa()[0]


def disease_excluded_terms() -> dict[str, set[int]]:
    return _load_hpoa()[1]


def disease_names() -> dict[str, str]:
    return _load_hpoa()[2]


def annotation_references() -> dict[str, dict[int, list[str]]]:
    return _load_hpoa()[3]


@lru_cache(maxsize=1)
def n_diseases() -> int:
    return len(disease_hpo_frequencies())


@lru_cache(maxsize=1)
def background_frequency() -> dict[int, float]:
    present = disease_hpo_frequencies()
    n = max(1, len(present))
    totals: dict[int, float] = {}
    for term_map in present.values():
        for term_int, freq in term_map.items():
            totals[term_int] = totals.get(term_int, 0.0) + freq
    floor = 1.0 / (2.0 * n)
    return {t: max(floor, s / n) for t, s in totals.items()}


@lru_cache(maxsize=1)
def _load_disease_gene_maps() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    d2g: dict[str, set[str]] = {}
    g2d: dict[str, set[str]] = {}
    if not _P2G_PATH.exists():
        logger.warning("hpo_lr: %s missing — disease↔gene bridge unavailable", _P2G_PATH)
        return d2g, g2d
    with _P2G_PATH.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        gi = idx.get("gene_symbol")
        di = idx.get("disease_id")
        if gi is None or di is None:
            logger.warning("hpo_lr: phenotype_to_genes.txt missing expected columns")
            return d2g, g2d
        for line in fh:
            row = line.rstrip("\n").split("\t")
            if gi >= len(row) or di >= len(row):
                continue
            gene = row[gi].strip()
            disease = row[di].strip()
            if not gene or not disease:
                continue
            d2g.setdefault(disease, set()).add(gene)
            g2d.setdefault(gene, set()).add(disease)
    return d2g, g2d


def genes_for_disease(disease_id: str) -> set[str]:
    return _load_disease_gene_maps()[0].get(disease_id, set())


def diseases_for_gene(gene_symbol: str) -> set[str]:
    return _load_disease_gene_maps()[1].get(gene_symbol, set())


def is_available() -> bool:
    return _HPOA_PATH.exists() and bool(disease_hpo_frequencies())
