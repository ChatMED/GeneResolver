from __future__ import annotations

import gzip
import json
import logging
import re
from pathlib import Path

from generesolver.config import DATA_DIR

logger = logging.getLogger(__name__)

_CLINVAR_SUMMARY_PATH = DATA_DIR / "clinvar" / "variant_summary.txt.gz"
_UNIPROT_HUMSAVAR_PATH = DATA_DIR / "uniprot" / "humsavar.txt"
_CACHE_PATH = DATA_DIR / "clinvar" / "pathogenic_protein_positions_cache.json.gz"
_TYPE_CACHE_PATH = DATA_DIR / "clinvar" / "pathogenic_variant_type_fraction_cache.json.gz"

_AA3 = {
    "Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His", "Ile",
    "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val",
}

_PROTEIN_HGVS_RE = re.compile(r"p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})")

# BP1's real premise is a CURATED fact ("this gene's known pathogenic variants are
# primarily truncating"), not a population-constraint statistic. We approximate the
# curated fact by classifying each gene's own ClinVar pathogenic/likely-pathogenic
# variants as truncating vs. missense vs. unclassifiable, from the same HGVS "Name"
# field already scanned for the missense-position index below (one file pass, not two).
_NONSENSE_RE = re.compile(r"p\.[A-Za-z]{3}\d+(Ter|\*)")
_FRAMESHIFT_RE = re.compile(r"p\.[A-Za-z]{3}\d+[A-Za-z]{3}fs")
# Canonical splice donor/acceptor SNV, e.g. c.123+1G>A or c.456-2A>G (VEP/ClinGen PVS1
# treats +1/+2/-1/-2 as canonical splice; anything further into the intron is not).
_CANONICAL_SPLICE_RE = re.compile(r"c\.\d+[+-][12][ACGT]>[ACGT]")

BP1_MIN_CLASSIFIED = 5  # disclosed minimum sample size before trusting a fraction
BP1_TRUNCATING_FRACTION_THRESHOLD = 0.9  # disclosed cutoff, not a guideline-derived number


def _clinvar_is_pathogenic(significance: str) -> bool:
    sig = significance.lower()
    if "conflicting" in sig or "uncertain" in sig:
        return False
    return "pathogenic" in sig and "benign" not in sig


def _parse_clinvar_row(name: str, significance: str) -> tuple[int, str, str] | None:
    if not _clinvar_is_pathogenic(significance):
        return None
    m = _PROTEIN_HGVS_RE.search(name)
    if not m:
        return None
    ref, pos, alt = m.group(1), int(m.group(2)), m.group(3)
    if ref not in _AA3 or alt not in _AA3:
        return None
    return pos, ref, alt


def _classify_consequence(name: str) -> str | None:
    """Classify a pathogenic/likely-pathogenic ClinVar row's HGVS Name as 'truncating'
    (nonsense, frameshift, or canonical +/-1,2 splice-site SNV), 'missense', or None
    (not classifiable from this field -- e.g. large deletions, in-frame indels, deep
    intronic variants). Order matters: check truncating patterns before the broad
    missense regex, since a frameshift/nonsense HGVS can superficially resemble it."""
    if _NONSENSE_RE.search(name) or _FRAMESHIFT_RE.search(name) or _CANONICAL_SPLICE_RE.search(name):
        return "truncating"
    m = _PROTEIN_HGVS_RE.search(name)
    if m and m.group(1) in _AA3 and m.group(3) in _AA3:
        return "missense"
    return None


def _build_index() -> tuple[dict[str, dict[int, list[tuple[str, str]]]], dict[str, dict[str, int]]]:
    index: dict[str, dict[int, list[tuple[str, str]]]] = {}
    type_counts: dict[str, dict[str, int]] = {}

    if not _CLINVAR_SUMMARY_PATH.exists():
        logger.warning(
            "ClinVar variant_summary.txt.gz not found at %s. Run "
            "scripts/download_clinvar_variant_summary.py to fetch. "
            "PS1/PM5 will only have UniProt humsavar coverage, if available.",
            _CLINVAR_SUMMARY_PATH,
        )
    else:
        n = 0
        with gzip.open(_CLINVAR_SUMMARY_PATH, "rt", encoding="utf-8", errors="replace") as f:
            header = f.readline().lstrip("#").rstrip("\n").split("\t")
            col = {name: i for i, name in enumerate(header)}
            required = ("Assembly", "GeneSymbol", "Name", "ClinicalSignificance")
            if any(r not in col for r in required):
                logger.error("variant_summary.txt.gz is missing expected column(s) %s -- skipping ClinVar source.",
                             [r for r in required if r not in col])
            else:
                idx_assembly, idx_gene, idx_name, idx_sig = (col[r] for r in required)
                max_idx = max(idx_assembly, idx_gene, idx_name, idx_sig)
                n_type = 0
                for line in f:
                    cols = line.rstrip("\n").split("\t")
                    if len(cols) <= max_idx or cols[idx_assembly] != "GRCh37":
                        continue
                    if not _clinvar_is_pathogenic(cols[idx_sig]):
                        continue
                    name = cols[idx_name]
                    gene = cols[idx_gene]

                    parsed = _parse_clinvar_row(name, cols[idx_sig])
                    if parsed is not None:
                        pos, ref, alt = parsed
                        positions = index.setdefault(gene, {})
                        pairs = positions.setdefault(pos, [])
                        if (ref, alt) not in pairs:
                            pairs.append((ref, alt))
                        n += 1

                    consequence = _classify_consequence(name)
                    if consequence is not None:
                        counts = type_counts.setdefault(gene, {"truncating": 0, "missense": 0})
                        counts[consequence] += 1
                        n_type += 1
        logger.info("ClinVar pathogenic-missense index: %d entr(y/ies) across %d gene(s)", n, len(index))
        logger.info("ClinVar truncating/missense classification: %d entr(y/ies) across %d gene(s)",
                    n_type, len(type_counts))

    if _UNIPROT_HUMSAVAR_PATH.exists():
        n2 = 0
        with _UNIPROT_HUMSAVAR_PATH.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5 or parts[4] != "LP/P":
                    continue
                gene, aa_change = parts[0], parts[3]
                m = re.match(r"^p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})$", aa_change)
                if not m:
                    continue
                ref, pos, alt = m.group(1), int(m.group(2)), m.group(3)
                if ref not in _AA3 or alt not in _AA3:
                    continue
                positions = index.setdefault(gene, {})
                pairs = positions.setdefault(pos, [])
                if (ref, alt) not in pairs:
                    pairs.append((ref, alt))
                n2 += 1
        logger.info("UniProt humsavar pathogenic-missense index: %d additional entr(y/ies)", n2)
    else:
        logger.warning(
            "UniProt humsavar.txt not found at %s. Run "
            "scripts/download_uniprot_humsavar.py to fetch.",
            _UNIPROT_HUMSAVAR_PATH,
        )

    return index, type_counts


def _load() -> tuple[dict[str, dict[str, list[list[str]]]], dict[str, dict[str, int]]]:
    if _CACHE_PATH.exists() and _TYPE_CACHE_PATH.exists():
        with gzip.open(_CACHE_PATH, "rt", encoding="utf-8") as f:
            positions = json.load(f)
        with gzip.open(_TYPE_CACHE_PATH, "rt", encoding="utf-8") as f:
            types = json.load(f)
        return positions, types

    index, type_counts = _build_index()
    serializable = {
        gene: {str(pos): [[r, a] for r, a in pairs] for pos, pairs in positions.items()}
        for gene, positions in index.items()
    }
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(_CACHE_PATH, "wt", encoding="utf-8") as f:
        json.dump(serializable, f)
    _TYPE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(_TYPE_CACHE_PATH, "wt", encoding="utf-8") as f:
        json.dump(type_counts, f)
    return serializable, type_counts


_index: dict[str, dict[str, list[list[str]]]] | None = None
_type_counts: dict[str, dict[str, int]] | None = None


def _ensure_loaded() -> None:
    global _index, _type_counts
    if _index is None or _type_counts is None:
        _index, _type_counts = _load()


def is_available() -> bool:
    return _CLINVAR_SUMMARY_PATH.exists() or _UNIPROT_HUMSAVAR_PATH.exists()


def truncating_fraction(gene: str) -> tuple[float | None, int]:
    """Fraction of `gene`'s ClinVar pathogenic/likely-pathogenic variants classified as
    truncating (nonsense/frameshift/canonical splice) vs. missense, and the number of
    variants that classification is based on. Returns (None, 0) if the gene has fewer
    than BP1_MIN_CLASSIFIED classified variants -- a disclosed minimum sample size, not
    a guideline requirement, chosen to avoid a spurious 100% from 1-2 variants.
    Variants that are neither cleanly truncating nor missense (large deletions,
    in-frame indels, deep intronic) are not classified and are excluded from both the
    numerator and denominator, not counted as evidence either way."""
    _ensure_loaded()
    counts = _type_counts.get(gene)
    if not counts:
        return None, 0
    total = counts.get("truncating", 0) + counts.get("missense", 0)
    if total < BP1_MIN_CLASSIFIED:
        return None, total
    return counts.get("truncating", 0) / total, total


def known_pathogenic_at_residue(gene: str, position: int) -> list[tuple[str, str]]:
    _ensure_loaded()
    return [(p[0], p[1]) for p in _index.get(gene, {}).get(str(position), [])]


def pathogenic_residues_near(gene: str, position: int, window: int) -> int:
    _ensure_loaded()
    gene_positions = _index.get(gene, {})
    if not gene_positions:
        return 0
    count = 0
    for pos_str in gene_positions:
        try:
            pos = int(pos_str)
        except ValueError:
            continue
        if pos != position and abs(pos - position) <= window:
            count += 1
    return count
