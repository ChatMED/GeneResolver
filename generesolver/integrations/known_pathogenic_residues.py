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

_AA3 = {
    "Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His", "Ile",
    "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val",
}

_PROTEIN_HGVS_RE = re.compile(r"p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})")


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


def _build_index() -> dict[str, dict[int, list[tuple[str, str]]]]:
    index: dict[str, dict[int, list[tuple[str, str]]]] = {}

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
                for line in f:
                    cols = line.rstrip("\n").split("\t")
                    if len(cols) <= max_idx or cols[idx_assembly] != "GRCh37":
                        continue
                    parsed = _parse_clinvar_row(cols[idx_name], cols[idx_sig])
                    if parsed is None:
                        continue
                    pos, ref, alt = parsed
                    gene = cols[idx_gene]
                    positions = index.setdefault(gene, {})
                    pairs = positions.setdefault(pos, [])
                    if (ref, alt) not in pairs:
                        pairs.append((ref, alt))
                    n += 1
        logger.info("ClinVar pathogenic-missense index: %d entr(y/ies) across %d gene(s)", n, len(index))

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

    return index


def _load() -> dict[str, dict[str, list[list[str]]]]:
    if _CACHE_PATH.exists():
        with gzip.open(_CACHE_PATH, "rt", encoding="utf-8") as f:
            return json.load(f)
    index = _build_index()
    serializable = {
        gene: {str(pos): [[r, a] for r, a in pairs] for pos, pairs in positions.items()}
        for gene, positions in index.items()
    }
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(_CACHE_PATH, "wt", encoding="utf-8") as f:
        json.dump(serializable, f)
    return serializable


_index: dict[str, dict[str, list[list[str]]]] | None = None


def is_available() -> bool:
    return _CLINVAR_SUMMARY_PATH.exists() or _UNIPROT_HUMSAVAR_PATH.exists()


def known_pathogenic_at_residue(gene: str, position: int) -> list[tuple[str, str]]:
    global _index
    if _index is None:
        _index = _load()
    return [(p[0], p[1]) for p in _index.get(gene, {}).get(str(position), [])]


def pathogenic_residues_near(gene: str, position: int, window: int) -> int:
    global _index
    if _index is None:
        _index = _load()
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
