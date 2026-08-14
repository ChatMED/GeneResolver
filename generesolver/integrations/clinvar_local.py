from __future__ import annotations

import gzip
import logging
from pathlib import Path

from generesolver.config import DATA_DIR

logger = logging.getLogger(__name__)

_CLINVAR_DIR = DATA_DIR / "clinvar"
_PATHS: dict[str, Path] = {
    "GRCh37": _CLINVAR_DIR / "clinvar_GRCh37.vcf.gz",
    "GRCh38": _CLINVAR_DIR / "clinvar_GRCh38.vcf.gz",
}

_caches: dict[str, dict[str, dict[str, str | None]]] = {}


def _parse_info(info: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for kv in info.split(";"):
        if "=" in kv:
            k, _, v = kv.partition("=")
            out[k] = v
    return out


def _clean(val: str | None) -> str | None:
    if not val or val == ".":
        return None
    return val.replace("_", " ").replace("|", "; ")


def _load(genome_build: str) -> dict[str, dict[str, str | None]]:
    path = _PATHS.get(genome_build)
    if not path or not path.exists():
        logger.warning(
            "Local ClinVar VCF not found for %s at %s. Run "
            "scripts/download_clinvar.py to fetch. Local ClinVar lookups will "
            "return None for all variants on this build (VEP's embedded "
            "CLIN_SIG, if any, is still used as a fallback).",
            genome_build, path,
        )
        return {}

    result: dict[str, dict[str, str | None]] = {}
    n = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 8:
                continue
            chrom, pos, vcf_id, ref, alt = cols[0], cols[1], cols[2], cols[3], cols[4]
            info = _parse_info(cols[7])
            clnsig = info.get("CLNSIG")
            if not clnsig:
                continue
            key = f"{chrom.lstrip('chr')}:{pos}:{ref}:{alt}"
            result[key] = {
                "clinvar_significance": _clean(clnsig),
                "clinvar_review_status": _clean(info.get("CLNREVSTAT")),
                "clinvar_condition": _clean(info.get("CLNDN")),
                "clinvar_allele_id": info.get("ALLELEID") or None,
                "clinvar_variation_id": vcf_id if vcf_id and vcf_id != "." else None,
            }
            n += 1

    logger.info(
        "Local ClinVar loaded: %d record(s) with a significance call for %s from %s",
        n, genome_build, path.name,
    )
    return result


def is_available(genome_build: str) -> bool:
    path = _PATHS.get(genome_build)
    return bool(path and path.exists())


def get_variant(
    genome_build: str, chrom: str, pos: int | str, ref: str, alt: str,
) -> dict[str, str | None] | None:
    if genome_build not in _caches:
        _caches[genome_build] = _load(genome_build)
    cache = _caches[genome_build]
    if not cache:
        return None
    key = f"{str(chrom).lstrip('chr')}:{pos}:{ref}:{alt}"
    return cache.get(key)
