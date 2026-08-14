from __future__ import annotations

import gzip
import logging
import subprocess
import tempfile
from pathlib import Path

from generesolver.config import DATA_DIR

logger = logging.getLogger(__name__)

_GNOMAD_POPMAX_ZIP = DATA_DIR / "gnomad_popmax" / "gnomad.hg37.zip"
_SPLICEAI_ZIP = DATA_DIR / "spliceai" / "spliceai.hg37.zip"
_DOCKER_IMAGE = "brentp/slivar"

_AN_INDEX = DATA_DIR / "gnomad_popmax" / "gnomad_an_popmax.hg37.tsv.gz"
_an_cache: dict[str, dict[str, object]] | None = None

_NOT_FOUND = -1.0


def is_available(genome_build: str) -> bool:
    return genome_build == "GRCh37" and _GNOMAD_POPMAX_ZIP.exists() and _SPLICEAI_ZIP.exists()


def an_index_available() -> bool:
    return _AN_INDEX.exists()


def _load_an_index() -> dict[str, dict[str, object]]:
    global _an_cache
    if _an_cache is not None:
        return _an_cache
    if not _AN_INDEX.exists():
        logger.warning(
            "Local gnomAD popmax allele-number index not found at %s. Run "
            "scripts/download_gnomad_exomes_sites.py then "
            "scripts/build_gnomad_an_index.py to build it -- until then, "
            "BA1/PM2 population-coverage confirmation stays unavailable (see "
            "acmg_engine.py's disclosed coverage-unconfirmed caveat).",
            _AN_INDEX,
        )
        _an_cache = {}
        return _an_cache

    result: dict[str, dict[str, object]] = {}
    with gzip.open(_AN_INDEX, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            chrom, pos, ref, alt, af_raw, an_raw = parts[:6]
            popmax_pop = parts[6] if len(parts) > 6 else ""
            try:
                af = float(af_raw)
                an = int(an_raw)
            except ValueError:
                continue
            key = f"{chrom.lstrip('chr')}:{pos}:{ref}:{alt}"
            result[key] = {
                "gnomad_popmax_af": af,
                "gnomad_popmax_an": an,
                "gnomad_popmax_population": popmax_pop or None,
            }
    logger.info("Local gnomAD popmax allele-number index loaded: %d record(s) from %s",
                len(result), _AN_INDEX.name)
    _an_cache = result
    return _an_cache


def get_allele_number(chrom: str, pos: int, ref: str, alt: str) -> int | None:
    entry = _load_an_index().get(f"{str(chrom).lstrip('chr')}:{pos}:{ref}:{alt}")
    return entry["gnomad_popmax_an"] if entry else None


def _vcf_record_line(chrom: str, pos: int, ref: str, alt: str) -> str:
    chrom_clean = str(chrom).lstrip("chr")
    return f"{chrom_clean}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\n"


def _parse_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    return None if val == _NOT_FOUND else val


def _run_slivar(variants: list[tuple[str, int, str, str]]) -> dict[str, dict]:
    work = Path(tempfile.mkdtemp(prefix="generesolver_slivar_"))
    in_vcf = work / "in.vcf"
    out_vcf = work / "out.vcf"
    rows = sorted({(str(c).lstrip("chr"), int(p), r, a) for c, p, r, a in variants},
                  key=lambda x: (x[0], x[1]))
    with in_vcf.open("w", encoding="utf-8") as f:
        f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for c, p, r, a in rows:
            f.write(_vcf_record_line(c, p, r, a))

    cache_dir = DATA_DIR.resolve()
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{cache_dir}:/data", "-v", f"{work}:/work",
        _DOCKER_IMAGE, "slivar", "expr",
        "--vcf", "/work/in.vcf",
        "--gnotate", "/data/gnomad_popmax/gnomad.hg37.zip",
        "--gnotate", "/data/spliceai/spliceai.hg37.zip",
        "--info", "true",
        "--out-vcf", "/work/out.vcf",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True,
                        env={"MSYS_NO_PATHCONV": "1"} | dict(__import__("os").environ))
    except subprocess.CalledProcessError as exc:
        logger.error("Local slivar gnomAD/SpliceAI annotation failed: %s",
                     (exc.stderr or b"").decode("utf-8", "ignore")[:600])
        return {}

    return _parse_slivar_vcf(out_vcf)


def annotate_batch(
    variants: list[tuple[str, int, str, str]], genome_build: str,
) -> dict[str, dict]:
    if not variants:
        return {}

    results: dict[str, dict] = {}
    if is_available(genome_build):
        results = _run_slivar(variants)

    if genome_build == "GRCh37" and an_index_available():
        an_idx = _load_an_index()
        keys = {f"{str(c).lstrip('chr')}:{p}:{r}:{a}" for c, p, r, a in variants}
        for key in keys:
            an_entry = an_idx.get(key)
            if an_entry is None:
                continue
            row = results.setdefault(key, {})
            row["gnomad_popmax_an"] = an_entry["gnomad_popmax_an"]
            row["gnomad_popmax_population"] = an_entry["gnomad_popmax_population"]
            row.setdefault("gnomad_popmax_af", an_entry["gnomad_popmax_af"])

    return results


def _parse_slivar_vcf(out_vcf: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    try:
        with out_vcf.open(encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 8:
                    continue
                chrom, pos, _, ref, alt, _, _, info = cols[:8]
                info_map: dict[str, str] = {}
                for kv in info.split(";"):
                    if "=" in kv:
                        k, _, v = kv.partition("=")
                        info_map[k] = v
                key = f"{chrom.lstrip('chr')}:{pos}:{ref}:{alt}"
                results[key] = {
                    "gnomad_popmax_af": _parse_float(info_map.get("gnomad_popmax_af")),
                    "gnomad_popmax_af_filtered": "gnomad_popmax_af_filter" in info_map,
                    "gnomad_nhomalt": _parse_float(info_map.get("gnomad_nhomalt")),
                    "spliceai_score": _parse_float(info_map.get("spliceai")),
                }
    except FileNotFoundError:
        logger.error("slivar produced no output VCF: %s", out_vcf)
    return results
