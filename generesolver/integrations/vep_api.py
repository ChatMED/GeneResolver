from __future__ import annotations

import gzip
import json
import os
import re
import subprocess
import tempfile
import time
import logging
from pathlib import Path
from typing import Any

import httpx

from generesolver import config
from generesolver.config import VEP_REST_URL

logger = logging.getLogger(__name__)

_VEP_BATCH_URL = "https://rest.ensembl.org/vep/human/region"
_BATCH_SIZE = 50
_RETRY_ATTEMPTS = 3
_RETRY_DELAY = 2.0
_REQUEST_INTERVAL = 0.07

_RESULT_CACHE_DIR = config.DATA_DIR / "vep_cache"
_RESULT_CACHE_PATH = _RESULT_CACHE_DIR / "vep_annotations.json.gz"
_result_cache: dict[str, dict] | None = None
_result_cache_dirty = False


def _variant_key(chrom: str, pos: int, ref: str, alt: str) -> str:
    return f"{str(chrom).lstrip('chr')}:{pos}:{ref}:{alt}"


def _load_result_cache() -> dict[str, dict]:
    global _result_cache
    if _result_cache is not None:
        return _result_cache
    if _RESULT_CACHE_PATH.exists():
        try:
            with gzip.open(_RESULT_CACHE_PATH, "rt", encoding="utf-8") as f:
                _result_cache = json.load(f)
            logger.info("Local VEP result cache loaded: %d annotation(s) from %s",
                        len(_result_cache), _RESULT_CACHE_PATH.name)
        except Exception as exc:
            logger.warning("Local VEP result cache unreadable (%s) -- starting fresh.", exc)
            _result_cache = {}
    else:
        _result_cache = {}
    return _result_cache


def _save_result_cache() -> None:
    global _result_cache_dirty
    if _result_cache is None or not _result_cache_dirty:
        return
    _RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = _RESULT_CACHE_PATH.with_suffix(".tmp")
    with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
        json.dump(_result_cache, f)
    tmp_path.replace(_RESULT_CACHE_PATH)
    _result_cache_dirty = False


def _vcf_record_to_region(chrom: str, pos: int, ref: str, alt: str) -> str:
    chrom_clean = chrom.lstrip("chr") if not chrom.startswith("chr") else chrom
    return f"{chrom_clean} {pos} . {ref} {alt} . . ."


def annotate_variants(
    variants: list[tuple[str, int, str, str]],
    canonical_only: bool = True,
) -> dict[str, dict]:
    if not variants:
        return {}

    cache = _load_result_cache()
    results: dict[str, dict] = {}
    misses: list[tuple[str, int, str, str]] = []
    for c, p, r, a in variants:
        key = _variant_key(c, p, r, a)
        cached = cache.get(key)
        if cached is not None:
            results[key] = cached
        else:
            misses.append((c, p, r, a))

    if misses:
        fresh = (
            annotate_variants_local(misses, canonical_only) if config.LOCAL_MODE
            else _annotate_variants_online(misses, canonical_only)
        )
        global _result_cache_dirty
        for key, ann in fresh.items():
            cache[key] = ann
            results[key] = ann
            _result_cache_dirty = True
        _save_result_cache()

    return results


def _annotate_variants_online(
    variants: list[tuple[str, int, str, str]],
    canonical_only: bool = True,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    chunks = [variants[i:i + _BATCH_SIZE] for i in range(0, len(variants), _BATCH_SIZE)]

    with httpx.Client(timeout=120.0) as client:
        for chunk in chunks:
            region_strings = [
                _vcf_record_to_region(c, p, r, a) for c, p, r, a in chunk
            ]
            payload: dict[str, Any] = {
                "variants": region_strings,
            }
            params = {
                "canonical": 1 if canonical_only else 0,
                "vcf_string": 1,
                "pick": 1,
                "dbNSFP": "REVEL_score,Ensembl_transcriptid",
                "SpliceAI": 1,
                "mane": 1,
                "numbers": 1,
            }

            for attempt in range(_RETRY_ATTEMPTS):
                try:
                    resp = client.post(
                        _VEP_BATCH_URL,
                        json=payload,
                        params=params,
                        headers={"Content-Type": "application/json", "Accept": "application/json"},
                    )
                    resp.raise_for_status()
                    annotations = resp.json()
                    break
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 429:
                        wait = float(exc.response.headers.get("Retry-After", _RETRY_DELAY * (attempt + 1)))
                        logger.warning("VEP rate-limited — waiting %.1fs", wait)
                        time.sleep(wait)
                    elif attempt < _RETRY_ATTEMPTS - 1:
                        time.sleep(_RETRY_DELAY)
                    else:
                        logger.error("VEP annotation failed: %s", exc)
                        annotations = []
                except Exception as exc:
                    if attempt < _RETRY_ATTEMPTS - 1:
                        time.sleep(_RETRY_DELAY)
                    else:
                        logger.error("VEP request error: %s", exc)
                        annotations = []
            else:
                annotations = []

            for ann in annotations:
                key = _parse_key(ann)
                results[key] = _parse_annotation(ann)

            time.sleep(_REQUEST_INTERVAL)

    return results


def _parse_key(ann: dict) -> str:
    input_str = ann.get("input", "")
    parts = input_str.split()
    if len(parts) >= 5:
        chrom, pos, _, ref, alt = parts[:5]
        return f"{chrom}:{pos}:{ref}:{alt}"
    return ann.get("id", "unknown")


def _best_transcript(ann: dict) -> dict:
    consequences = ann.get("transcript_consequences", [])
    if not consequences:
        return {}
    for tc in consequences:
        if tc.get("mane_select"):
            return tc
    for tc in consequences:
        if tc.get("canonical"):
            return tc
    return consequences[0]


def _mane_status(tc: dict) -> str | None:
    if tc.get("mane_select"):
        return "MANE_Select"
    if tc.get("mane_plus_clinical"):
        return "MANE_Plus_Clinical"
    return None


def _protein_position(tc: dict) -> str | None:
    start, end = tc.get("protein_start"), tc.get("protein_end")
    if start is None:
        return None
    return str(start) if start == end or end is None else f"{start}-{end}"


def _parse_annotation(ann: dict) -> dict:
    tc = _best_transcript(ann)
    all_terms = tc.get("consequence_terms") or []
    result: dict[str, Any] = {
        "gene": tc.get("gene_symbol") or tc.get("gene_id"),
        "transcript": tc.get("transcript_id"),
        "hgvs_c": tc.get("hgvsc"),
        "hgvs_p": tc.get("hgvsp"),
        "consequence": (all_terms or [""])[0],
        "all_consequence_terms": all_terms,
        "is_splice_relevant": any("splice" in t for t in all_terms),
        "mane_status": _mane_status(tc),
        "protein_position": _protein_position(tc),
        "exon": tc.get("exon"),
        "intron": tc.get("intron"),
        "impact": tc.get("impact"),
        "sift": tc.get("sift_prediction"),
        "sift_score": tc.get("sift_score"),
        "polyphen": tc.get("polyphen_prediction"),
        "polyphen_score": tc.get("polyphen_score"),
        "revel_score": _parse_revel(tc.get("revel_score"), tc.get("ensembl_transcriptid"), tc.get("transcript_id")),
        "spliceai_score": _parse_spliceai(tc.get("spliceai")),
        "most_severe_consequence": ann.get("most_severe_consequence"),
        "gnomad_af": None,
        "gnomad_af_popmax": None,
        "gnomad_af_by_population": {},
        "gnomad_allele_count": None,
        "gnomad_allele_number": None,
        "gnomad_homozygote_count": None,
        "gnomad_af_popmax_allele_number": None,
        "gnomad_filter_status": None,
        "gnomad_coverage": None,
        "clinvar_significance": None,
        "clinvar_id": None,
    }

    for cv in ann.get("colocated_variants", []):
        freqs = cv.get("frequencies", {})
        by_pop: dict[str, float] = {}
        for key, val in freqs.items():
            if not isinstance(val, (int, float)):
                continue
            key_lower = key.lower()
            if key_lower in ("gnomade_af", "gnomad_af", "af", "gnomade", "gnomad"):
                if result["gnomad_af"] is None:
                    result["gnomad_af"] = val
            else:
                by_pop[key] = val
        if by_pop:
            result["gnomad_af_by_population"].update(by_pop)
            result["gnomad_af_popmax"] = max(
                result["gnomad_af_popmax"] or 0.0, max(by_pop.values())
            )

        clin_sigs = cv.get("clin_sig", [])
        if clin_sigs:
            result["clinvar_significance"] = "; ".join(clin_sigs)
            result["clinvar_id"] = cv.get("id")

    result["gnomad_frequency_status"] = (
        "observed"
        if (result["gnomad_af"] is not None or result["gnomad_af_by_population"])
        else "not_evaluated"
    )

    return result

_VEP_CACHE_VERSION = os.environ.get("GENERESOLVER_VEP_CACHE_VERSION", "116")
_VEP_ASSEMBLY = os.environ.get("GENERESOLVER_VEP_ASSEMBLY", "GRCh37")


def _pred_label(v):
    return (v.split("(")[0] or None) if v else None


def _pred_score(v):
    if not v or "(" not in v:
        return None
    try:
        return float(v.split("(")[1].rstrip(")"))
    except ValueError:
        return None


def _flt(v):
    try:
        return float(v) if v not in (None, "", "&", ".") else None
    except (ValueError, TypeError):
        return None


def _parse_revel(revel_str, ensembl_transcriptid_str, picked_transcript_id: str | None) -> float | None:
    if not revel_str:
        return None
    scores = str(revel_str).split(",")
    transcripts = str(ensembl_transcriptid_str or "").split(",")
    if picked_transcript_id and len(transcripts) == len(scores):
        picked_bare = picked_transcript_id.split(".")[0]
        for score, tid in zip(scores, transcripts):
            if tid.split(".")[0] == picked_bare:
                f = _flt(score)
                if f is not None:
                    return f
    for part in scores:
        f = _flt(part)
        if f is not None:
            return f
    return None


_AA1_TO_AA3 = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln",
    "E": "Glu", "G": "Gly", "H": "His", "I": "Ile", "L": "Leu", "K": "Lys",
    "M": "Met", "F": "Phe", "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp",
    "Y": "Tyr", "V": "Val",
}


def _reconstruct_missense_hgvs_p(protein_position: str | None, amino_acids: str | None) -> str | None:
    if not protein_position or not amino_acids or "/" not in amino_acids:
        return None
    if not protein_position.isdigit():
        return None
    ref1, _, alt1 = amino_acids.partition("/")
    ref3, alt3 = _AA1_TO_AA3.get(ref1), _AA1_TO_AA3.get(alt1)
    if not ref3 or not alt3:
        return None
    return f"p.{ref3}{protein_position}{alt3}"


def _parse_spliceai(d: dict | None) -> float | None:
    if not d:
        return None
    scores = [_flt(d.get(k)) for k in ("DS_AG", "DS_AL", "DS_DG", "DS_DL")]
    scores = [s for s in scores if s is not None]
    return max(scores) if scores else None


def annotate_variants_local(variants, canonical_only: bool = True) -> dict[str, dict]:
    if not variants:
        return {}
    work = tempfile.mkdtemp(prefix="generesolver_vep_")
    in_vcf = os.path.join(work, "in.vcf")
    out_vcf = os.path.join(work, "out.vcf")
    rows = sorted({(str(c).replace("chr", ""), int(p), r, a) for c, p, r, a in variants},
                  key=lambda x: (x[0], x[1]))
    with open(in_vcf, "w", encoding="utf-8") as f:
        f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for c, p, r, a in rows:
            f.write(f"{c}\t{p}\t.\t{r}\t{a}\t.\t.\t.\n")
    cache = str(config.VEP_LOCAL_CACHE_DIR)
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{cache}:/data", "-v", f"{work}:/work",
        config.VEP_LOCAL_DOCKER_IMAGE, "vep",
        "--offline", "--cache", "--dir_cache", "/data",
        "--assembly", _VEP_ASSEMBLY, "--cache_version", _VEP_CACHE_VERSION,
        "--vcf", "--no_stats", "--pick", "--symbol", "--mane",
        "--sift", "b", "--polyphen", "b", "--af_gnomad", "--check_existing",
        "--max_af", "--domains",
        "-i", "/work/in.vcf", "-o", "/work/out.vcf", "--force_overwrite",
    ]
    try:
        subprocess.run(cmd, check=True, env={**os.environ, "MSYS_NO_PATHCONV": "1"},
                       capture_output=True)
    except subprocess.CalledProcessError as exc:
        logger.error("Local VEP failed: %s", (exc.stderr or b"").decode("utf-8", "ignore")[:600])
        return {}
    return _parse_vep_vcf(out_vcf)


def _parse_vep_vcf(out_vcf: str) -> dict[str, dict]:
    fields: list[str] = []
    results: dict[str, dict] = {}
    try:
        with open(out_vcf, encoding="utf-8") as f:
            for line in f:
                if line.startswith("##INFO=<ID=CSQ"):
                    m = re.search(r"Format: ([^\"]+)", line)
                    if m:
                        fields = m.group(1).strip().split("|")
                    continue
                if line.startswith("#"):
                    continue
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 8:
                    continue
                chrom, pos, _, ref, alt = cols[0], cols[1], cols[2], cols[3], cols[4]
                csq = next((kv[4:] for kv in cols[7].split(";") if kv.startswith("CSQ=")), None)
                if not csq:
                    continue
                d = dict(zip(fields, csq.split(",")[0].split("|")))
                all_terms = [t for t in (d.get("Consequence") or "").split("&") if t]
                cons = all_terms[0] if all_terms else ""
                mane_status = (
                    "MANE_Select" if d.get("MANE_SELECT")
                    else "MANE_Plus_Clinical" if d.get("MANE_PLUS_CLINICAL")
                    else None
                )
                by_pop: dict[str, float] = {}
                for fkey in fields:
                    m_pop = re.match(r"gnomAD([eg])_([A-Za-z]+)_AF$", fkey)
                    if not m_pop or m_pop.group(2) == "grpmax":
                        continue
                    pop_val = _flt(d.get(fkey))
                    if pop_val is not None:
                        by_pop[fkey] = pop_val
                gnomad_ac = _flt(d.get("gnomADe_AC")) or _flt(d.get("gnomADg_AC"))
                gnomad_an = _flt(d.get("gnomADe_AN")) or _flt(d.get("gnomADg_AN"))
                gnomad_nhomalt = _flt(d.get("gnomADe_nhomalt")) or _flt(d.get("gnomADg_nhomalt"))
                gnomad_af_local = _flt(d.get("gnomADe_AF") or d.get("gnomADg_AF"))
                gnomad_af_popmax_local = _flt(d.get("gnomADe_grpmax_AF") or d.get("gnomADg_grpmax_AF"))
                max_af = _flt(d.get("MAX_AF"))
                if gnomad_af_popmax_local is None and max_af is not None:
                    gnomad_af_popmax_local = max_af
                elif gnomad_af_popmax_local is None and by_pop:
                    gnomad_af_popmax_local = max(by_pop.values())
                gnomad_af_popmax_an_local = _flt(d.get("gnomADe_grpmax_AN") or d.get("gnomADg_grpmax_AN"))
                gnomad_frequency_status = (
                    "observed" if (gnomad_af_local is not None or by_pop) else "not_evaluated"
                )
                results[f"{chrom.lstrip('chr')}:{pos}:{ref}:{alt}"] = {
                    "gene": d.get("SYMBOL") or d.get("Gene") or None,
                    "transcript": d.get("Feature") or None,
                    "hgvs_c": d.get("HGVSc") or None,
                    "hgvs_p": d.get("HGVSp") or _reconstruct_missense_hgvs_p(
                        d.get("Protein_position"), d.get("Amino_acids")
                    ),
                    "consequence": cons,
                    "all_consequence_terms": all_terms,
                    "is_splice_relevant": any("splice" in t for t in all_terms),
                    "mane_status": mane_status,
                    "protein_position": d.get("Protein_position") or None,
                    "exon": d.get("EXON") or None,
                    "intron": d.get("INTRON") or None,
                    "impact": d.get("IMPACT") or None,
                    "domains": [x for x in (d.get("DOMAINS") or "").split("&") if x],
                    "sift": _pred_label(d.get("SIFT")),
                    "sift_score": _pred_score(d.get("SIFT")),
                    "polyphen": _pred_label(d.get("PolyPhen")),
                    "polyphen_score": _pred_score(d.get("PolyPhen")),
                    "most_severe_consequence": cons,
                    "gnomad_af": gnomad_af_local,
                    "gnomad_af_popmax": gnomad_af_popmax_local,
                    "gnomad_af_by_population": by_pop,
                    "gnomad_allele_count": int(gnomad_ac) if gnomad_ac is not None else None,
                    "gnomad_allele_number": int(gnomad_an) if gnomad_an is not None else None,
                    "gnomad_af_popmax_allele_number": (
                        int(gnomad_af_popmax_an_local) if gnomad_af_popmax_an_local is not None else None
                    ),
                    "gnomad_homozygote_count": int(gnomad_nhomalt) if gnomad_nhomalt is not None else None,
                    "gnomad_filter_status": None,
                    "gnomad_coverage": None,
                    "gnomad_frequency_status": gnomad_frequency_status,
                    "clinvar_significance": ((d.get("CLIN_SIG") or "").replace("&", "; ") or None),
                    "clinvar_id": d.get("Existing_variation") or None,
                }
    except FileNotFoundError:
        logger.error("Local VEP produced no output: %s", out_vcf)
    return results
