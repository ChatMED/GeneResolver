from __future__ import annotations

import gzip
import json
import os
import re
import subprocess
import shutil
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
_VEP_GRCH37_BATCH_URL = "https://grch37.rest.ensembl.org/vep/human/region"
_BATCH_SIZE = 50
_RETRY_ATTEMPTS = 3
_RETRY_DELAY = 2.0
_REQUEST_INTERVAL = 0.07

_RESULT_CACHE_DIR = config.DATA_DIR / "vep_cache"
# v3: every key carries a fingerprint of the exact annotation configuration (mode, VEP
# image/version, cache version, flags, parser version), so a change to any of them can no
# longer silently reuse annotations produced by another configuration. v2 is left untouched.
_RESULT_CACHE_PATH = _RESULT_CACHE_DIR / "vep_annotations_v3.json.gz"
_result_cache: dict[str, dict] | None = None
_result_cache_dirty = False

# Bump when _parse_vep_vcf / _parse_annotation change what they emit.
_PARSER_VERSION = "3"

_VEP_CACHE_VERSION = os.environ.get("GENERESOLVER_VEP_CACHE_VERSION", "116")
_VEP_ASSEMBLY = os.environ.get("GENERESOLVER_VEP_ASSEMBLY", "GRCh37")

# Local VEP options, kept in one place so the exact command is recorded in provenance.
_LOCAL_VEP_FLAGS = [
    "--offline", "--cache", "--vcf", "--no_stats", "--pick", "--symbol", "--mane",
    "--sift", "b", "--polyphen", "b", "--af_gnomad", "--check_existing",
    "--max_af", "--domains",
]

_REST_PARAMS_BASE = {"vcf_string": 1, "pick": 1, "dbNSFP": "REVEL_score,Ensembl_transcriptid",
                     "SpliceAI": 1, "numbers": 1}

_metadata_cache: dict[str, dict] = {}


def _reference_fasta(genome_build: str | None) -> Path | None:
    env_key = {"GRCh37": "GENERESOLVER_REFERENCE_FASTA_GRCH37",
               "GRCh38": "GENERESOLVER_REFERENCE_FASTA_GRCH38"}.get(genome_build or "")
    p = os.environ.get(env_key) if env_key else None
    if p and Path(p).exists() and Path(p + ".fai").exists():
        return Path(p)
    return None


def _local_flags(genome_build: str | None) -> list[str]:
    # HGVS needs the reference sequence in offline mode; add it only when a FASTA is configured,
    # so the reported (FASTA-less) configuration is unchanged.
    flags = list(_LOCAL_VEP_FLAGS)
    if _reference_fasta(genome_build):
        flags += ["--hgvs", "--shift_hgvs", "1", "--fasta", "/fasta/" + _reference_fasta(genome_build).name]
    return flags


def _read_cache_info(genome_build: str) -> dict[str, str]:
    info = config.VEP_LOCAL_CACHE_DIR / "homo_sapiens" / f"{_VEP_CACHE_VERSION}_{genome_build}" / "info.txt"
    out: dict[str, str] = {}
    try:
        for line in info.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("\t")
            if k.startswith("source_") or k in ("assembly", "sift", "polyphen"):
                out[k] = v
    except OSError:
        pass
    return out


def _docker_image_id(image: str) -> str | None:
    if not shutil.which("docker"):
        return None
    r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def _rest_release(genome_build: str) -> str:
    base = "https://grch37.rest.ensembl.org" if genome_build == "GRCh37" else "https://rest.ensembl.org"
    try:
        r = httpx.get(f"{base}/info/software", headers={"Accept": "application/json"}, timeout=10)
        return str(r.json().get("release"))
    except Exception:
        return "unknown"


def annotation_metadata(genome_build: str | None) -> dict:
    """Exact annotation configuration: written to provenance and hashed into cache keys."""
    build = genome_build or _VEP_ASSEMBLY
    key = f"{'local' if config.LOCAL_MODE else 'rest'}:{build}"
    if key in _metadata_cache:
        return _metadata_cache[key]
    if config.LOCAL_MODE:
        meta = {
            "mode": "local_docker",
            "vep_image": config.VEP_LOCAL_DOCKER_IMAGE,
            "vep_image_id": _docker_image_id(config.VEP_LOCAL_DOCKER_IMAGE),
            "vep_release": _VEP_CACHE_VERSION,
            "cache": f"{_VEP_CACHE_VERSION}_{build}",
            "cache_dir": str(config.VEP_LOCAL_CACHE_DIR),
            "cache_sources": _read_cache_info(build) or "info.txt NOT FOUND -- cache contents unverified",
            "flags": _local_flags(build),
            "plugins": [],
            "pick_order": "VEP default (--pick without --pick_order)",
            "parser_version": _PARSER_VERSION,
        }
    else:
        meta = {
            "mode": "ensembl_rest",
            "server": "grch37.rest.ensembl.org" if build == "GRCh37" else "rest.ensembl.org",
            "ensembl_release": _rest_release(build),
            "params": {**_REST_PARAMS_BASE, **({"mane": 1} if build == "GRCh38" else {})},
            "plugins": ["dbNSFP (REVEL_score)", "SpliceAI"],
            "parser_version": _PARSER_VERSION,
        }
    import hashlib
    fp_src = json.dumps(meta, sort_keys=True)
    meta["fingerprint"] = hashlib.sha256(fp_src.encode()).hexdigest()[:12]
    _metadata_cache[key] = meta
    return meta


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
    chrom_clean = chrom[3:] if chrom.lower().startswith("chr") else chrom
    return f"{chrom_clean} {pos} . {ref} {alt} . . ."


def annotate_variants(
    variants: list[tuple[str, int, str, str]],
    canonical_only: bool = True,
    genome_build: str | None = None,
) -> dict[str, dict]:
    if not variants:
        return {}

    if genome_build not in {"GRCh37", "GRCh38"}:
        raise RuntimeError(f"VEP annotation requires a confirmed GRCh37 or GRCh38 assembly; got {genome_build!r}.")

    if config.LOCAL_MODE:
        _check_local_vep(genome_build)
    fp = annotation_metadata(genome_build)["fingerprint"]
    cache = _load_result_cache()
    results: dict[str, dict] = {}
    misses: list[tuple[str, int, str, str]] = []
    for c, p, r, a in variants:
        key = _variant_key(c, p, r, a)
        cached = cache.get(f"{fp}:{genome_build}:{key}")
        if cached is not None:
            results[key] = cached
        else:
            misses.append((c, p, r, a))

    if misses:
        fresh = (
            annotate_variants_local(misses, canonical_only, genome_build) if config.LOCAL_MODE
            else _annotate_variants_online(misses, canonical_only, genome_build)
        )
        global _result_cache_dirty
        for key, ann in fresh.items():
            cache[f"{fp}:{genome_build}:{key}"] = ann
            results[key] = ann
            _result_cache_dirty = True
        _save_result_cache()

    missing = [_variant_key(c, p, r, a) for c, p, r, a in variants
               if _variant_key(c, p, r, a) not in results]
    if missing:
        raise RuntimeError(f"VEP returned no annotation for {len(missing)} of {len(variants)} variants; "
                           f"first missing: {missing[0]}. Variant analysis stopped.")
    return results


def _annotate_variants_online(
    variants: list[tuple[str, int, str, str]],
    canonical_only: bool = True,
    genome_build: str = "GRCh38",
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
            params = {"canonical": 1 if canonical_only else 0, **_REST_PARAMS_BASE}
            if genome_build == "GRCh38":
                params["mane"] = 1

            for attempt in range(_RETRY_ATTEMPTS):
                try:
                    resp = client.post(
                        _VEP_GRCH37_BATCH_URL if genome_build == "GRCh37" else _VEP_BATCH_URL,
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
        return _variant_key(chrom, int(pos), ref, alt)
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

    # REST frequencies are keyed by allele: {"A": {"gnomade": .., "gnomade_afr": .., "af": ..}}.
    # Only gnomAD exome values are used (the same dataset as local --af_gnomad); 1000 Genomes
    # ("af", "afr", ...) and gnomAD genome ("gnomadg*") values are deliberately ignored.
    alt_allele = _input_alt_allele(ann)
    for cv in ann.get("colocated_variants", []):
        freqs_by_allele = cv.get("frequencies") or {}
        freqs = freqs_by_allele.get(alt_allele) if alt_allele in freqs_by_allele else (
            next(iter(freqs_by_allele.values())) if len(freqs_by_allele) == 1 else None)
        if isinstance(freqs, dict):
            for key, val in freqs.items():
                if not isinstance(val, (int, float)):
                    continue
                k = key.lower()
                if k == "gnomade" and result["gnomad_af"] is None:
                    result["gnomad_af"] = val
                elif k.startswith("gnomade_"):
                    # Same key spelling as the local parser so ACMG/filter code reads both.
                    result["gnomad_af_by_population"][f"gnomADe_{k[8:].upper()}_AF"] = val
        id_ = cv.get("id") or ""
        if id_.startswith("rs") and not result.get("dbsnp_id"):
            result["dbsnp_id"] = id_
        clin_sigs = cv.get("clin_sig", [])
        if clin_sigs:
            result["clinvar_significance"] = "; ".join(clin_sigs)

    cont = [v for k, v in result["gnomad_af_by_population"].items()
            if _CONTINENTAL_KEY_RE.match(k)]
    if cont:
        result["gnomad_af_popmax"] = max(cont)
        result["gnomad_af_popmax_source"] = "gnomADe_continental_max"
    result["gnomad_frequency_status"] = (
        "observed"
        if (result["gnomad_af"] is not None or result["gnomad_af_by_population"])
        else "not_evaluated"
    )

    return result


_CONTINENTAL_KEY_RE = re.compile(r"^gnomADe_(AFR|AMR|EAS|NFE|SAS)_AF$")


def _input_alt_allele(ann: dict) -> str | None:
    parts = (ann.get("input") or "").split()
    if len(parts) < 5:
        return None
    ref, alt = parts[3], parts[4]
    # VEP keys frequencies by its minimal allele: shared leading base removed, "-" if empty.
    if len(ref) != len(alt) and ref[:1] == alt[:1]:
        alt = alt[1:] or "-"
    return alt

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
    if v is None:
        return None
    # VEP joins values from co-located known variants with "&" (e.g. "0.1351&" when an
    # HGMD/COSMIC record without frequencies shares the position); take the first number.
    for part in str(v).split("&"):
        if part in ("", "."):
            continue
        try:
            return float(part)
        except ValueError:
            continue
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


def _check_local_vep(genome_build: str | None) -> None:
    if genome_build not in {"GRCh37", "GRCh38"}:
        raise RuntimeError(f"Local VEP requires a confirmed GRCh37 or GRCh38 assembly; got {genome_build!r}.")
    if not shutil.which("docker"):
        raise RuntimeError("Local VEP requires Docker on PATH. Install/start Docker before VCF annotation.")
    cache = config.VEP_LOCAL_CACHE_DIR.resolve()
    assembly_dir = cache / "homo_sapiens" / f"{_VEP_CACHE_VERSION}_{genome_build}"
    if not cache.is_dir() or not assembly_dir.is_dir() or not any(assembly_dir.iterdir()):
        raise RuntimeError(f"Local VEP cache for {genome_build} version {_VEP_CACHE_VERSION} missing under {cache}. "
                           "Run setup_quickstart.py --with-vep-cache or set the matching VEP cache/version.")
    probe = subprocess.run(["docker", "image", "inspect", config.VEP_LOCAL_DOCKER_IMAGE],
                           capture_output=True, text=True)
    if probe.returncode:
        raise RuntimeError(f"VEP Docker image {config.VEP_LOCAL_DOCKER_IMAGE} unavailable. "
                           "Pull the image and verify it contains the vep executable.")
    # VEP has no --version flag; --help prints "ensembl-vep : <release>".
    version = subprocess.run(["docker", "run", "--rm", config.VEP_LOCAL_DOCKER_IMAGE,
                              "vep", "--help"], capture_output=True, text=True)
    found = re.search(r"ensembl-vep\s*:\s*(\d+)", version.stdout + version.stderr)
    if not found or found.group(1) != str(_VEP_CACHE_VERSION):
        raise RuntimeError(f"VEP executable/image version does not match cache version {_VEP_CACHE_VERSION}: "
                           f"{found.group(0) if found else (version.stdout + version.stderr)[:200]}")
    # The host folder can exist while Docker cannot see it (Docker Desktop file sharing, a
    # path on another drive), which makes VEP fail with "Cache directory /data/homo_sapiens
    # not found". Check once from inside the container.
    key = (str(cache), config.VEP_LOCAL_DOCKER_IMAGE, genome_build)
    if key not in _MOUNT_VERIFIED:
        inside = f"/data/homo_sapiens/{_VEP_CACHE_VERSION}_{genome_build}"
        seen = subprocess.run(["docker", "run", "--rm", "-v", f"{cache}:/data", config.VEP_LOCAL_DOCKER_IMAGE,
                               "ls", inside], capture_output=True, text=True)
        if seen.returncode or not seen.stdout.strip():
            raise RuntimeError(f"Docker cannot see the VEP cache: {cache} is mounted at /data but {inside} is "
                               "missing or empty inside the container. Share this folder with Docker "
                               "(Docker Desktop > Settings > Resources > File sharing) or set "
                               "GENERESOLVER_VEP_CACHE_DIR to a shared path.")
        _MOUNT_VERIFIED.add(key)


_MOUNT_VERIFIED: set[tuple[str, str, str]] = set()


def annotate_variants_local(variants, canonical_only: bool = True,
                            genome_build: str | None = None) -> dict[str, dict]:
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
    build = genome_build or _VEP_ASSEMBLY
    mounts = ["-v", f"{cache}:/data", "-v", f"{work}:/work"]
    fasta = _reference_fasta(build)
    if fasta:
        mounts += ["-v", f"{fasta.parent}:/fasta"]
    cmd = [
        "docker", "run", "--rm", *mounts,
        config.VEP_LOCAL_DOCKER_IMAGE, "vep",
        "--dir_cache", "/data", "--assembly", build, "--cache_version", _VEP_CACHE_VERSION,
        *_local_flags(build),
        "-i", "/work/in.vcf", "-o", "/work/out.vcf", "--force_overwrite",
    ]
    try:
        subprocess.run(cmd, check=True, env={**os.environ, "MSYS_NO_PATHCONV": "1"},
                       capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Local VEP failed: " + (exc.stderr or b"").decode("utf-8", "ignore")[:600]) from exc
    result = _parse_vep_vcf(out_vcf)
    # Keep VEP's own header record (version, cache, data sources) for provenance.
    try:
        with open(out_vcf, encoding="utf-8") as f:
            vep_header = [ln.rstrip("\n") for ln in f
                          if ln.startswith("##VEP") and not ln.startswith("##VEP-command-line")]
        if vep_header:
            annotation_metadata(build)["vep_output_header"] = vep_header[:5]
    except OSError:
        pass
    if not result:
        raise RuntimeError("Local VEP produced no annotations; inspect cache mount, assembly and Docker image.")
    return result


def _parse_vep_vcf(out_vcf: str) -> dict[str, dict]:
    fields: list[str] = []
    gnomad_fields_present = False
    results: dict[str, dict] = {}
    try:
        with open(out_vcf, encoding="utf-8") as f:
            for line in f:
                if line.startswith("##INFO=<ID=CSQ"):
                    m = re.search(r"Format: ([^\"]+)", line)
                    if m:
                        fields = m.group(1).strip().split("|")
                        gnomad_fields_present = any(
                            f in ("gnomADe_AF", "gnomADg_AF") for f in fields)
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
                # Kept for the original profile; tagged so the revised profile can refuse it
                # (MAX_AF mixes 1000 Genomes and gnomAD populations).
                popmax_source = "gnomAD_grpmax" if gnomad_af_popmax_local is not None else None
                if gnomad_af_popmax_local is None and max_af is not None:
                    gnomad_af_popmax_local = max_af
                    popmax_source = "VEP_MAX_AF"
                elif gnomad_af_popmax_local is None and by_pop:
                    gnomad_af_popmax_local = max(by_pop.values())
                    popmax_source = "gnomAD_all_groups_max"
                gnomad_af_popmax_an_local = _flt(d.get("gnomADe_grpmax_AN") or d.get("gnomADg_grpmax_AN"))
                gnomad_frequency_status = (
                    "observed" if (gnomad_af_local is not None or by_pop)
                    else "absent" if gnomad_fields_present else "not_evaluated"
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
                    "gnomad_af_popmax_source": popmax_source,
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
                    # Existing_variation holds rsIDs/COSMIC IDs, never a ClinVar VariationID.
                    "clinvar_id": None,
                    "dbsnp_id": next((x for x in (d.get("Existing_variation") or "").split("&")
                                      if x.startswith("rs")), None),
                }
    except FileNotFoundError:
        logger.error("Local VEP produced no output: %s", out_vcf)
    return results
