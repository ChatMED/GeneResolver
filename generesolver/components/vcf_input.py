from __future__ import annotations

import logging
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import vcfpy

from generesolver.schemas.case import CaseState
from generesolver.schemas.genomic import GenomeBuild, QCStatus, VCFContext
from generesolver.schemas.report import ProvenanceItem

logger = logging.getLogger(__name__)

_MAX_VARIANTS = 50_000

_VARIANT_SAMPLE_SEED = 20260811


_BUILD_PATTERNS = {
    GenomeBuild.GRCH38: re.compile(r"grch38|hg38|b38", re.IGNORECASE),
    GenomeBuild.GRCH37: re.compile(r"grch37|hg19|b37|hs37", re.IGNORECASE),
}


def _detect_build_from_header(header: vcfpy.Header) -> GenomeBuild:
    for line in header.lines:
        value = ""
        if hasattr(line, "value"):
            value = str(line.value)
        elif hasattr(line, "mapping"):
            value = str(line.mapping)
        for build, pattern in _BUILD_PATTERNS.items():
            if pattern.search(value):
                return build

    try:
        contig_lines = header.get_lines("contig")
        contig_names = [
            ln.mapping.get("ID", "")
            for ln in contig_lines
            if hasattr(ln, "mapping")
        ]
    except Exception:
        contig_names = []

    if contig_names:
        if contig_names[0].startswith("chr"):
            return GenomeBuild.GRCH38
        else:
            return GenomeBuild.GRCH37
    return GenomeBuild.UNKNOWN


def _is_dragen_header(header: vcfpy.Header) -> bool:
    for line in header.lines:
        if hasattr(line, "key") and "DRAGEN" in str(line.key).upper():
            return True
        if hasattr(line, "value") and "DRAGEN" in str(line.value).upper():
            return True
    return False


def _find_vcf_in_dragen_dir(dragen_dir: str) -> str | None:
    p = Path(dragen_dir)
    for pattern in (
        "*.hard-filtered.vcf.gz",
        "*.hard-filtered.vcf",
        "*.vcf.gz",
        "*.vcf",
    ):
        matches = sorted(p.glob(pattern))
        if matches:
            return str(matches[0])
    return None


def _dragen_qc_warnings(dragen_dir: str) -> list[str]:
    warnings: list[str] = []
    p = Path(dragen_dir)
    if not list(p.glob("*_mapping_metrics.csv")):
        warnings.append("DRAGEN mapping metrics file not found.")
    if not list(p.glob("*_vc_metrics.csv")) and not list(p.glob("*_variant_calling_metrics.csv")):
        warnings.append("DRAGEN variant calling metrics file not found.")
    return warnings

def _parse_vcf_file(vcf_path: str) -> VCFContext:
    warnings: list[str] = []
    path = Path(vcf_path)

    if not path.exists():
        return VCFContext(
            file_path=vcf_path,
            qc_status=QCStatus.FAIL,
            warnings=[f"VCF file not found: {vcf_path}"],
        )

    file_type = "VCF.GZ" if vcf_path.endswith(".gz") else "VCF"

    try:
        reader = vcfpy.Reader.from_path(vcf_path)
    except Exception as exc:
        return VCFContext(
            file_path=vcf_path,
            file_type=file_type,
            qc_status=QCStatus.FAIL,
            warnings=[f"Failed to open VCF: {exc}"],
        )

    header = reader.header
    genome_build = _detect_build_from_header(header)
    is_dragen = _is_dragen_header(header)

    sample_names = header.samples.names
    sample_id = sample_names[0] if sample_names else None

    if genome_build == GenomeBuild.UNKNOWN:
        warnings.append("Genome build could not be determined from VCF header.")

    if not sample_id:
        warnings.append("No sample ID found in VCF header.")

    if len(sample_names) > 1:
        warnings.append(
            f"Multiple samples in VCF ({len(sample_names)}). "
            "Only the first sample will be used."
        )

    variant_count = 0
    pass_count = 0
    capped = False

    try:
        for record in reader:
            variant_count += 1
            filters = [f.value if hasattr(f, "value") else str(f) for f in (record.FILTER or [])]
            if not filters or filters == ["PASS"]:
                pass_count += 1
            if _MAX_VARIANTS > 0 and variant_count >= _MAX_VARIANTS * 10:
                capped = True
                break
    except Exception as exc:
        warnings.append(f"Error reading VCF records: {exc}")

    reader.close()

    if capped:
        warnings.append(
            f"VCF has >={variant_count} variants. "
            "Large file detected — annotation will be limited to first "
            f"{_MAX_VARIANTS} PASS variants."
        )

    if _MAX_VARIANTS > 0 and pass_count > _MAX_VARIANTS:
        warnings.append(
            f"VCF contains {pass_count} PASS variants — annotation will be limited "
            f"to the first {_MAX_VARIANTS} PASS variants."
        )

    qc_status = QCStatus.PASS
    if genome_build == GenomeBuild.UNKNOWN:
        qc_status = QCStatus.WARN
    if warnings:
        qc_status = QCStatus.WARN

    return VCFContext(
        file_path=vcf_path,
        file_type="DRAGEN" if is_dragen else file_type,
        sample_id=sample_id,
        genome_build=genome_build,
        variant_count=variant_count,
        pass_variant_count=pass_count,
        qc_status=qc_status,
        warnings=warnings,
    )

_MIN_QUAL = float(os.environ.get("GENERESOLVER_MIN_QUAL", "20") or 0)


def _has_alt_genotype(record) -> bool:
    calls = getattr(record, "calls", None)
    if not calls:
        return True
    gt = (getattr(calls[0], "data", {}) or {}).get("GT")
    if not gt:
        return True
    return any(a not in ("0", ".", "") for a in re.split(r"[/|]", str(gt)))


def load_pass_variants(
    vcf_path: str,
    max_variants: int | None = None,
    *,
    min_qual: float = _MIN_QUAL,
    require_non_ref: bool = True,
) -> tuple[list[tuple[str, int, str, str]], int]:
    if max_variants is None:
        max_variants = _MAX_VARIANTS
    uncapped = max_variants <= 0
    rng = random.Random(_VARIANT_SAMPLE_SEED)
    variants: list[tuple[str, int, str, str]] = []
    n_skipped_structural = 0
    n_eligible = 0
    try:
        reader = vcfpy.Reader.from_path(vcf_path)
        for record in reader:
            filters = [f.value if hasattr(f, "value") else str(f) for f in (record.FILTER or [])]
            if filters and filters != ["PASS"]:
                continue
            if min_qual and record.QUAL is not None and float(record.QUAL) < min_qual:
                continue
            if require_non_ref and not _has_alt_genotype(record):
                continue
            if len(record.REF) > 50:
                continue
            for alt in record.ALT:
                if not isinstance(alt, vcfpy.Substitution):
                    n_skipped_structural += 1
                    continue
                alt_str = str(alt.value)
                if len(alt_str) > 50:
                    continue
                item = (record.CHROM, record.POS, record.REF, alt_str)
                if uncapped or n_eligible < max_variants:
                    variants.append(item)
                else:
                    j = rng.randint(0, n_eligible)
                    if j < max_variants:
                        variants[j] = item
                n_eligible += 1
        reader.close()
        variants.sort(key=lambda v: (v[0], v[1]))
    except Exception as exc:
        logger.error("Failed to load variants from %s: %s", vcf_path, exc)
    return variants, n_skipped_structural



@dataclass
class NormalizedVariant:
    chrom: str
    pos: int
    ref: str
    alt: str
    genotype: str | None = None
    zygosity: str | None = None
    allele_depth_ref: int | None = None
    allele_depth_alt: int | None = None
    read_depth: int | None = None
    quality: float | None = None
    filter_status: str | None = None
    was_multiallelic: bool = False
    was_indel_normalized: bool = False
    ref_matches_reference: bool | None = None



def _trim_variant(pos: int, ref: str, alt: str) -> tuple[int, str, str, bool]:
    orig = (pos, ref, alt)
    while len(ref) > 1 and len(alt) > 1 and ref[-1] == alt[-1]:
        ref, alt = ref[:-1], alt[:-1]
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref, alt = ref[1:], alt[1:]
        pos += 1
    return pos, ref, alt, (pos, ref, alt) != orig



class _IndexedFasta:

    def __init__(self, fasta_path: Path):
        self.fasta_path = fasta_path
        self.fai_path = Path(str(fasta_path) + ".fai")
        self._index: dict[str, tuple[int, int, int, int]] = {}
        with self.fai_path.open(encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                name, length, offset, linebases, linewidth = parts[:5]
                self._index[name] = (int(length), int(offset), int(linebases), int(linewidth))
        self._fh = fasta_path.open("rb")

    def fetch(self, chrom: str, start: int, end: int) -> str | None:
        chrom = chrom if chrom in self._index else chrom.replace("chr", "", 1)
        entry = self._index.get(chrom)
        if entry is None:
            return None
        length, offset, linebases, linewidth = entry
        start0 = max(0, start - 1)
        end0 = min(length, end)
        if start0 >= end0:
            return ""
        seq = bytearray()
        pos = start0
        while pos < end0:
            line_idx, col = divmod(pos, linebases)
            file_pos = offset + line_idx * linewidth + col
            self._fh.seek(file_pos)
            chunk = self._fh.read(min(linebases - col, end0 - pos))
            seq += chunk.replace(b"\n", b"")
            pos += len(chunk)
        return seq.decode("ascii", errors="replace").upper()

    def close(self) -> None:
        self._fh.close()


_FASTA_CACHE: dict[GenomeBuild, _IndexedFasta | None] = {}


def _get_reference_fasta(genome_build: GenomeBuild) -> _IndexedFasta | None:
    if genome_build in _FASTA_CACHE:
        return _FASTA_CACHE[genome_build]
    env_key = {
        GenomeBuild.GRCH37: "GENERESOLVER_REFERENCE_FASTA_GRCH37",
        GenomeBuild.GRCH38: "GENERESOLVER_REFERENCE_FASTA_GRCH38",
    }.get(genome_build)
    reader = None
    path_str = os.environ.get(env_key) if env_key else None
    if path_str:
        path = Path(path_str)
        if path.exists() and Path(str(path) + ".fai").exists():
            try:
                reader = _IndexedFasta(path)
            except Exception as exc:
                logger.warning("Failed to open reference FASTA %s: %s", path, exc)
        else:
            logger.warning(
                "%s=%s set but file or .fai index not found -- "
                "REF validation/left-alignment will be skipped.", env_key, path_str,
            )
    _FASTA_CACHE[genome_build] = reader
    return reader


def _left_align_with_reference(
    chrom: str, pos: int, ref: str, alt: str, fasta: _IndexedFasta,
) -> tuple[int, str, str, bool]:
    was_changed = False
    for _ in range(200):
        if len(ref) == len(alt):
            break
        left_base = fasta.fetch(chrom, pos - 1, pos - 1)
        if not left_base:
            break
        if ref[-1] != alt[-1] if ref and alt else True:
            if len(ref) > len(alt) and ref[-len(alt):] != alt:
                break
            if len(alt) > len(ref) and alt[-len(ref):] != ref:
                break
        pos -= 1
        ref = left_base + ref[:-1] if len(ref) > 1 else left_base
        alt = left_base + alt[:-1] if len(alt) > 1 else left_base
        was_changed = True
        pos2, ref2, alt2, _ = _trim_variant(pos, ref, alt)
        pos, ref, alt = pos2, ref2, alt2
    return pos, ref, alt, was_changed


def _validate_and_normalize(
    chrom: str, pos: int, ref: str, alt: str, genome_build: GenomeBuild,
) -> tuple[int, str, str, bool, bool | None]:
    pos, ref, alt, changed = _trim_variant(pos, ref, alt)
    fasta = _get_reference_fasta(genome_build)
    if fasta is None:
        return pos, ref, alt, changed, None
    ref_matches = None
    observed = fasta.fetch(chrom, pos, pos + len(ref) - 1)
    if observed is not None:
        ref_matches = observed == ref
    pos, ref, alt, changed2 = _left_align_with_reference(chrom, pos, ref, alt, fasta)
    return pos, ref, alt, (changed or changed2), ref_matches



def _zygosity_for_alt(gt: str, alt_index: int) -> str | None:
    alleles = [a for a in re.split(r"[/|]", gt) if a not in (".", "")]
    if not alleles:
        return None
    try:
        int_alleles = [int(a) for a in alleles]
    except ValueError:
        return None
    count = int_alleles.count(alt_index)
    if count == 0:
        return "absent"
    if len(int_alleles) == 1:
        return "hemizygous"
    if count == len(int_alleles):
        return "homozygous"
    return "heterozygous"


def _extract_ad_dp(call_data: dict, alt_index: int) -> tuple[int | None, int | None, int | None]:
    ad = call_data.get("AD")
    ref_depth = alt_depth = None
    if ad:
        try:
            ad_list = [int(x) for x in (ad if isinstance(ad, list) else [ad])]
            if ad_list:
                ref_depth = ad_list[0]
                if len(ad_list) > alt_index:
                    alt_depth = ad_list[alt_index]
        except (ValueError, TypeError):
            pass
    dp = call_data.get("DP")
    try:
        total_depth = int(dp) if dp not in (None, ".", "") else None
    except (ValueError, TypeError):
        total_depth = None
    return ref_depth, alt_depth, total_depth


def load_normalized_variants(
    vcf_path: str,
    genome_build: GenomeBuild = GenomeBuild.UNKNOWN,
    max_variants: int | None = None,
    *,
    min_qual: float = _MIN_QUAL,
    require_non_ref: bool = True,
) -> tuple[list[NormalizedVariant], int]:
    if max_variants is None:
        max_variants = _MAX_VARIANTS
    uncapped = max_variants <= 0
    rng = random.Random(_VARIANT_SAMPLE_SEED)
    variants: list[NormalizedVariant] = []
    n_skipped_structural = 0
    n_eligible = 0
    try:
        reader = vcfpy.Reader.from_path(vcf_path)
        for record in reader:
            filters = [f.value if hasattr(f, "value") else str(f) for f in (record.FILTER or [])]
            filter_status = ";".join(filters) if filters else "PASS"
            if filters and filters != ["PASS"]:
                continue
            if min_qual and record.QUAL is not None and float(record.QUAL) < min_qual:
                continue

            calls = getattr(record, "calls", None)
            call_data = (getattr(calls[0], "data", {}) or {}) if calls else {}
            gt = str(call_data.get("GT") or "")

            if require_non_ref and gt and not any(
                a not in ("0", ".", "") for a in re.split(r"[/|]", gt)
            ):
                continue

            if len(record.REF) > 50:
                continue

            is_multiallelic = len(record.ALT) > 1
            for alt_idx, alt in enumerate(record.ALT, start=1):
                if not isinstance(alt, vcfpy.Substitution):
                    n_skipped_structural += 1
                    continue
                alt_str = str(alt.value)
                if len(alt_str) > 50:
                    continue

                zygosity = _zygosity_for_alt(gt, alt_idx) if gt else None
                if zygosity == "absent":
                    continue

                norm_pos, norm_ref, norm_alt, was_changed, ref_matches = _validate_and_normalize(
                    record.CHROM, record.POS, record.REF, alt_str, genome_build,
                )
                ref_depth, alt_depth, total_depth = _extract_ad_dp(call_data, alt_idx)

                item = NormalizedVariant(
                    chrom=record.CHROM,
                    pos=norm_pos,
                    ref=norm_ref,
                    alt=norm_alt,
                    genotype=gt or None,
                    zygosity=zygosity,
                    allele_depth_ref=ref_depth,
                    allele_depth_alt=alt_depth,
                    read_depth=total_depth,
                    quality=float(record.QUAL) if record.QUAL is not None else None,
                    filter_status=filter_status,
                    was_multiallelic=is_multiallelic,
                    was_indel_normalized=was_changed,
                    ref_matches_reference=ref_matches,
                )
                if uncapped or n_eligible < max_variants:
                    variants.append(item)
                else:
                    j = rng.randint(0, n_eligible)
                    if j < max_variants:
                        variants[j] = item
                n_eligible += 1
        reader.close()
        variants.sort(key=lambda v: (v.chrom, v.pos))
    except Exception as exc:
        logger.error("Failed to load/normalize variants from %s: %s", vcf_path, exc)
    return variants, n_skipped_structural


def run(state: CaseState) -> CaseState:
    vcf_path = state.input.vcf_path
    dragen_dir = state.input.dragen_dir

    if dragen_dir and not vcf_path:
        vcf_path = _find_vcf_in_dragen_dir(dragen_dir)
        if vcf_path:
            dragen_warnings = _dragen_qc_warnings(dragen_dir)
        else:
            state.vcf_context = VCFContext(
                qc_status=QCStatus.FAIL,
                warnings=["No VCF file found in DRAGEN directory."],
            )
            state.add_safety_flag("No VCF found in DRAGEN bundle — variant interpretation blocked.")
            state.mark_done("vcf_input")
            return state

    if not vcf_path:
        state.add_warning("VCF input skipped: no file provided.")
        state.mark_done("vcf_input")
        return state

    context = _parse_vcf_file(vcf_path)

    if dragen_dir:
        context.file_type = "DRAGEN"
        context.warnings.extend(dragen_warnings)
        if dragen_warnings:
            context.qc_status = QCStatus.WARN

    state.vcf_context = context

    if context.qc_status == QCStatus.FAIL:
        state.add_safety_flag("VCF quality check FAILED — variant interpretation blocked.")
    elif context.qc_status == QCStatus.WARN:
        for w in context.warnings:
            state.add_warning(w)
        state.human_review_reasons.append("VCF has quality warnings — review before interpreting variants.")
        state.requires_human_review = True

    if context.genome_build == GenomeBuild.UNKNOWN:
        state.add_safety_flag("Genome build unknown — variant annotation may be unreliable.")

    state.provenance.append(ProvenanceItem(
        component="vcf_input",
        tool_name="vcfpy",
        tool_version="0.14",
    ))
    state.mark_done("vcf_input")
    return state
