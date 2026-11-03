"""Transcript structure from GENCODE v19 (the gene set of the VEP 116 GRCh37 cache).

Used by the ACMG engine's PVS1 rules: CDS coordinates, exon boundaries, the last exon-exon
junction (NMD prediction) and the coding length of an exon that a canonical splice variant
would skip. Built once from data/gencode/gencode.v19.annotation.gtf.gz by
scripts/download_gencode_gtf.py; only protein-coding transcripts are kept.
"""
from __future__ import annotations

import gzip
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache

from generesolver.config import DATA_DIR

logger = logging.getLogger(__name__)

GTF_PATH = DATA_DIR / "gencode" / "gencode.v19.annotation.gtf.gz"
INDEX_PATH = DATA_DIR / "gencode" / "gencode_v19_transcripts.json.gz"
SOURCE_VERSION = "GENCODE v19 (GRCh37)"

_ATTR = re.compile(r'(\S+) "?([^";]*)"?;')   # GENCODE writes exon_number unquoted


def _strip(tid: str) -> str:
    return tid.split(".", 1)[0]


@dataclass(frozen=True)
class Transcript:
    transcript_id: str
    gene: str
    chrom: str
    strand: str
    exons: tuple[tuple[int, int], ...]   # genomic, 1-based inclusive, in transcript order
    cds: tuple[tuple[int, int], ...]     # coding part of each exon, same order; (0, 0) if none
    tags: tuple[str, ...]

    # -- coding coordinates -------------------------------------------------------------
    @property
    def cds_length(self) -> int:
        return sum(e - s + 1 for s, e in self.cds if s)

    @property
    def protein_length(self) -> int:
        return self.cds_length // 3   # GENCODE CDS features exclude the stop codon

    def cds_offset(self, pos: int) -> int | None:
        """1-based CDS coordinate of a genomic position inside a coding exon."""
        done = 0
        for s, e in self.cds:
            if not s:
                continue
            if s <= pos <= e:
                return done + (e - pos + 1 if self.strand == "-" else pos - s + 1)
            done += e - s + 1
        return None

    def cds_end_of_exon(self, index: int) -> int:
        """CDS coordinate of the last coding base of exon `index` (0-based, transcript order)."""
        return sum(e - s + 1 for s, e in self.cds[: index + 1] if s)

    def exon_index(self, pos: int) -> int | None:
        for i, (s, e) in enumerate(self.exons):
            if s <= pos <= e:
                return i
        return None

    def last_junction_cds(self) -> int | None:
        """CDS coordinate of the last base before the last exon-exon junction, or None for a
        single-exon transcript."""
        if len(self.exons) < 2:
            return None
        return self.cds_end_of_exon(len(self.exons) - 2)

    def nmd_predicted(self, cds_pos: int) -> bool:
        """ClinGen PVS1 (Abou Tayoun 2018): NMD is predicted when the premature stop lies more
        than 50 nt upstream of the last exon-exon junction."""
        junction = self.last_junction_cds()
        return junction is not None and cds_pos < junction - 50

    def skipped_exon_for_splice_site(self, pos: int) -> tuple[int, str] | None:
        """For a variant at intronic position ±1/±2 of a canonical splice site, return the exon
        index that would be skipped and 'donor'/'acceptor'."""
        for i, (s, e) in enumerate(self.exons):
            five, three = (e, s) if self.strand == "-" else (s, e)   # exon 5' and 3' ends
            step = -1 if self.strand == "-" else 1
            if i < len(self.exons) - 1 and pos in (three + step, three + 2 * step):
                return i, "donor"
            if i > 0 and pos in (five - step, five - 2 * step):
                return i, "acceptor"
        return None

    def coding_length_of_exon(self, index: int) -> int:
        s, e = self.cds[index]
        return e - s + 1 if s else 0

    def protein_range_of_exon(self, index: int) -> tuple[int, int] | None:
        length = self.coding_length_of_exon(index)
        if not length:
            return None
        end = self.cds_end_of_exon(index)
        return (end - length) // 3 + 1, (end + 2) // 3

    @property
    def has_ccds(self) -> bool:
        return "CCDS" in self.tags

    @property
    def is_basic(self) -> bool:
        return "basic" in self.tags


def is_available() -> bool:
    return INDEX_PATH.exists() or GTF_PATH.exists()


def build_index(force: bool = False) -> int:
    if INDEX_PATH.exists() and not force:
        return len(_load_raw())
    if not GTF_PATH.exists():
        raise FileNotFoundError(f"{GTF_PATH} missing: run scripts/download_gencode_gtf.py")
    tx: dict[str, dict] = {}
    with gzip.open(GTF_PATH, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] not in ("exon", "CDS"):
                continue
            attrs = dict(_ATTR.findall(f[8]))
            if attrs.get("transcript_type") != "protein_coding":
                continue
            tid = _strip(attrs["transcript_id"])
            rec = tx.setdefault(tid, {"gene": attrs.get("gene_name"), "chrom": f[0].replace("chr", ""),
                                      "strand": f[6], "exons": {}, "cds": {}, "tags": set()})
            rec["tags"].update(t for k, t in re.findall(r'(tag) "([^"]*)"', f[8]))
            if "ccdsid" in attrs:
                rec["tags"].add("CCDS")
            n = int(attrs["exon_number"])
            (rec["exons"] if f[2] == "exon" else rec["cds"])[n] = (int(f[3]), int(f[4]))
    out = {}
    for tid, r in tx.items():
        order = sorted(r["exons"])
        out[tid] = {"g": r["gene"], "c": r["chrom"], "s": r["strand"],
                    "e": [r["exons"][n] for n in order],
                    "d": [r["cds"].get(n, (0, 0)) for n in order],
                    "t": sorted(r["tags"])}
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(INDEX_PATH, "wt", encoding="utf-8") as fh:
        json.dump(out, fh)
    _load_raw.cache_clear()
    return len(out)


@lru_cache(maxsize=1)
def _load_raw() -> dict:
    if not INDEX_PATH.exists():
        if not GTF_PATH.exists():
            logger.warning("GENCODE v19 transcript index missing (run scripts/download_gencode_gtf.py); "
                           "PVS1 falls back to the VEP exon field only.")
            return {}
        build_index()
    with gzip.open(INDEX_PATH, "rt", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _by_gene() -> dict[str, list[str]]:
    genes: dict[str, list[str]] = {}
    for tid, r in _load_raw().items():
        genes.setdefault(r["g"], []).append(tid)
    return genes


def get(transcript_id: str | None) -> Transcript | None:
    if not transcript_id:
        return None
    r = _load_raw().get(_strip(transcript_id))
    if not r:
        return None
    return Transcript(_strip(transcript_id), r["g"], r["c"], r["s"],
                      tuple(map(tuple, r["e"])), tuple(map(tuple, r["d"])), tuple(r["t"]))


def transcripts_for_gene(gene: str | None) -> list[Transcript]:
    return [t for t in (get(tid) for tid in _by_gene().get(gene or "", [])) if t]
