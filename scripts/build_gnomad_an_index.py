"""Build the gnomAD v2.1.1 exome popmax AF/AN index used for BA1/PM2 coverage checks.

Writes data/gnomad_popmax/gnomad_an_popmax.hg37.tsv.gz with one row per variant that has
both AF_popmax and AN_popmax (the frequency and allele number of the same population).

By default the 59 GiB gnomAD sites VCF is streamed from Google Cloud and never stored;
only the slim index (a few hundred MB) is written. If you already downloaded the VCF with
scripts/download_gnomad_exomes_sites.py, pass --local to read it from disk instead.

    python scripts/build_gnomad_an_index.py            # stream (several hours)
    python scripts/build_gnomad_an_index.py --local    # read data/gnomad_sites/...vcf.bgz
"""
from __future__ import annotations

import argparse
import gzip
import sys
import time
import urllib.request
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_URL = ("https://storage.googleapis.com/gcp-public-data--gnomad/release/2.1.1/"
        "vcf/exomes/gnomad.exomes.r2.1.1.sites.vcf.bgz")
_LOCAL_VCF = _PROJECT_ROOT / "data" / "gnomad_sites" / "gnomad.exomes.r2.1.1.sites.vcf.bgz"
_OUTPUT = _PROJECT_ROOT / "data" / "gnomad_popmax" / "gnomad_an_popmax.hg37.tsv.gz"
_PROGRESS_EVERY = 2_000_000


def _field(info: str, key: str) -> str | None:
    tag = key + "="
    for kv in info.split(";"):
        if kv.startswith(tag):
            return kv[len(tag):]
    return None


def _open_source(local: bool):
    if local:
        if not _LOCAL_VCF.exists():
            sys.exit(f"FATAL: {_LOCAL_VCF} not found. Run scripts/download_gnomad_exomes_sites.py "
                     "or drop --local to stream.")
        print(f"reading {_LOCAL_VCF}")
        return gzip.open(_LOCAL_VCF, "rt", encoding="utf-8", errors="replace")
    print(f"streaming {_URL}")
    response = urllib.request.urlopen(_URL)  # noqa: S310 - fixed public gnomAD URL
    # BGZF is multi-member gzip, which GzipFile decodes transparently.
    return gzip.open(response, "rt", encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true",
                        help="read the already-downloaded sites VCF instead of streaming")
    args = parser.parse_args()

    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = _OUTPUT.with_suffix(".partial")
    t0, n_read, n_written = time.time(), 0, 0
    with _open_source(args.local) as fin, gzip.open(tmp, "wt", encoding="utf-8") as fout:
        fout.write("#chrom\tpos\tref\talt\taf_popmax\tan_popmax\tpopmax_population\n")
        for line in fin:
            if line.startswith("#"):
                continue
            n_read += 1
            if n_read % _PROGRESS_EVERY == 0:
                print(f"  {n_read:,} read, {n_written:,} written "
                      f"({(time.time() - t0) / 60:.0f} min)", file=sys.stderr)
            cols = line.split("\t", 8)
            if len(cols) < 8:
                continue
            af_raw, an_raw = _field(cols[7], "AF_popmax"), _field(cols[7], "AN_popmax")
            if not af_raw or af_raw == "." or not an_raw or an_raw == ".":
                continue
            try:
                af, an = float(af_raw), int(an_raw)
            except ValueError:
                continue
            pop = _field(cols[7], "popmax") or ""
            fout.write(f"{cols[0].removeprefix('chr')}\t{cols[1]}\t{cols[3]}\t{cols[4]}\t"
                       f"{af}\t{an}\t{pop}\n")
            n_written += 1

    if n_written == 0:
        tmp.unlink(missing_ok=True)
        sys.exit("FATAL: 0 records written; check the AF_popmax/AN_popmax field names.")
    tmp.replace(_OUTPUT)
    print(f"done in {(time.time() - t0) / 60:.1f} min: {n_read:,} read, "
          f"{n_written:,} written to {_OUTPUT}")


if __name__ == "__main__":
    main()
