"""Download the GENCODE v19 gene annotation (GRCh37), the gene set of the local VEP 116 GRCh37
cache (its info.txt reports source_gencode GENCODE 19), and build the compact transcript index
used by the ACMG engine's PVS1 rules (exon structure, CDS, NMD position, exon skipping).

    python scripts/download_gencode_gtf.py           # ~38 MB download, index built once

Output: data/gencode/gencode.v19.annotation.gtf.gz and data/gencode/gencode_v19_transcripts.json.gz
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_URL = ("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/"
        "gencode.v19.annotation.gtf.gz")
_OUT = _ROOT / "data" / "gencode" / "gencode.v19.annotation.gtf.gz"


def main() -> None:
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    if not _OUT.exists():
        print(f"downloading {_URL}")
        tmp = _OUT.with_suffix(".part")
        try:
            urllib.request.urlretrieve(_URL, tmp)  # noqa: S310 - fixed URL
        except Exception as exc:
            sys.exit(f"FATAL: download failed: {exc}")
        tmp.replace(_OUT)
    digest = hashlib.sha256(_OUT.read_bytes()).hexdigest()
    print(f"{_OUT} ({_OUT.stat().st_size / 1e6:.1f} MB, sha256 {digest[:16]})")

    from generesolver.integrations import gencode_transcripts
    n = gencode_transcripts.build_index(force=True)
    print(f"index: {n} protein-coding transcripts -> {gencode_transcripts.INDEX_PATH}")


if __name__ == "__main__":
    main()
