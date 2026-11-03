"""Download the ClinGen haploinsufficiency gene track (GRCh37 BED).

Used by generesolver/integrations/clingen_haploinsufficiency.py: a gene with HI score 3
(sufficient evidence) is treated as having a loss-of-function disease mechanism for PVS1
when Orphanet does not state the mechanism.

    python scripts/download_clingen_haploinsufficiency.py
"""
from __future__ import annotations

import datetime as dt
import sys
import urllib.request
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_URL = "https://ftp.clinicalgenome.org/ClinGen_haploinsufficiency_gene_GRCh37.bed"
_OUTPUT = _PROJECT_ROOT / "data" / "clingen" / "hi_check.bed"


def main() -> None:
    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {_URL}")
    try:
        with urllib.request.urlopen(_URL, timeout=120) as r:  # noqa: S310 - fixed URL
            data = r.read()
    except Exception as exc:
        sys.exit(f"FATAL: download failed: {exc}")
    text = data.decode("utf-8", "replace")
    rows = [l for l in text.splitlines() if l and not l.startswith(("track", "#"))]
    if not rows or len(rows[0].split("\t")) < 5:
        sys.exit("FATAL: unexpected file format (expected chrom, start, end, gene, score).")
    _OUTPUT.write_bytes(data)
    n3 = sum(1 for l in rows if l.split("\t")[4] == "3")
    print(f"saved {_OUTPUT}: {len(rows)} genes, {n3} with score 3 "
          f"(downloaded {dt.date.today().isoformat()})")


if __name__ == "__main__":
    main()
