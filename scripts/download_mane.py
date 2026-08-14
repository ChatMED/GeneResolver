from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "data" / "mane"

_INDEX_URL = "https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/"
_FILENAME_RE = re.compile(r'href="(MANE\.GRCh38\.v[\d.]+\.summary\.txt\.gz)"')


def _discover_current_filename() -> str:
    with urllib.request.urlopen(_INDEX_URL, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    m = _FILENAME_RE.search(html)
    if not m:
        raise RuntimeError(
            f"Could not find a MANE.GRCh38.v*.summary.txt.gz link at {_INDEX_URL} "
            "-- NCBI may have changed their directory layout or filename convention."
        )
    return m.group(1)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing = list(OUTPUT_DIR.glob("MANE.*.summary.txt.gz"))
    if existing:
        size_mb = existing[0].stat().st_size / (1024 * 1024)
        print(f"Already exists: {existing[0]} ({size_mb:.2f} MB)")
        print("Delete the file first if you want to refresh to a newer MANE release.")
        return

    print(f"Discovering current MANE release from {_INDEX_URL} ...")
    try:
        filename = _discover_current_filename()
    except Exception as exc:
        print(f"FATAL: could not discover current MANE filename: {exc}", file=sys.stderr)
        sys.exit(1)

    url = _INDEX_URL + filename
    out_path = OUTPUT_DIR / filename
    print(f"Downloading {url} (~1.1 MB)...")
    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception as exc:
        print(f"FATAL: download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
