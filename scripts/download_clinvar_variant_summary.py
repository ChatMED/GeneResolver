from __future__ import annotations

import sys
import gzip
import urllib.request
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = _PROJECT_ROOT / "data" / "clinvar" / "variant_summary.txt.gz"

SOURCE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        try:
            with gzip.open(OUTPUT_PATH, "rb") as handle:
                while handle.read(1024 * 1024):
                    pass
        except (EOFError, OSError) as exc:
            print(f"Incomplete ClinVar download found ({exc}); will replace it.")
        else:
            size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
            print(f"Already exists: {OUTPUT_PATH} ({size_mb:.1f} MB)")
            return

    print(f"Downloading {SOURCE_URL} (~421 MB)...")
    try:
        partial = OUTPUT_PATH.with_name(OUTPUT_PATH.name + ".part")
        urllib.request.urlretrieve(SOURCE_URL, partial)
        with gzip.open(partial, "rb") as handle:
            while handle.read(1024 * 1024):
                pass
        partial.replace(OUTPUT_PATH)
    except Exception as exc:
        print(f"FATAL: download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
