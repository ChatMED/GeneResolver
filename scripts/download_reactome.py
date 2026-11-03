from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = _PROJECT_ROOT / "data" / "reactome" / "NCBI2Reactome_All_Levels.txt"

SOURCE_URL = "https://download.reactome.org/97/NCBI2Reactome_All_Levels.txt"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
        print(f"Already exists: {OUTPUT_PATH} ({size_mb:.1f} MB)")
        print("Delete the file first if you want to refresh.")
        return

    print(f"Downloading {SOURCE_URL} (~93 MB, Reactome release 97)...")
    try:
        urllib.request.urlretrieve(SOURCE_URL, OUTPUT_PATH)
    except Exception as exc:
        print(f"FATAL: download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    with OUTPUT_PATH.open(encoding="utf-8") as f:
        lines = sum(1 for _ in f)
    print(f"Saved {OUTPUT_PATH} ({size_mb:.1f} MB, {lines:,} lines)")


if __name__ == "__main__":
    main()
