from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = _PROJECT_ROOT / "data" / "conservation" / "hg19.100way.phyloP100way.bw"

SOURCE_URL = (
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/phyloP100way/"
    "hg19.100way.phyloP100way.bw"
)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        size_gb = OUTPUT_PATH.stat().st_size / (1024 ** 3)
        print(f"Already exists: {OUTPUT_PATH} ({size_gb:.2f} GB)")
        print("Delete the file first if you want to refresh.")
        return

    print(f"Downloading {SOURCE_URL} (~8.9 GB -- this will take a while)...")
    try:
        urllib.request.urlretrieve(SOURCE_URL, OUTPUT_PATH)
    except Exception as exc:
        print(f"FATAL: download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    size_gb = OUTPUT_PATH.stat().st_size / (1024 ** 3)
    print(f"Saved {OUTPUT_PATH} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
