from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "data" / "orphanet"

SOURCES: dict[str, str] = {
    "en_product1.xml":      "https://www.orphadata.com/data/xml/en_product1.xml",
    "en_product6.xml":      "https://www.orphadata.com/data/xml/en_product6.xml",
    "en_product9_prev.xml": "https://www.orphadata.com/data/xml/en_product9_prev.xml",
    "en_product4.xml":      "https://www.orphadata.com/data/xml/en_product4.xml",
}


def _fetch(filename: str, url: str) -> bool:
    dest = OUTPUT_DIR / filename
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  skip: {filename} already exists ({size_mb:.1f} MB) — delete to refresh")
        return True
    print(f"  GET  {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        print(f"  FAIL: {exc}", file=sys.stderr)
        return False
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  done: {dest.name} ({size_mb:.1f} MB)")
    return True


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Orphadata XML products into {OUTPUT_DIR}")
    failures: list[str] = []
    for filename, url in SOURCES.items():
        ok = _fetch(filename, url)
        if not ok:
            failures.append(filename)

    if failures:
        print(
            f"\n{len(failures)} file(s) failed: {failures}.\n"
            "If Orphadata is unreachable, browse https://www.orphadata.com/ "
            "and place the XML files manually in data/orphanet/.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("All Orphadata files present.")


if __name__ == "__main__":
    main()
