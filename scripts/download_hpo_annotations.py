from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "data" / "hpo_a"
_FILES = ("phenotype_to_genes.txt", "phenotype.hpoa")
_SOURCE_ROOT = "https://purl.obolibrary.org/obo/hp/hpoa/"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename in _FILES:
        path = OUTPUT_DIR / filename
        if path.exists() and path.stat().st_size > 1000:
            print(f"Already exists: {path} ({path.stat().st_size / 1024**2:.1f} MB)")
            continue
        url = _SOURCE_ROOT + filename
        partial = path.with_name(filename + ".part")
        print(f"Downloading {url}...")
        try:
            urllib.request.urlretrieve(url, partial)
            if partial.stat().st_size <= 1000:
                raise ValueError("HPO annotation file is unexpectedly small")
            partial.replace(path)
        except Exception as exc:
            print(f"FATAL: download failed: {exc}", file=sys.stderr)
            print("Try downloading manually from https://hpo.jax.org/data/annotations", file=sys.stderr)
            sys.exit(1)
        print(f"Saved {path} ({path.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
