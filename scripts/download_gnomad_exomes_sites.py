from __future__ import annotations

import sys
import shutil
import urllib.request
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _PROJECT_ROOT / "data" / "gnomad_sites"

_URL = (
    "https://storage.googleapis.com/gcp-public-data--gnomad/release/2.1.1/"
    "vcf/exomes/gnomad.exomes.r2.1.1.sites.vcf.bgz"
)
_EXPECTED_SIZE_GIB = 58.81


def _reporthook(block_num: int, block_size: int, total_size: int):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100 / total_size)
        gib = downloaded / (1024 ** 3)
        print(f"\r  {pct:5.1f}%  ({gib:.2f} GiB)", end="", flush=True)


def main() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vcf_path = _OUTPUT_DIR / "gnomad.exomes.r2.1.1.sites.vcf.bgz"
    tbi_path = _OUTPUT_DIR / "gnomad.exomes.r2.1.1.sites.vcf.bgz.tbi"

    if vcf_path.exists():
        size_gib = vcf_path.stat().st_size / (1024 ** 3)
        print(f"already exists: {vcf_path} ({size_gib:.2f} GiB) -- delete first to re-download")
    else:
        free_gib = shutil.disk_usage(_OUTPUT_DIR).free / (1024 ** 3)
        if free_gib < _EXPECTED_SIZE_GIB + 5:
            print(f"FATAL: gnomAD sites needs at least {_EXPECTED_SIZE_GIB + 5:.0f} GiB free; "
                  f"only {free_gib:.1f} GiB is available.", file=sys.stderr)
            sys.exit(1)
        print(f"downloading {_URL}")
        print(f"  expected size: ~{_EXPECTED_SIZE_GIB} GiB -- this will take a while")
        try:
            urllib.request.urlretrieve(_URL, vcf_path, reporthook=_reporthook)
            print()
        except Exception as exc:
            print(f"\nFATAL: VCF download failed: {exc}", file=sys.stderr)
            sys.exit(1)
        size_gib = vcf_path.stat().st_size / (1024 ** 3)
        print(f"saved {vcf_path} ({size_gib:.2f} GiB)")

    if tbi_path.exists():
        print(f"index already exists: {tbi_path}")
    else:
        tbi_url = _URL + ".tbi"
        print(f"downloading index {tbi_url}...")
        try:
            urllib.request.urlretrieve(tbi_url, tbi_path)
        except Exception as exc:
            print(f"FATAL: index download failed: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"saved {tbi_path}")

    print("\nNext: run scripts/build_gnomad_an_index.py to extract the slim "
          "popmax AF/AN index gnomad_popmax_local.py actually queries.")


if __name__ == "__main__":
    main()
