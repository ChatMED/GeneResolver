from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _PROJECT_ROOT / "data" / "clinvar"

_SOURCES = {
    "GRCh37": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh37/clinvar.vcf.gz",
    "GRCh38": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz",
}


def _download_one(build: str, base_url: str) -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vcf_path = _OUTPUT_DIR / f"clinvar_{build}.vcf.gz"
    tbi_path = _OUTPUT_DIR / f"clinvar_{build}.vcf.gz.tbi"

    if vcf_path.exists():
        size_mb = vcf_path.stat().st_size / (1024 * 1024)
        print(f"[{build}] already exists: {vcf_path} ({size_mb:.1f} MB) -- "
              f"delete first to refresh")
    else:
        print(f"[{build}] downloading {base_url} (~100-200 MB)...")
        try:
            urllib.request.urlretrieve(base_url, vcf_path)
        except Exception as exc:
            print(f"[{build}] FATAL: VCF download failed: {exc}", file=sys.stderr)
            sys.exit(1)
        size_mb = vcf_path.stat().st_size / (1024 * 1024)
        print(f"[{build}] saved {vcf_path} ({size_mb:.1f} MB)")

    if tbi_path.exists():
        print(f"[{build}] index already exists: {tbi_path}")
    else:
        tbi_url = base_url + ".tbi"
        print(f"[{build}] downloading index {tbi_url}...")
        try:
            urllib.request.urlretrieve(tbi_url, tbi_path)
        except Exception as exc:
            print(f"[{build}] FATAL: index download failed: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"[{build}] saved {tbi_path}")


def main() -> None:
    for build, url in _SOURCES.items():
        _download_one(build, url)


if __name__ == "__main__":
    main()
