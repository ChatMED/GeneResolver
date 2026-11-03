"""Download the NCBI ClinVar VCFs (GRCh37 + GRCh38).

    python scripts/download_clinvar.py                      # current weekly release
    python scripts/download_clinvar.py --release 20260728   # a pinned, archived release

Every download writes `<file>.source.json` (URL, release date from ##fileDate, SHA-256) next
to the VCF so the exact release used can be reported and re-fetched.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _PROJECT_ROOT / "data" / "clinvar"
_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar"


def _url(build: str, release: str | None) -> str:
    if not release:
        return f"{_BASE}/vcf_{build}/clinvar.vcf.gz"
    return f"{_BASE}/vcf_{build}/archive_2.0/{release[:4]}/clinvar_{release}.vcf.gz"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_date(path: Path) -> str | None:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for _ in range(50):
            m = re.match(r"##fileDate=(\S+)", f.readline())
            if m:
                return m.group(1)
    return None


def _write_source(vcf_path: Path, url: str | None) -> None:
    meta = {
        "url": url,
        "release_fileDate": _file_date(vcf_path),
        "sha256": _sha256(vcf_path),
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    }
    vcf_path.with_name(vcf_path.name + ".source.json").write_text(json.dumps(meta, indent=2))
    print(f"  recorded release {meta['release_fileDate']} sha256={meta['sha256'][:12]}...")


def _download_one(build: str, base_url: str) -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vcf_path = _OUTPUT_DIR / f"clinvar_{build}.vcf.gz"
    tbi_path = _OUTPUT_DIR / f"clinvar_{build}.vcf.gz.tbi"

    if vcf_path.exists():
        size_mb = vcf_path.stat().st_size / (1024 * 1024)
        print(f"[{build}] already exists: {vcf_path} ({size_mb:.1f} MB) -- "
              f"delete first to refresh")
        if not vcf_path.with_name(vcf_path.name + ".source.json").exists():
            # Source URL of an earlier download is unknown; record what can be read.
            _write_source(vcf_path, None)
    else:
        print(f"[{build}] downloading {base_url} (~100-200 MB)...")
        try:
            urllib.request.urlretrieve(base_url, vcf_path)
        except Exception as exc:
            print(f"[{build}] FATAL: VCF download failed: {exc}", file=sys.stderr)
            sys.exit(1)
        size_mb = vcf_path.stat().st_size / (1024 * 1024)
        print(f"[{build}] saved {vcf_path} ({size_mb:.1f} MB)")
        _write_source(vcf_path, base_url)

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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", help="archived weekly release, YYYYMMDD (default: current)")
    args = ap.parse_args()
    if args.release and not re.fullmatch(r"\d{8}", args.release):
        sys.exit("--release must be YYYYMMDD")
    for build in ("GRCh37", "GRCh38"):
        _download_one(build, _url(build, args.release))


if __name__ == "__main__":
    main()
