"""
Clones `monarch_pheval` into this repo's external/ directory. It carries the
LIRICAL phenopacket corpus and the public hg19 template exome that
`eval_genome_vcf_ranking.py` spikes candidate variants into:

    external/monarch_pheval/corpora/lirical/default/phenopackets/
    external/monarch_pheval/testdata/template_vcf/template_exome_hg19.vcf.gz

Source: https://github.com/monarch-initiative/monarch_pheval

    python -m evaluation.table10_vcf_ablation.download_monarch_pheval
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LEGACY_DEST = _PROJECT_ROOT.parent / "monarch_pheval"
_DEST = _LEGACY_DEST if _LEGACY_DEST.exists() else _PROJECT_ROOT / "external" / "monarch_pheval"
_MARKER = _DEST / "corpora" / "lirical" / "default" / "phenopackets"
_TEMPLATE = _DEST / "testdata" / "template_vcf" / "template_exome_hg19.vcf.gz"

SOURCE_REPO = "https://github.com/monarch-initiative/monarch_pheval.git"
SOURCE_COMMIT = "a99d7d9556bc362c3c4bf929b24b69100bd609fc"


def main() -> None:
    print("=== monarch_pheval sibling checkout (stage 5: +VCF gene ranking) ===")
    if _MARKER.exists() and _TEMPLATE.exists():
        commit = subprocess.check_output(
            ["git", "-c", f"safe.directory={_DEST.as_posix()}", "-C", str(_DEST), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if commit != SOURCE_COMMIT:
            raise SystemExit(f"LIRICAL checkout is at {commit}, expected {SOURCE_COMMIT}")
        print(f"  [ok] already present at {_MARKER} ({len(list(_MARKER.glob('*.json')))} cases)")
        print(f"  pinned commit: {commit}")
        return

    if shutil.which("git") is None:
        print("  [!] git not found on PATH -- install git or clone manually:", file=sys.stderr)
        print(f"      git clone {SOURCE_REPO} {_DEST}", file=sys.stderr)
        sys.exit(1)

    if _DEST.exists():
        print(f"  [!] {_DEST} already exists but is missing the expected phenopacket "
              f"path ({_MARKER}) -- check it manually rather than re-cloning over it.",
              file=sys.stderr)
        sys.exit(1)

    print(f"  fetching {SOURCE_REPO} at {SOURCE_COMMIT}")
    try:
        subprocess.run(["git", "init", str(_DEST)], check=True)
        subprocess.run(["git", "-C", str(_DEST), "remote", "add", "origin", SOURCE_REPO], check=True)
        subprocess.run(["git", "-C", str(_DEST), "fetch", "--depth", "1", "origin", SOURCE_COMMIT], check=True)
        subprocess.run(["git", "-C", str(_DEST), "checkout", "--detach", "FETCH_HEAD"], check=True)
    except Exception as exc:
        print(f"  [!] clone failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if not _MARKER.exists() or not _TEMPLATE.exists():
        print(f"  [!] cloned but expected paths are missing: {_MARKER}, {_TEMPLATE} -- repo layout may "
              f"have changed.", file=sys.stderr)
        sys.exit(1)

    print(f"  [ok] monarch_pheval ready at {_DEST}")


if __name__ == "__main__":
    main()
