"""
Clones the `monarch_pheval` sibling repo (stage 5: genome-level +VCF gene ranking)
next to this repo, per evaluation/README.md's "Stage 5 data setup". It carries the
LIRICAL phenopacket corpus and the public hg19 template exome that
`eval_genome_vcf_ranking.py` spikes candidate variants into:

    ../monarch_pheval/corpora/lirical/default/phenopackets/
    ../monarch_pheval/testdata/template_vcf/template_exome_hg19.vcf.gz

Source: https://github.com/monarch-initiative/monarch_pheval

    python -m evaluation.stage5_genome_vcf_ranking.download_monarch_pheval
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEST = _PROJECT_ROOT.parent / "monarch_pheval"
_MARKER = _DEST / "corpora" / "lirical" / "default" / "phenopackets"

SOURCE_REPO = "https://github.com/monarch-initiative/monarch_pheval.git"


def main() -> None:
    print("=== monarch_pheval sibling checkout (stage 5: +VCF gene ranking) ===")
    if _MARKER.exists():
        print(f"  [ok] already present at {_MARKER}")
        print(f"  `git -C {_DEST} pull` first if you want to refresh.")
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

    print(f"  git clone {SOURCE_REPO} {_DEST}")
    try:
        subprocess.run(["git", "clone", "--depth", "1", SOURCE_REPO, str(_DEST)], check=True)
    except Exception as exc:
        print(f"  [!] clone failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if not _MARKER.exists():
        print(f"  [!] cloned but expected path is missing: {_MARKER} -- repo layout may "
              f"have changed.", file=sys.stderr)
        sys.exit(1)

    print(f"  [ok] monarch_pheval ready at {_DEST}")


if __name__ == "__main__":
    main()
