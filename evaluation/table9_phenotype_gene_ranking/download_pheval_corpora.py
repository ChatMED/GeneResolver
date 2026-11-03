"""
Fetches the PhEval paper's Phenopacket Store corpora (stage 4: phenotype-only gene
ranking) from the archived Zenodo record and lays them out where
`eval_phenotype_gene_ranking.py` expects them:

    external/pheval_corpora/corpora/phenopacket_store_0.1.11_phenotypes/phenopackets/
    external/pheval_corpora/corpora/phenopacket_store_0.1.11_variants/phenopackets/

Source: https://zenodo.org/records/14679713 ("pheval-paper-corpora.tar.gz", ~5.7 MB --
this is the small phenopacket-only archive on that record, not the 5.6 GB results
archive alongside it).

    python -m evaluation.table9_phenotype_gene_ranking.download_pheval_corpora
"""
from __future__ import annotations

import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEST_ROOT = _PROJECT_ROOT / "external" / "pheval_corpora"
_DEST_CORPORA = _DEST_ROOT / "corpora"
_MARKER = _DEST_CORPORA / "phenopacket_store_0.1.11_phenotypes" / "phenopackets"

SOURCE_URL = "https://zenodo.org/records/14679713/files/pheval-paper-corpora.tar.gz?download=1"


def _find_corpora_dir(staging: Path) -> Path | None:
    """The archive's internal layout isn't pinned by us -- search for the
    `corpora/` directory it contains rather than assuming a fixed depth."""
    if (staging / "corpora").is_dir():
        return staging / "corpora"
    for candidate in staging.rglob("corpora"):
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    return None


def main() -> None:
    print("=== PhEval Phenopacket Store corpora (stage 4: phenotype-only gene ranking) ===")
    if _MARKER.exists():
        print(f"  [ok] already present at {_MARKER}")
        print("  Delete external/pheval_corpora/ first if you want to refresh.")
        return

    _DEST_ROOT.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=_DEST_ROOT) as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "pheval-paper-corpora.tar.gz"

        print(f"  Downloading {SOURCE_URL} (~5.7 MB) ...")
        try:
            urllib.request.urlretrieve(SOURCE_URL, archive)
        except Exception as exc:
            print(f"  [!] download failed: {exc}", file=sys.stderr)
            print("  Fetch manually from https://zenodo.org/records/14679713 "
                  "(the corpora archive, not the results archive).", file=sys.stderr)
            sys.exit(1)

        print(f"  Extracting {archive.name} ...")
        staging = tmp_path / "extracted"
        staging.mkdir()
        try:
            with tarfile.open(archive) as tf:
                tf.extractall(staging)
        except Exception as exc:
            print(f"  [!] extraction failed: {exc}", file=sys.stderr)
            sys.exit(1)

        corpora_dir = _find_corpora_dir(staging)
        if corpora_dir is None:
            print(f"  [!] extraction succeeded but no 'corpora/' directory was found "
                  f"under {staging} -- the archive layout may have changed. Move the "
                  f"phenopacket_store_* directories into {_DEST_CORPORA} manually.",
                  file=sys.stderr)
            sys.exit(1)

        if _DEST_CORPORA.exists():
            shutil.rmtree(_DEST_CORPORA)
        shutil.move(str(corpora_dir), str(_DEST_CORPORA))

    if not _MARKER.exists():
        print(f"  [!] moved corpora but expected path is still missing: {_MARKER}",
              file=sys.stderr)
        sys.exit(1)

    n_cases = len(list(_MARKER.glob("*.json")))
    print(f"  [ok] corpora ready at {_DEST_CORPORA} ({n_cases} phenotype-only cases)")


if __name__ == "__main__":
    main()
