from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _PROJECT_ROOT / "data" / "vep_cache"
_MARKER = CACHE_DIR / "homo_sapiens" / "116_GRCh37"
_ARCHIVE = CACHE_DIR / "homo_sapiens_vep_116_GRCh37.tar.gz"

SOURCE_URL = (
    "https://ftp.ensembl.org/pub/release-116/variation/indexed_vep_cache/"
    "homo_sapiens_vep_116_GRCh37.tar.gz"
)
DOCKER_IMAGE = os.environ.get("GENERESOLVER_VEP_DOCKER_IMAGE", "ensemblorg/ensembl-vep:release_116.0")


def _pull_docker_image() -> bool:
    if shutil.which("docker") is None:
        print("  [!] docker not found on PATH -- install/start Docker Desktop first, "
              "then re-run this script.", file=sys.stderr)
        return False
    print(f"  docker pull {DOCKER_IMAGE} ...")
    result = subprocess.run(["docker", "pull", DOCKER_IMAGE])
    return result.returncode == 0


def _download_and_extract() -> bool:
    if _MARKER.is_dir() and any(_MARKER.iterdir()):
        print(f"  [ok] cache already present at {_MARKER}")
        return True

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(CACHE_DIR).free / (1024 ** 3)
    if free_gib < 50:
        print(f"  [!] VEP cache setup needs at least 50 GiB free for the 23 GiB archive "
              f"and extraction; only {free_gib:.1f} GiB is available.", file=sys.stderr)
        return False
    if not _ARCHIVE.exists():
        print(f"  Downloading {SOURCE_URL}\n  (~23 GB -- this will take a while)")
        try:
            urllib.request.urlretrieve(SOURCE_URL, _ARCHIVE)
        except Exception as exc:
            print(f"  [!] download failed: {exc}", file=sys.stderr)
            return False
    else:
        print(f"  [ok] archive already downloaded at {_ARCHIVE}")

    print(f"  Extracting {_ARCHIVE.name} -> {CACHE_DIR} ...")
    try:
        with tarfile.open(_ARCHIVE) as tf:
            tf.extractall(CACHE_DIR)
    except Exception as exc:
        print(f"  [!] extraction failed: {exc}", file=sys.stderr)
        return False

    if not _MARKER.is_dir() or not any(_MARKER.iterdir()):
        print(f"  [!] extraction finished but expected path missing: {_MARKER}",
              file=sys.stderr)
        return False

    _ARCHIVE.unlink(missing_ok=True)
    print(f"  [ok] cache ready at {_MARKER}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-docker-pull", action="store_true",
                     help="only fetch/extract the cache, skip `docker pull`")
    args = ap.parse_args()

    print("=== VEP local cache + Docker image setup (GRCh37) ===")
    ok_docker = args.skip_docker_pull or _pull_docker_image()
    ok_cache = _download_and_extract() if ok_docker else False

    if ok_docker and ok_cache:
        print("\nDone. GENERESOLVER_LOCAL_MODE=1 will now use this local Docker VEP cache "
              "instead of the Ensembl REST API.")
    else:
        print("\nSome steps failed -- see messages above; re-run to retry.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
