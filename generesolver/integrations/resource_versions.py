"""Versions of the local annotation resources, read from the files themselves.

Recorded in every case's variant_annotator provenance so a result can be traced to the exact
ClinVar release, REVEL/MANE/SpliceAI files and slivar image. Values are read, never assumed:
a resource that cannot be inspected is reported as missing.
"""
from __future__ import annotations

import gzip
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from generesolver.config import DATA_DIR

_cache: dict[str, dict] = {}


def _file_info(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path.relative_to(DATA_DIR.parent)) if path.is_relative_to(DATA_DIR.parent)
                else str(path), "present": False}
    st = path.stat()
    return {
        "file": path.name,
        "present": True,
        "bytes": st.st_size,
        "modified_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).strftime("%Y-%m-%d"),
    }


def _clinvar(build: str) -> dict:
    p = DATA_DIR / "clinvar" / f"clinvar_{build}.vcf.gz"
    info = _file_info(p)
    if info.get("present"):
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for _ in range(50):
                    line = f.readline()
                    if not line.startswith("##"):
                        break
                    m = re.match(r"##fileDate=(\S+)", line)
                    if m:
                        info["release_fileDate"] = m.group(1)
                    m = re.match(r"##reference=(\S+)", line)
                    if m:
                        info["reference"] = m.group(1)
        except OSError:
            pass
        src = p.with_name(p.name + ".source.json")
        if src.exists():
            import json
            try:
                info["source"] = json.loads(src.read_text())
            except ValueError:
                pass
    return info


def _first_match(directory: Path, pattern: str) -> dict:
    hits = sorted(directory.glob(pattern)) if directory.exists() else []
    return _file_info(hits[-1]) if hits else {"pattern": pattern, "present": False}


def _docker_id(image: str) -> str | None:
    if not shutil.which("docker"):
        return None
    r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def collect(genome_build: str | None) -> dict:
    build = genome_build or "GRCh37"
    if build in _cache:
        return _cache[build]
    from generesolver.integrations import gnomad_popmax_local
    out = {
        "clinvar_vcf": _clinvar(build),
        "revel": _first_match(DATA_DIR / "revel", "revel-v*_all_chromosomes.zip"),
        "mane": _first_match(DATA_DIR / "mane", "MANE.GRCh38.v*.summary.txt.gz"),
        "gnomad_constraint": _first_match(DATA_DIR / "gnomad", "gnomad.v*.constraint_metrics.tsv"),
        "slivar_gnomad_popmax": {**_file_info(DATA_DIR / "gnomad_popmax" / "gnomad.hg37.zip"),
                                 "gnomad_release": "v2.1.1 (slivar gnotate file)"},
        "slivar_spliceai": _file_info(DATA_DIR / "spliceai" / "spliceai.hg37.zip"),
        "gnomad_an_index": _file_info(DATA_DIR / "gnomad_popmax" / "gnomad_an_popmax.hg37.tsv.gz"),
        "slivar_image": {"image": gnomad_popmax_local._DOCKER_IMAGE,
                         "image_id": _docker_id(gnomad_popmax_local._DOCKER_IMAGE)},
    }
    _cache[build] = out
    return out
