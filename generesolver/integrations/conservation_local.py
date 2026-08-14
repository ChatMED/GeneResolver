from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

from generesolver.config import DATA_DIR

logger = logging.getLogger(__name__)

_PHYLOP_PATH = DATA_DIR / "conservation" / "hg19.100way.phyloP100way.bw"
_DOCKER_IMAGE = "quay.io/biocontainers/pybigwig:0.3.25--py311h4721021_1"

_QUERY_SCRIPT = """
import json, sys
import pyBigWig
positions = json.load(open("/work/positions.json"))
bw = pyBigWig.open("/data/hg19.100way.phyloP100way.bw")
out = {}
for chrom, pos in positions:
    key = f"{chrom}:{pos}"
    try:
        vals = bw.values(f"chr{chrom}" if not chrom.startswith("chr") else chrom, pos - 1, pos)
        out[key] = vals[0] if vals and vals[0] == vals[0] else None  # NaN check: x==x is False for NaN
    except RuntimeError:
        out[key] = None
bw.close()
json.dump(out, open("/work/scores.json", "w"))
"""


def is_available() -> bool:
    return _PHYLOP_PATH.exists()


def get_conservation_batch(positions: list[tuple[str, int]]) -> dict[str, float | None]:
    if not positions or not is_available():
        return {}

    work = Path(tempfile.mkdtemp(prefix="generesolver_phylop_"))
    (work / "positions.json").write_text(
        json.dumps([[str(c).lstrip("chr"), p] for c, p in positions]), encoding="utf-8"
    )
    (work / "query.py").write_text(_QUERY_SCRIPT, encoding="utf-8")

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{DATA_DIR.resolve() / 'conservation'}:/data",
        "-v", f"{work}:/work",
        _DOCKER_IMAGE, "python3", "/work/query.py",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True,
                        env={"MSYS_NO_PATHCONV": "1"} | dict(__import__("os").environ))
    except subprocess.CalledProcessError as exc:
        logger.error("Local phyloP lookup failed: %s", (exc.stderr or b"").decode("utf-8", "ignore")[:600])
        return {}

    scores_path = work / "scores.json"
    if not scores_path.exists():
        return {}
    return json.loads(scores_path.read_text(encoding="utf-8"))
