from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_SUBPROCESS_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def _pheval_utils_executable() -> str:
    candidates = [
        ROOT / ".venv" / "Scripts" / "pheval-utils.exe",
        ROOT / ".venv" / "Scripts" / "pheval-utils",
        ROOT / ".venv" / "bin" / "pheval-utils",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    found = shutil.which("pheval-utils")
    if found:
        return found
    raise RuntimeError(
        "pheval-utils not found (checked .venv/Scripts, .venv/bin, and PATH). "
        "Install it with `pip install pheval` in this project's .venv, then retry."
    )


def run_benchmark(run_yaml: Path, out_dir: Path) -> dict:
    exe = _pheval_utils_executable()
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.rglob("*.duckdb"):
        stale.unlink()
    subprocess.run([exe, "benchmark", "-r", str(run_yaml), "-o", str(out_dir)],
                   env=_SUBPROCESS_ENV, check=True)

    duckdb_files = list(out_dir.rglob("*.duckdb"))
    if not duckdb_files:
        raise RuntimeError(
            f"pheval-utils benchmark ran but produced no .duckdb file under {out_dir}")

    try:
        import duckdb
    except ImportError:
        raise RuntimeError(
            "the `duckdb` package is required to read pheval-utils' output "
            "(`pip install duckdb`)")

    con = duckdb.connect(str(duckdb_files[0]), read_only=True)
    try:
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        summary_table = next((t for t in tables if t.endswith("_gene_summary")), None)
        if summary_table is None:
            raise RuntimeError(
                f"no *_gene_summary table found in {duckdb_files[0]} (tables present: {tables})")
        cur = con.execute(f'SELECT * FROM "{summary_table}"')
        cols = [d[0] for d in cur.description]
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"{summary_table} in {duckdb_files[0]} is empty")
        return dict(zip(cols, row))
    finally:
        con.close()


def _gold_genes(phenopacket_path: Path) -> set[str]:
    try:
        d = json.loads(phenopacket_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    genes: set[str] = set()
    for interp in d.get("interpretations", []):
        diagnosis = interp.get("diagnosis") or {}
        for gi in diagnosis.get("genomicInterpretations", []):
            ctx = ((gi.get("variantInterpretation") or {})
                   .get("variationDescriptor", {}).get("geneContext"))
            if ctx and ctx.get("symbol"):
                genes.add(ctx["symbol"].upper())
    return genes


def per_case_hits(raw_dir: Path, phenopacket_dir: Path, gid=None) -> dict[str, int | None]:
    hits: dict[str, int | None] = {}
    for raw_path in sorted(raw_dir.glob("*.json")):
        case_id = raw_path.stem
        pp_path = phenopacket_dir / f"{case_id}.json"
        if not pp_path.exists():
            continue
        gold = _gold_genes(pp_path)
        if not gold:
            continue
        try:
            ranked = json.loads(raw_path.read_text(encoding="utf-8"))
        except Exception:
            ranked = []

        gold_ids: set[str] = set()
        if gid is not None:
            for g in gold:
                try:
                    ident = gid.find_identifier(g)
                except Exception:
                    ident = None
                if ident:
                    gold_ids.add(str(ident).upper())

        rank = None
        for i, entry in enumerate(ranked, start=1):
            sym = str(entry[0]).upper() if entry else ""
            if sym in gold or sym in gold_ids:
                rank = i
                break
        hits[case_id] = rank
    return hits


def hit_vector(hits: dict[str, int | None], k: int = 1) -> list[int]:
    return [1 if (r is not None and r <= k) else 0 for r in hits.values()]


def per_case_hits_from_results(results_dir: Path) -> dict[str, int | None]:
    """Read each case's hit rank straight from pheval-utils' own scored parquet output
    (the `rank`/`true_positive` columns `generate_gene_result()` wrote), instead of
    re-deriving truth with a second, independent gene-symbol matcher.

    `per_case_hits()` above re-implements its own symbol/identifier matching and can
    disagree with what pheval-utils itself counted as a hit on a nontrivial fraction of
    cases (verified: ~0.8pp on the 4,677-case PhEval corpus) -- any CI bootstrapped from
    its output is then a CI for a *different* statistic than the reported point estimate,
    not a wider/narrower CI for the same one. This function guarantees the hit vector used
    for the CI is definitionally the same one pheval-utils used for precision@1/summary.
    """
    import pandas as pd

    hits: dict[str, int | None] = {}
    result_dir = results_dir / "pheval_gene_results"
    if not result_dir.exists():
        result_dir = results_dir
    for f in sorted(result_dir.glob("*-gene_result.parquet")):
        case_id = f.name[: -len("-gene_result.parquet")]
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        tp = df[df["true_positive"] == True]  # noqa: E712
        # pheval-utils writes a true_positive=True row even for a gene GeneScout never
        # ranked, using rank=0 as a "not found" sentinel -- exclude it, or it silently
        # counts as a top-1 "hit" (verified: inflated 0.4353 -> 0.6889 on the PhEval corpus).
        tp = tp[tp["rank"] > 0]
        hits[case_id] = int(tp["rank"].min()) if not tp.empty else None
    return hits
