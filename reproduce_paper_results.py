"""Reproduce every evaluation table of the GeneResolver paper (Tables 4-10) with one command.

    python reproduce_paper_results.py                      # run everything, then build all tables
    python reproduce_paper_results.py --table 4 5          # run only these tables (others are reused if saved)
    python reproduce_paper_results.py --smoke              # a few cases per table, separate output folder
    python reproduce_paper_results.py --tables-only        # rebuild tables from saved results, no runs
    python reproduce_paper_results.py --tables-only --results-dir evaluation/paper_results   # the paper's results

Table -> script (evaluation/<folder>):
    4     de-identification, 100 synthetic notes          table4_deidentification
    5     phenotype extraction, GSC+ 228 documents         table5_phenotype_extraction
    6, 7  etiology classification, 196 cases, 4 configs    table6_7_etiology_classification   [MedGemma]
    8     ACMG/AMP engine vs ClinVar and GeneBe, 214 held-out variants   table8_acmg_validation   [Docker VEP]
    9     phenotype-only ranking, PhEval 4,677 cases       table9_phenotype_gene_ranking
    10    VCF ablation, LIRICAL spiked exomes, 385 cases   table10_vcf_ablation               [Docker VEP]

Results are saved as ``<results-dir>/stageN.json`` (1 = Table 4, 2 = Table 5, 3 = Tables 6-7,
6 = Table 8, 4 = Table 9, 5 = Table 10); tables go to ``<results-dir>/tables/``. Tables 6-7 run every
configuration ``--repeats`` times with the LLM response cache disabled; ``--stage3-predictions``
scores existing prediction files instead (when several contain the same configuration, the later
file wins for it).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generesolver.env import load_project_env  # noqa: E402

load_project_env(ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STAGE_NAMES = {1: "de-identification", 2: "phenotype extraction", 3: "etiology classification",
               4: "phenotype-only gene ranking (PhEval)", 5: "VCF ablation (LIRICAL spiked exomes)",
               6: "ACMG/AMP validation (held-out ClinVar variants, GeneBe)"}
FULL_RUNTIME = {1: "~20 min", 2: "~20 min",
                3: "~1 h per run x 4 configs (MedGemma)", 4: "many hours (4,677 cases, no LLM calls)",
                5: "~20 h per arm cold (385 cases, Docker VEP, no LLM calls)",
                6: "minutes with cached VEP annotations (214 variants, no LLM or GeneBe calls)"}
STAGE3_VIGNETTES = ROOT / "data_creation" / "etiology_silver_pipeline" / "output" / "gold_vignettes_v2.json"
PHEVAL_CORPUS = ROOT / "external" / "pheval_corpora" / "corpora" / "phenopacket_store_0.1.11_phenotypes"


def _python() -> str:
    for c in (ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"):
        if c.exists():
            return str(c)
    return sys.executable


def _subprocess(module: str, args: list[str], env: dict[str, str]) -> None:
    sys.stdout.flush()
    # UTF-8 output: on Windows a piped child otherwise prints in cp1252, and a non-ASCII title
    # (e.g. "é" in a retrieved abstract) raised inside RAG retrieval and dropped the case.
    subprocess.run([_python(), "-m", module, *args], cwd=ROOT,
                   env={**os.environ, "PYTHONIOENCODING": "utf-8", **env}, check=True)


def _lirical_root() -> Path:
    local = ROOT / "external" / "monarch_pheval"
    return local if local.exists() else ROOT.parent / "monarch_pheval"


def medgemma_server_info() -> dict:
    """Ask the MedGemma endpoint what it serves; raises if it is unreachable or serves another model."""
    import httpx
    from generesolver.integrations import medgemma_client
    from generesolver.integrations.llm_client import active_backend
    active_backend()
    base = os.environ.get("GENERESOLVER_MEDGEMMA_BASE_URL", "").rstrip("/")
    if not base:
        raise RuntimeError("GENERESOLVER_MEDGEMMA_BASE_URL is not set (needed for Stage 2 enrichment and Stage 3).")
    model = os.environ.get("GENERESOLVER_MEDGEMMA_MODEL") or medgemma_client.DEFAULT_MODEL
    headers = {"Authorization": f"Bearer {os.environ['GENERESOLVER_MEDGEMMA_API_KEY']}"}         if os.environ.get("GENERESOLVER_MEDGEMMA_API_KEY") else {}
    try:
        served = httpx.get(f"{base}/models", headers=headers, timeout=15).json()
    except Exception as exc:
        raise RuntimeError(f"MedGemma server at {base} is not reachable: {exc}") from exc
    ids = [m.get("id") for m in served.get("data", [])]
    if model not in ids:
        raise RuntimeError(f"{base} serves {ids}, not {model!r} (set GENERESOLVER_MEDGEMMA_MODEL).")
    info = {"base_url": base, "model": model, "models_endpoint": served,
            "checked": datetime.now(timezone.utc).isoformat()}
    root = base[:-3] if base.endswith("/v1") else base
    try:  # vLLM exposes its version here; other servers may not
        info["server_version"] = httpx.get(f"{root}/version", timeout=10).json()
    except Exception:
        info["server_version"] = None
    return info


def _check_prereqs(stages: list[int]) -> None:
    problems = []
    if 3 in stages:
        try:
            medgemma_server_info()
        except (RuntimeError, ValueError) as exc:
            problems.append(str(exc))
    if 4 in stages and not (PHEVAL_CORPUS / "phenopackets").exists():
        problems.append(f"Table 9 corpus missing: {PHEVAL_CORPUS} (run python setup_paper_datasets.py)")
    if 5 in stages and not (_lirical_root() / "corpora" / "lirical" / "default" / "phenopackets").exists():
        problems.append(f"Table 10 LIRICAL corpus missing under {_lirical_root()} (run python setup_paper_datasets.py)")
    if problems:
        sys.exit("Prerequisite check failed:\n  - " + "\n  - ".join(problems))


# ----------------------------------------------------------------------------- stages
def stage1(results_dir: Path, args) -> dict:
    from evaluation.table4_deidentification.eval_deidentification import run
    res = run(limit=10 if args.smoke else 0)
    return {"n": res.n, "metrics": res.metrics}


def stage2(results_dir: Path, args) -> dict:
    from evaluation.table5_phenotype_extraction.eval_phenotype_extraction import run
    res = run(limit=20 if args.smoke else 0)
    return {"n": res.n, "mode": res.mode, "metrics": dict(res.metrics)}


def _load_predictions(paths: list[Path]) -> list[dict]:
    from evaluation.table6_7_etiology_classification.eval_etiology_classification_score import load_rows
    merged: dict[tuple[bool, bool], list[dict]] = {}
    for path in paths:
        by_cond: dict[tuple[bool, bool], list[dict]] = {}
        for row in load_rows(path):
            by_cond.setdefault((row["few_shot"], row["rag"]), []).append(row)
        merged.update(by_cond)
    return [r for rows in merged.values() for r in rows]


def stage3(results_dir: Path, args) -> dict:
    from evaluation.paper_tables import score_etiology
    if args.stage3_predictions:
        paths = [Path(p) for p in args.stage3_predictions]
    else:
        out = results_dir / "stage3_predictions.jsonl"
        cmd = ["--vignettes", str(STAGE3_VIGNETTES), "--out", str(out), "--repeats", str(args.repeats)]
        if args.smoke:
            cmd += ["--limit", "3"]
        # Repeats must be real model calls, so the response cache is always off here.
        _subprocess("evaluation.table6_7_etiology_classification.eval_etiology_classification", cmd,
                    {"GENERESOLVER_LLM_CACHE_DISABLE": "1"})
        paths = [out]
    scored = score_etiology(_load_predictions(paths))
    scored["predictions"] = [p.as_posix() for p in paths]
    if not args.stage3_predictions:
        scored["server"] = medgemma_server_info()
    return scored


def stage4(results_dir: Path, args) -> dict:
    run_dir = Path(args.stage4_run_dir) if args.stage4_run_dir else results_dir / "stage4_run"
    if not args.reuse_runs:
        _subprocess("evaluation.table9_phenotype_gene_ranking.eval_phenotype_gene_ranking",
                    ["50" if args.smoke else "0", str(PHEVAL_CORPUS)],
                    {"GENERESOLVER_STAGE4_RUN_DIR": str(run_dir)})
    return {**json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")), "run_dir": run_dir.as_posix()}


def stage5(results_dir: Path, args) -> dict:
    run_dir = Path(args.stage5_run_dir) if args.stage5_run_dir else results_dir / "stage5_run"
    if not args.reuse_runs:
        env = {"GENERESOLVER_STAGE5_RUN_DIR": str(run_dir), "GENERESOLVER_LOCAL_MODE": "1"}
        if args.smoke:
            env["GENERESOLVER_VCF_LIMIT"] = "5"
        _subprocess("evaluation.table10_vcf_ablation.eval_genome_vcf_ranking", [], env)
    out = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest = run_dir / "run_manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        out["manifest"] = {k: m.get(k) for k in ("filter_profile", "clinvar_triage", "git_commit", "code_hash")}
    return {**out, "run_dir": run_dir.as_posix()}


def stage6(results_dir: Path, args) -> dict:
    # Re-scores the held-out variants with the current engine (local VEP) and compares with the saved
    # GeneBe responses (data_creation/acmg_validation/genebe_responses.jsonl); GeneBe is not queried.
    from evaluation.table8_acmg_validation import eval_acmg_validation as t8
    os.environ["GENERESOLVER_LOCAL_MODE"] = "1"
    tag = f"harness_{datetime.now():%Y%m%d_%H%M%S}"
    t8.score(tag)
    return {**t8.report(tag, results_dir / "acmg_heldout"), "tag": tag}


STAGES = {1: stage1, 2: stage2, 3: stage3, 4: stage4, 5: stage5, 6: stage6}
TABLE_TO_STAGE = {4: 1, 5: 2, 6: 3, 7: 3, 8: 6, 9: 4, 10: 5}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=int, nargs="+", choices=sorted(TABLE_TO_STAGE),
                    help="paper tables to reproduce (default: all)")
    ap.add_argument("--stage", type=int, nargs="+", choices=sorted(STAGES), help=argparse.SUPPRESS)
    ap.add_argument("--tables-only", action="store_true", help="build tables from saved stage results only")
    ap.add_argument("--smoke", action="store_true", help="a few cases per stage, written to eval_runs/paper_smoke")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="where stageN.json and tables/ live (default eval_runs/paper, or eval_runs/paper_smoke)")
    ap.add_argument("--repeats", type=int, default=5, help="Stage 3 runs per configuration (default 5)")
    ap.add_argument("--stage3-predictions", nargs="+", default=None,
                    help="score these Stage 3 prediction files instead of calling the model")
    ap.add_argument("--stage4-run-dir", default=None, help="Table 9 run folder (default <results-dir>/stage4_run)")
    ap.add_argument("--stage5-run-dir", default=None, help="Table 10 run folder (default <results-dir>/stage5_run)")
    ap.add_argument("--reuse-runs", action="store_true",
                    help="Tables 9-10: score the existing run folders without launching the pipeline")
    args = ap.parse_args()

    results_dir = args.results_dir or ROOT / "eval_runs" / ("paper_smoke" if args.smoke else "paper")
    results_dir.mkdir(parents=True, exist_ok=True)
    chosen = sorted({TABLE_TO_STAGE[t] for t in args.table or []} | set(args.stage or []))
    stages = [] if args.tables_only else (chosen or sorted(STAGES))
    will_run = set(stages)  # stages that actually launch a model or pipeline
    if args.stage3_predictions:
        will_run.discard(3)
    if args.reuse_runs:
        will_run -= {4, 5}
    _check_prereqs(sorted(will_run))

    for s in stages:
        print(f"\n=== Stage {s}: {STAGE_NAMES[s]} ({'smoke' if args.smoke else FULL_RUNTIME[s]}) ===", flush=True)
        result = STAGES[s](results_dir, args)
        result["_meta"] = {"stage": s, "smoke": args.smoke, "finished": datetime.now(timezone.utc).isoformat()}
        (results_dir / f"stage{s}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    results = {}
    for s in STAGES:
        f = results_dir / f"stage{s}.json"
        if f.exists():
            results[s] = json.loads(f.read_text(encoding="utf-8"))
    missing = [s for s in STAGES if s not in results]
    from evaluation.paper_tables import build_tables
    build_tables(results, results_dir / "tables")
    print(f"\nTables -> {results_dir / 'tables'} (paper_tables.md, paper_tables.xlsx, one CSV/MD per table)")
    if missing:
        print(f"No saved results for stage(s) {missing}; their tables were skipped.")


if __name__ == "__main__":
    main()
