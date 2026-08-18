from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):  # type: ignore[misc]
        return it


def _venv_python() -> str:
    """Stages 4-5 import polars/pheval, which live in this project's .venv — not
    necessarily whatever interpreter launched this script. Falls back to sys.executable
    if no .venv is found (e.g. those packages are on the current interpreter instead)."""
    for candidate in (
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "bin" / "python",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _pheval_utils_available() -> bool:
    for candidate in (
        ROOT / ".venv" / "Scripts" / "pheval-utils.exe",
        ROOT / ".venv" / "Scripts" / "pheval-utils",
        ROOT / ".venv" / "bin" / "pheval-utils",
    ):
        if candidate.exists():
            return True
    import shutil
    return shutil.which("pheval-utils") is not None


def _check_prereqs(stages: list[int]) -> None:
    problems: list[str] = []

    if {4, 5} & set(stages) and not _pheval_utils_available():
        problems.append(
            "`pheval-utils` not found (checked .venv/Scripts, .venv/bin, and PATH) — "
            "needed for stages 4 and 5. Install with `pip install pheval` into this "
            "project's `.venv` specifically (stages 4-5 always run under `.venv`'s "
            "interpreter regardless of what launched this script)."
        )

    if 4 in stages:
        corpus = (ROOT / "external" / "pheval_corpora" / "corpora"
                  / "phenopacket_store_0.1.11_phenotypes" / "phenopackets")
        if not corpus.exists():
            problems.append(f"Stage 4 corpus missing: {corpus}\n"
                             "    See evaluation/README.md for how to fetch it.")

    if 5 in stages:
        sibling = ROOT.parent / "monarch_pheval"
        if not sibling.exists():
            problems.append(
                f"Stage 5 needs a sibling checkout at {sibling} "
                "(LIRICAL corpus + hg19 template exome). Not vendored in this repo — "
                "see evaluation/README.md."
            )

    if problems:
        print("Prerequisite check failed:\n", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)


def _run_subprocess(module: str, args: list[str], env_overrides: dict[str, str] | None = None) -> None:
    import os
    sys.stdout.flush()  # keep our own stage-header prints from reordering vs. subprocess output
    env = {**os.environ, **(env_overrides or {})}
    subprocess.run([_venv_python(), "-m", module, *args], cwd=ROOT, env=env, check=True)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [warn] could not read {path}: {exc}", file=sys.stderr)
        return {}


def _stage1_deidentification(smoke: bool) -> dict:
    print("\n=== Stage 1/5 — de-identification ===")
    from evaluation.stage1_deidentification.eval_deidentification import run as run_deid
    res = run_deid(limit=10 if smoke else 0)
    return {"n": res.n, "metrics": res.metrics}


def _stage2_phenotype_extraction(smoke: bool) -> dict:
    print("\n=== Stage 2/5 — phenotype extraction (GSC+) ===")
    from evaluation.stage2_phenotype_extraction.eval_phenotype_extraction import run as run_hpo
    res = run_hpo(limit=20 if smoke else 0)
    return {"n": res.n, "mode": res.mode, "metrics": res.metrics}


def _stage3_etiology_classification(smoke: bool) -> dict:
    print("\n=== Stage 3/5 — etiology classification (92-case set, few-shot x context ablation) ===")
    from evaluation.stage3_etiology_classification.eval_etiology_classification import run as run_cls
    res = run_cls(limit=10 if smoke else 0)
    return {"n": res.n, "mode": res.mode, "metrics": res.metrics}


def _stage4_phenotype_gene_ranking(smoke: bool) -> dict:
    print("\n=== Stage 4/5 — phenotype-only gene ranking (PhEval / Phenopacket Store) ===")
    n = "50" if smoke else "0"  # 0 = full 4,677-case corpus, matching the paper
    _run_subprocess("evaluation.stage4_phenotype_gene_ranking.eval_phenotype_gene_ranking", [n])
    corpus_name = "phenopacket_store_0.1.11_phenotypes"
    return _read_json(ROOT / "eval_runs" / f"_pheval_{corpus_name}" / "metrics.json")


def _stage5_genome_vcf_ranking(smoke: bool) -> dict:
    print("\n=== Stage 5/5 — genome-level (phenotype+VCF) ranking (LIRICAL-385) ===")
    env = {"GENERESOLVER_VCF_LIMIT": "5"} if smoke else {}
    _run_subprocess("evaluation.stage5_genome_vcf_ranking.eval_genome_vcf_ranking", [], env_overrides=env)
    return _read_json(ROOT / "eval_runs" / "_lirical_vcf" / "metrics.json")


_STAGE_NAMES = {
    1: "de-identification",
    2: "phenotype extraction",
    3: "etiology classification",
    4: "phenotype-only gene ranking",
    5: "genome-level (+VCF) ranking",
}
_STAGE_DURATION_SMOKE = {
    1: "~1 min", 2: "~1 min", 3: "a few minutes",
    4: "a couple minutes", 5: "a few minutes",
}
_STAGE_DURATION_FULL = {
    1: "~1 min", 2: "~1 min", 3: "depends on LLM backend",
    4: "hours (4,677 cases)",
    5: "up to ~16 hours COLD (385 cases x2 arms, local VEP/Docker + LLM per case; "
       "cached per-case under eval_runs/_lirical_vcf/ so a re-run only redoes what's missing)",
}
_STAGES = {
    1: _stage1_deidentification,
    2: _stage2_phenotype_extraction,
    3: _stage3_etiology_classification,
    4: _stage4_phenotype_gene_ranking,
    5: _stage5_genome_vcf_ranking,
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                     help="small limit on every stage (minutes, not hours)")
    ap.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5],
                     help="run only this stage (default: stage 5 only -- the slowest one, "
                          "up to ~16h without --smoke; pass --stage explicitly to pick a "
                          "cheaper stage instead)")
    ap.add_argument("--out", default=str(ROOT / "eval_runs" / "reproduce_paper_etiology" / "paper_tables.xlsx"),
                     help="Excel workbook to write")
    args = ap.parse_args()

    stages = [args.stage] if args.stage else [5]
    _check_prereqs(stages)

    durations = _STAGE_DURATION_SMOKE if args.smoke else _STAGE_DURATION_FULL
    print("Estimated runtime:")
    for s in stages:
        print(f"  Stage {s} ({_STAGE_NAMES[s]}): {durations[s]}")

    results: dict[int, dict] = {}
    for s in tqdm(stages, desc="reproduce_paper_results", unit="stage"):
        tqdm.write(f"\n[{stages.index(s) + 1}/{len(stages)}] Stage {s} — {_STAGE_NAMES[s]}")
        results[s] = _STAGES[s](args.smoke)

    from evaluation.paper_tables import build_workbook
    out_path = Path(args.out)
    build_workbook(results, out_path)

    print("\n== done ==")
    print(f"Wrote tables -> {out_path}")
    print("Compare against unused/paper/generesolve_paper.tex's Table~tab:deid/hpo/classifier/"
          "pheval/vcf.")


if __name__ == "__main__":
    main()
