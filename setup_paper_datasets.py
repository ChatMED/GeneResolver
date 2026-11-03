"""Fetch and verify the five small evaluation corpora described in the manuscript.

This does not download runtime reference databases or the optional local VEP cache.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    for module in (
        "evaluation.table5_phenotype_extraction.download_gsc_plus",
        "evaluation.table9_phenotype_gene_ranking.download_pheval_corpora",
        "evaluation.table10_vcf_ablation.download_monarch_pheval",
    ):
        subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)

    deid = ROOT / "data_creation" / "deid" / "generated_deid_dataset.jsonl"
    if not deid.exists():
        raise SystemExit(f"Stage 1 synthetic PHI notes missing: {deid}")
    print(f"Stage 1 synthetic notes: {sum(bool(line.strip()) for line in deid.open(encoding='utf-8'))}")

    from evaluation.table6_7_etiology_classification.eval_etiology_classification import (
        load_cases, pick_few_shot_pool,
    )
    vignettes = ROOT / "data_creation" / "etiology_silver_pipeline" / "output" / "gold_vignettes_v2.json"
    cases = load_cases(vignettes)
    pool, scored = pick_few_shot_pool(cases, 2, 20260731)
    if (len(cases), len(pool), len(scored)) != (204, 8, 196):
        raise SystemExit(f"Stage 3 corpus mismatch: {len(cases)} usable, {len(pool)} held out, {len(scored)} scored")
    print("Stage 3 etiology cases: 204 usable, 8 held out, 196 scored")
    print("Paper evaluation corpora are ready. Runtime reference datasets are separate; see README.md.")


if __name__ == "__main__":
    main()
