# GeneResolver Evaluation Harness

Reproduces the paper's 5 evaluations. Fastest path:

```bash
python reproduce_paper_results.py --smoke --stage N   # quick check, any stage
python reproduce_paper_results.py --stage N            # full run
```

## Install

```bash
pip install "datasets>=2.16,<3.0" numpy pheval duckdb openpyxl openmed
```
Run everything from the repo root so `generesolver` and `evaluation` import correctly.

## The 5 stages

| # | Stage | Folder | Data | Full-run time                              |
|---|---|---|---|--------------------------------------------|
| 1 | De-identification | `stage1_deidentification/` | 100 synthetic PHI notes | ~20 min                                    |
| 2 | Phenotype extraction | `stage2_phenotype_extraction/` | GSC+ corpus, 228 docs | ~20 min                                    |
| 3 | Etiology classification | `stage3_etiology_classification/` | 220-case corpus (`gold_vignettes_v2.json`), 8 held out as few-shot pool, 196 scored | depends on LLM backend                     |
| 4 | Phenotype-only gene ranking | `stage4_phenotype_gene_ranking/` | PhEval/Phenopacket Store, 4,677 cases | hours                                      |
| 5 | Genome-level (+VCF) ranking | `stage5_genome_vcf_ranking/` | LIRICAL, 385 cases | **up to ~16h cold**, cached per-case after |

## Stages 1-3 — no extra setup

```bash
python -m evaluation.stage1_deidentification.eval_deidentification
python -m evaluation.stage2_phenotype_extraction.eval_phenotype_extraction
python -m evaluation.stage3_etiology_classification.eval_etiology_classification
```
Production config is **few-shot=No, context=Raw** (Table 5's top row) — that's what `generesolver/classification/engine.py` runs.

## Stage 4 — fetch then run

```bash
python -m evaluation.stage4_phenotype_gene_ranking.download_pheval_corpora
python -m evaluation.stage4_phenotype_gene_ranking.eval_phenotype_gene_ranking 0   # 0 = full 4,677; smaller N for a quick check
```

## Stage 5 — fetch then run

```bash
python -m evaluation.stage5_genome_vcf_ranking.download_monarch_pheval   # clones ../monarch_pheval
python -m evaluation.stage5_genome_vcf_ranking.eval_genome_vcf_ranking
```
Needs Docker running (local VEP). For a quick check set `GENERESOLVER_VCF_LIMIT=5`.

## Notes

- Stages 4/5 auto-set `GENERESOLVER_CASE_BRANCHING=0` and score via `common/pheval_scoring.py` (`pheval-utils benchmark`).
- Stage 5's spiked VCFs must match GeneResolver's VEP build or matching silently fails.
- RareBench/Phenopacket Store diseases overlap with GeneResolver's KBs and possibly LLM training data — report bootstrap CIs, not just point estimates.
