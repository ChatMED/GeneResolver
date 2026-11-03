# GeneResolver evaluation

Each folder reproduces one table of the paper. One command runs everything and builds the tables:

```bash
python reproduce_paper_results.py                    # every table
python reproduce_paper_results.py --table 8 9        # only these tables (others rebuilt from saved results)
python reproduce_paper_results.py --smoke            # a few cases per table, written to eval_runs/paper_smoke/
python reproduce_paper_results.py --tables-only      # rebuild the tables from saved results, no runs
```

Results are saved in `--results-dir` (default `eval_runs/paper/`); tables go to `<results-dir>/tables/`
(one CSV and Markdown file per table, plus `paper_tables.md` and `paper_tables.xlsx`).

| Table | Folder | Evaluation | Data | Needs | Full run |
|---|---|---|---|---|---|
| 4 | `table4_deidentification/` | De-identification | 100 synthetic notes, 3,632 PHI tokens (`data_creation/deid/`) | CPU, ~8 GB free RAM | ~20 min |
| 5 | `table5_phenotype_extraction/` | HPO extraction: FastHPOCR vs ClinPhen | GSC+, 228 documents | CPU | ~20 min |
| 6, 7 | `table6_7_etiology_classification/` | Etiology classification, few-shot × RAG | 220-case silver corpus, 196 evaluated (`data_creation/etiology_silver_pipeline/`) | MedGemma | ~1 h per run per configuration |
| 8 | `table8_acmg_validation/` | ACMG/AMP engine vs ClinVar, and vs GeneBe | 214 held-out ClinVar variants (`data_creation/acmg_validation/`) | Docker VEP | minutes with cached annotations |
| 9 | `table9_phenotype_gene_ranking/` | Phenotype-only gene ranking | PhEval / Phenopacket Store 0.1.11, 4,677 cases | CPU | many hours |
| 10 | `table10_vcf_ablation/` | VCF ablation: phenotype only vs phenotype + VCF | LIRICAL, 385 cases, public hg19 template exome | Docker VEP | ~20 h per arm |

Only Tables 6–7 call MedGemma; its context length (`--max-model-len`) must be at
least 16,384 tokens. No evaluation sends data to an external service except Table 8's optional
`genebe` step (public ClinVar variants only); the saved GeneBe responses are used by default.

## Setup

```bash
pip install -r requirements.txt
pip install "datasets>=2.16,<3.0" numpy "pheval==0.5.8" "duckdb==1.2.2" openpyxl pandas tabulate openmed "clinphen==1.28"
python setup_paper_datasets.py                     # GSC+, PhEval and LIRICAL corpora
python setup_quickstart.py --with-vep-cache        # VEP 116 GRCh37 cache + Docker image (Tables 8, 10)
export GENERESOLVER_MEDGEMMA_BASE_URL=http://localhost:8000/v1   # Tables 6-7
```

Run from the repository root. The only language model is MedGemma 27B IT on your own server.

## Tables 6–7

Each configuration runs `--repeats` times (default 5) with the LLM response cache disabled. Cases
need a valid label in every run of every configuration; Table 6 uses the majority label over runs.
`--stage3-predictions <file.jsonl>` scores saved predictions without calling the model.

## Table 8

`data_creation/acmg_validation/` holds the variant sets and the GeneBe responses:
`dev_400.json` (development set, 100 each P, LP, LB, B; used to design the engine),
`heldout_214.json` (81 P, 6 LP, 27 LB, 100 B; never used for design), `acmg_validation_variants.tsv`
(both sets) and `genebe_responses.jsonl` (ACMG fields of GeneBe's responses, queried 2026-11-02).

```bash
python -m evaluation.table8_acmg_validation.eval_acmg_validation score --tag <tag>    # current engine, local VEP
python -m evaluation.table8_acmg_validation.eval_acmg_validation report --tag <tag>   # vs ClinVar and GeneBe
python -m evaluation.table8_acmg_validation.eval_acmg_validation genebe               # re-query GeneBe (optional)
python -m evaluation.table8_acmg_validation.acmg_validation_balanced --rescore data_creation/acmg_validation/dev_400.json --tag <tag>
```

Set `GENERESOLVER_LOCAL_MODE=1`. Both tools are scored as returned, with all criteria; both use
ClinVar-derived evidence, so agreement with ClinVar is an upper bound (see `docs/ACMG_RULES.md`).

## Tables 9–10

`--stage4-run-dir` (Table 9) / `--stage5-run-dir` (Table 10) choose the run folders and `--reuse-runs`
scores them without running the pipeline. Table 10 caches per case and resumes; `run_manifest.json`
records the code version, VEP configuration and resources and refuses to mix code versions. Defaults
are the paper's configuration (original variant filter, ClinVar P/LP triage on);
`GENERESOLVER_CLINVAR_TRIAGE=0` gives the ClinVar-blind arm and
`GENERESOLVER_VARIANT_FILTER_PROFILE=revised` the revised filter, each in its own run folder. The
Table 9 and Table 10 scripts set `GENERESOLVER_CASE_BRANCHING=0` (every case is ranked, no etiology
routing) and are scored with `pheval-utils benchmark`
(`common/pheval_scoring.py`). Table 9's baselines are the published PhEval statistics.
Phenopacket Store diseases overlap with GeneResolver's knowledge bases and possibly with the model's
training data, so report the bootstrap CIs, not only point estimates.

## Results folder

Each run saves one `stageN.json`. "Stage" is the harness's internal run number, not a pipeline
step; stage 6 was added last, so the numbers do not follow the table order:

| Stage file | Paper table | Harness options that use the stage number |
|---|---|---|
| `stage1.json` | Table 4, de-identification | |
| `stage2.json` | Table 5, phenotype extraction | |
| `stage3.json` | Tables 6–7, etiology classification | `--stage3-predictions` |
| `stage4.json` | Table 9, phenotype-only ranking | `--stage4-run-dir`, `GENERESOLVER_STAGE4_RUN_DIR` |
| `stage5.json` | Table 10, VCF ablation | `--stage5-run-dir`, `GENERESOLVER_STAGE5_RUN_DIR` |
| `stage6.json` | Table 8, ACMG/AMP validation | |

`--table` takes paper table numbers and maps them to stages. `tests/test_paper_reproduction.py`
checks that saved results rebuild the tables.
