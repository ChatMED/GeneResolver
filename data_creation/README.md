# Data creation

Scripts and data for the three evaluation datasets built for the paper. The datasets themselves are in the repository;
the scripts are only needed to regenerate them. Dataset construction used cloud LLMs through
`llm_clients/` (install with `pip install -r data_creation/requirements.txt`). GeneResolver itself
never calls these clients; it runs only the local MedGemma model.

## De-identification test set (`deid/`)

`deid/generated_deid_dataset.jsonl`: 100 synthetic clinical notes (`PHI_001`–`PHI_100`). Each
record holds the note and `expected_pii_tokens`, the list of PHI strings planted in it
(3,632 PHI tokens in total, scored as token-level recall in Table 4). Generated with Claude via
the Anthropic API; protocol and prompt in Supplementary Material 2.

```bash
python data_creation/deid/generate_deid_dataset.py --source-dir .   # resumable
```

## Etiology silver-standard set (`etiology_silver_pipeline/`)

`etiology_silver_pipeline/output/gold_vignettes_v2.json`: 220 vignettes derived from published
PubMed/PMC case reports and labelled single-gene, chromosomal, multifactorial or non-genetic
(196 scored in Tables 6–7 after eight few-shot pool cases are held out). Retrieval and vignette
generation used `gpt-5.4-mini` through the OpenAI API; procedure and prompts in Supplementary
Material 3. The classifier prompt and the exact input it receives are in `docs/LLM_PROMPTS.md`.

```bash
cd data_creation/etiology_silver_pipeline
python 01_retrieve_cases.py --source-dir ../.. --full-text-only
python 02_backfill_pmcid.py
python 03_demote_abstract_only.py
python 04_generate_dataset.py --source-dir ../.. --allow-live-fetch
```

`00_confirm_disease_seeds.py` audits the seed disease list independently. Every step
checkpoints to `work/` and resumes.

## ACMG/AMP validation sets (`acmg_validation/`)

ClinVar ≥2-star (multiple submitters, expert panel or practice guideline) coding/splice variants,
GRCh37, from the local ClinVar VCF (2026-10-19); used in Table 8 and `docs/ACMG_RULES.md`. No LLM
was used.

| File | Content |
|---|---|
| `dev_400.json` | development set: 100 each Pathogenic, Likely pathogenic, Likely benign, Benign (seed 20260731); used to design engine v2/v3 |
| `heldout_214.json` | held-out set: 81 P, 6 LP, 27 LB, 100 B (seed 20261102), none in `dev_400`; drawn from variants already annotated in a Table 10 run (same VEP configuration) whose gene has an Orphanet gene–disease pair, which limits the LP and LB counts; never used for design |
| `acmg_validation_variants.tsv` | both sets in one table (`set` column) |
| `genebe_responses.jsonl` | ACMG fields of GeneBe's public-API responses for the held-out set (2026-11-02) |

```bash
python -m evaluation.table8_acmg_validation.acmg_validation_balanced --per-class 100       # development set
python -m evaluation.table8_acmg_validation.eval_acmg_validation sample --stage5-run <run>  # held-out set
python -m evaluation.table8_acmg_validation.eval_acmg_validation genebe
```
