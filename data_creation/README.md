# Data creation

Scripts for the two datasets used in the paper's evaluations.

## De-identification test set (`deid/`)

100 synthetic clinical notes with annotated PHI spans, in `generated_deid_dataset.jsonl`.

```bash
cd deid
python generate_deid_dataset.py --source-dir ../..
```

Resumable — re-running only fills in missing `PHI_001`..`PHI_100` cases.

## Etiology silver-standard set (`etiology_silver_pipeline/`)

220 PubMed/PMC case reports converted to labeled vignettes, in
`etiology_silver_pipeline/output/gold_vignettes_v2.json`.

```bash
cd etiology_silver_pipeline
python 01_retrieve_cases.py --source-dir ../.. --full-text-only
python 02_backfill_pmcid.py
python 03_demote_abstract_only.py
python 04_generate_dataset.py --source-dir ../.. --allow-live-fetch
```

`00_confirm_disease_seeds.py --source-dir ../..` runs independently (before or after)
to audit the seed disease list against literature.

Each stage checkpoints to `work/` and is resumable.
