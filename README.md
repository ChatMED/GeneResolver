# GeneResolver

Privacy-aware, multi-agent pipeline for gene prioritization: it
ingests a clinical note, HPO terms, and/or a VCF, decides whether a case is a single-gene
problem before ranking, and fuses multi-source evidence (phenotype, variant, ACMG,
literature, GWAS) into a ranked gene list — keeping the full case on-premise when
`GENERESOLVER_LOCAL_MODE=1` is set.

## Setup

```bash
pip install -r requirements.txt
python setup_quickstart.py
```

`setup_quickstart.py` fetches every reference dataset the pipeline needs (HPO, Orphanet,
Reactome, gnomAD constraint, MONDO, ClinVar, MANE, ClinGen, UniProt, gnomAD exomes-sites,
REVEL, UCSC phyloP/simpleRepeat — all idempotent, safe to re-run), checks your LLM backend
configuration, then runs a demo case. The one large opt-in step (~23GB local VEP cache +
Docker image, needed for on-premise VCF annotation) is behind `--with-vep-cache`.
It does not fetch evaluation-only corpora (GSC+, PhEval Phenopacket Store, monarch_pheval)

## Running a case

```bash
python run_generesolver.py --example single_gene
python run_generesolver.py --example multifactorial
python run_generesolver.py --example vcf_lirical
```

Or with your own input: `python run_generesolver.py --note "..." --vcf path/to.vcf.gz`
(see `python run_generesolver.py --help` for the full option list). Output is written to
`outputs/<case_id>/`.

## LLM backend

Defaults to a local MedGemma endpoint (`GENERESOLVER_LLM_BACKEND=medgemma`, the GDPR-local
default — no patient data leaves the machine). Point it at your own server:

```bash
export GENERESOLVER_MEDGEMMA_BASE_URL=http://localhost:8000/v1   # your vLLM endpoint
```

Claude and OpenAI backends are also available (`GENERESOLVER_LLM_BACKEND=claude` /
`openai`) but are not local — only use them on data you're permitted to send off-premise.

## Demo outputs

`outputs/` ships pre-generated results for the three bundled examples so you can inspect
real output without running anything first.

- `DEMO_CASE_SINGLE_GENE/`, `DEMO_CASE_MULTIFACTORIAL/` — full output (`case_state.json`,
  `report.json`, `diagnostic_report.fhir.json`, `phenopacket.json`).
- `DEMO_CASE_VCF_LIRICAL/` — only the compact artifacts are included
  (`gene_ranking.csv`, `diagnostic_report.fhir.json`, `phenopacket.json`). **Its
  `case_state.json`/`report.json` are not included** — for this case they're ~65MB each,
  because the full internal pipeline state includes the entire annotated-variant list
  from the whole-exome VCF. That's over GitHub's file-size warning threshold and not
  something that belongs permanently in git history. Regenerate them locally if you need
  them:

  ```bash
  python run_generesolver.py --example vcf_lirical
  ```
