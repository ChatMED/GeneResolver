# GeneResolver

Etiology-aware gene prioritization for rare disease. GeneResolver takes a clinical note and/or a
VCF, first classifies the case as single-gene, chromosomal, multifactorial or non-genetic, and
ranks candidate genes only for single-gene cases, fusing phenotype, variant (VEP + ACMG/AMP),
gene–disease, pathway, constraint, literature and GWAS evidence. It is a research prototype:
every output requires review by a qualified clinician.

## Architecture

A LangGraph orchestrator runs a fixed sequence of steps and routes each case by its etiology
class. **Agents** are the steps that call the language model (clinical synthesis, etiology
classification, branch reasoning, differential reasoning, mechanism, report generation).
**Components** are deterministic (de-identification, FastHPOCR phenotype extraction, VCF
parsing, VEP annotation, ACMG/AMP evaluation, retrieval, scoring, export); disease and phenotype
extraction also call MedGemma for present/absent/uncertain status. The HPO profile is the
FastHPOCR output (or the coded HPO terms supplied); the clinical synthesizer proposes syndrome
hypotheses. The only language
model is **MedGemma 27B IT**, served on your own infrastructure.

## Install

```bash
pip install -r requirements.txt
python setup_quickstart.py --with-vep-cache    # reference data (+ ~23 GB VEP 116 cache and Docker image)
export GENERESOLVER_MEDGEMMA_BASE_URL=http://localhost:8000/v1   # your OpenAI-compatible MedGemma server
export GENERESOLVER_LOCAL_MODE=1                                 # local Docker VEP, no remote variant lookups
```

Docker is required for VCF annotation. `.env.example` lists the settings; a project `.env` is
loaded automatically.

### Docker images

| Use | Image | Set in |
|---|---|---|
| Variant annotation (VEP 116, offline, GRCh37/38 cache) | [`ensemblorg/ensembl-vep:release_116.0`](https://hub.docker.com/r/ensemblorg/ensembl-vep/tags?name=release_116.0) | `GENERESOLVER_VEP_DOCKER_IMAGE` (`generesolver/config.py`) |
| gnomAD popmax lookup | [`brentp/slivar`](https://hub.docker.com/r/brentp/slivar) | `generesolver/integrations/gnomad_popmax_local.py` |
| phyloP conservation (bigWig) | [`quay.io/biocontainers/pybigwig:0.3.25--py311h4721021_1`](https://quay.io/repository/biocontainers/pybigwig) | `generesolver/integrations/conservation_local.py` |

```bash
docker pull ensemblorg/ensembl-vep:release_116.0
```

The VEP release in the image is checked against the cache version before annotation, and each run
records the image name and ID in its manifest. The paper's Table 10 runs used
`ensemblorg/ensembl-vep:latest`, built 2026-06-10 (image ID
`sha256:f354dd8d09073e4d943acbbd02f5eb234a9d9e9d444371c1c349910f2123de11`, VEP release 116, cache
`116_GRCh37`).

## Run

```bash
python run_generesolver.py --example single_gene     # phenotype only
python run_generesolver.py --example vcf_lirical     # phenotype + exome VCF
python run_generesolver.py --note-file note.txt --vcf sample.vcf.gz --sex female --family-history "..."
```

The mode is chosen from the inputs:

| Mode | Inputs | Output |
|---|---|---|
| Phenotype only | `--note` / `--note-file` | etiology class; ranked genes for single-gene cases |
| Phenotype + VCF | note + `--vcf` or `--dragen-dir` | as above, with variant evidence and provisional ACMG/AMP classes |
| Variant interpretation | `--vcf` / `--dragen-dir` (or `--candidate-variant`) without a note; optional `--sex`, `--family-history`, `--consanguinity` | ranked genes from variant evidence only (phenotype term 0, score ≤ 0.70) with ACMG/AMP classes |
| Gene–disease query | `--candidate-gene` without a VCF | gene-level evidence summary |
| Locus lookup | `--rsid` | ClinVar records for the rsID (NCBI E-utilities) |

Only the first two modes were evaluated in the paper. The inheritance model is inferred from the
family history (autosomal recessive is assumed when consanguinity is reported and no model can
be inferred); it cannot be set directly. Results go to `outputs/<case_id>/`: `report.md` (readable
report), `gene_ranking.csv`, `diagnostic_report.fhir.json`, `phenopacket.json`, and
`case_state.json` / `report.json` (full state with provenance and step status). `outputs/`
contains the reports of the bundled examples.

## Reproduce the paper

```bash
python setup_paper_datasets.py        # GSC+, PhEval and LIRICAL corpora
python reproduce_paper_results.py     # all evaluations -> Tables 4-10
python reproduce_paper_results.py --table 8          # one table (results in eval_runs/paper/)
python reproduce_paper_results.py --tables-only      # rebuild the tables from saved results, no runs
```

See `evaluation/README.md`.

## Privacy

Clinical text and VCFs are processed on your machine: de-identification, MedGemma inference
and VEP annotation run locally, and variant coordinates, alleles and genotypes are never sent
out. Evidence retrieval queries public services (PubMed, Open Targets, UniProt, Reactome, GWAS
Catalog) with gene symbols; Open Targets also receives disease names extracted from the
narrative. Exports contain the original note. See `docs/DATA_LOCALITY.md`.

## Documentation

- `docs/VARIANT_AND_RANKING.md`: VEP configuration, filtering, ranking equations and parameters
- `docs/ACMG_RULES.md`: ACMG/AMP criteria as implemented, and validation against ClinVar
- `docs/DATA_LOCALITY.md`: what leaves the machine and what is stored
- `docs/LLM_PROMPTS.md`: every MedGemma prompt and decoding setting
- `data_creation/README.md`: how the evaluation datasets were built

License: see `LICENSE`.
