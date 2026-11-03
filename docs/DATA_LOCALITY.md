# Data locality, storage and retention

Checked against the code and, for Table 10, against what the evaluation driver runs
(`orchestrator` steps up to `acmg_rerank`, `GENERESOLVER_LOCAL_MODE=1`). Nothing below is a
guarantee enforced by the code unless it says "gated".

## 1. What leaves the machine

| Destination | Component | What is sent | Gated by `LOCAL_MODE` | Called in Table 10 |
|---|---|---|---|---|
| Ensembl VEP REST | `vep_api` | variant coordinates + alleles | **yes** (local Docker VEP, `--offline`) | no |
| gnomAD GraphQL | `variant_annotator` → `gnomad_api` | variant coordinates + alleles | **yes** | no |
| NCBI ClinVar E-utilities | `locus_lookup` → `clinvar_api` | a user-supplied rsID | no | no (rsID lookup mode only) |
| PubMed E-utilities | `literature` | candidate **gene symbols** | no | yes |
| Europe PMC / PubMed | `spine`, `class_retrieval`, `hpo_differential` | phenotype/disease terms | **yes** | no |
| Open Targets | `bio_integrator` (genes), `gene_aggregator` (disease names from the narrative) | gene symbols; disease labels | no | yes (genes; no narrative in Table 10) |
| UniProt REST | `bio_integrator`, ACMG PM1 | gene symbols | no | yes |
| Reactome | `bio_integrator` | gene symbols / UniProt IDs | no | yes |
| GWAS Catalog | `gwas_lookup` | gene symbols | no | yes |
| PubCaseFinder | `hpo_differential` | HPO IDs | off by default and **gated** | no |
| MedGemma server | all LLM steps | prompts incl. the de-identified note | the only backend; runs at `GENERESOLVER_MEDGEMMA_BASE_URL` on your infrastructure | no LLM calls in Tables 9–10 (coded HPO input, no family history, etiology routing off) |

**Variant coordinates, alleles and genotypes are not sent anywhere in the evaluated
configuration.** VEP and slivar run in local Docker containers against local files. The
containers are not started with `--network none`, so isolation rests on VEP's `--offline`
mode and slivar's file-only operation, not on a network policy.

**Gene symbols are sent.** The candidate gene list for a case partly reflects the patient's
rare variants (genes enter the pool through them), so the set of genes queried is
patient-derived information, although it contains no coordinates, alleles or genotypes.
Responses are cached on disk (`outputs/_*_cache`), so a gene is queried once per cache.

## 2. What is written to disk

| Location | Content | Retention |
|---|---|---|
| `outputs/<case>/case_state.json`, `report.json` | full `CaseState`: **raw `input.clinical_note` and `input.family_history`** next to the de-identified text, `vcf_path`, demographics, all retained variants with genotype/zygosity/depth | kept until deleted by hand |
| `outputs/<case>/report.md`, `*.fhir.json`, `phenopacket.json`, `gene_ranking.csv` | derived report, ranked genes/variants, HPO terms, sex/age | kept |
| `data/vep_cache/vep_annotations_v*.json.gz`, `eval_runs/*/vep_cache_local_v2.json.gz` | every annotated variant (coordinates + alleles + annotation), pooled across all cases, no sample ID or genotype | kept indefinitely |
| `%TEMP%/generesolver_vep_*`, `%TEMP%/generesolver_slivar_*` | `in.vcf`/`out.vcf` with the variants of each annotation batch | **never deleted** (64 such directories present on 2026-11-01) |
| `outputs/_llm_cache/` | LLM responses keyed by a prompt hash (no prompt text); responses can contain extracted clinical content | kept |
| `outputs/_*_cache/` | external API responses keyed by gene/disease query | kept |

The code configures **no encryption, access control or retention/deletion policy**. Access
is whatever the operating-system account and disk permissions allow. `outputs/` is
git-ignored except three demo cases (public/synthetic), which are tracked.

## 3. De-identification and stored outputs

`deidentifier` writes the de-identified text to `state.clinical_note_deidentified` and
de-identifies `state.family_history`, and all downstream LLM calls use those. It does **not**
overwrite `state.input.clinical_note` or `state.input.family_history`, and the exporter
serialises the whole state (`state.model_dump()`). Verified on
`outputs/DEMO_CASE_SINGLE_GENE`: both `case_state.json` and `report.json` contain the raw note
(1,175 characters) as well as the de-identified one (1,187). **De-identification protects what
is sent to models and services, not what is stored.**

## 4. Data used in the evaluations

Table 10 inputs were published LIRICAL phenopackets (HPO terms, no narrative) and spiked VCFs
built on the public PhEval hg19 template exome (sample `NIST7035`, a public reference
sample). Table 4 used synthetic notes, Table 5 published GSC+ abstracts, Tables 6–7 vignettes
derived from published case reports, Table 8 public ClinVar variants. No private patient
data was processed.

## 5. Known gaps (deployment responsibilities)

1. Exclude or redact `input.clinical_note` / `input.family_history` in exported state, or write
   raw input only to an explicitly opted-in file.
2. Delete VEP/slivar temp directories after parsing (`shutil.rmtree` in a `finally`).
3. Run VEP/slivar containers with `--network none`.
4. Gate gene-symbol queries (PubMed, Open Targets, UniProt, Reactome, GWAS) behind an
   explicit "offline evidence" switch, or document them as allowed outbound traffic.
5. Optional per-case cache scope or a purge command for the pooled VEP/API caches.
