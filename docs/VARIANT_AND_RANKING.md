# Variant annotation, filtering and ranking

Configuration of the evaluated setup (Table 10, `GENERESOLVER_LOCAL_MODE=1`, GRCh37) and the
order of the ranking steps. All values are set in code; the file named in each row is the source.

## 1. Annotation

| Item | Value | Source |
|---|---|---|
| Annotator | Ensembl VEP release 116 in Docker, offline. Default image `ensemblorg/ensembl-vep:release_116.0`; the revision Table 10 runs used image `sha256:f354dd8d0907…` (tag `latest`, built 2026-06-10). The VEP release is checked against the cache version before annotation | `config.VEP_LOCAL_DOCKER_IMAGE` |
| Cache | VEP cache 116, GRCh37.p13, GENCODE 19 (GRCh38 also supported) | `data/vep_cache/homo_sapiens/116_GRCh37/` |
| Cache contents (`info.txt`) | gnomAD v4.1 exomes and genomes, dbSNP 156, ClinVar 202306, 1000 Genomes phase 3, COSMIC 98, HGMD-public 20204, SIFT 5.2.2, PolyPhen 2.2.2 | cache `info.txt` |
| VEP options | `--offline --cache --vcf --pick --symbol --mane --sift b --polyphen b --af_gnomad --check_existing --max_af --domains` (default pick order: one consequence and gene per allele) | `vep_api._LOCAL_VEP_FLAGS` |
| VEP plugins | none; REVEL and SpliceAI are added outside VEP | — |
| REVEL | v1.3 local table (missense) | `scripts/download_revel.py` |
| SpliceAI | maximum delta score via `slivar gnotate` with `spliceai.hg37.zip` (image `brentp/slivar`, GRCh37 only) | `gnomad_popmax_local` |
| ClinVar | NCBI ClinVar VCF, release recorded with SHA-256 in `data/clinvar/*.source.json` (revision runs: 2026-10-19). VEP's ClinVar 202306 is used only for variants absent from that VCF. Pin a release with `scripts/download_clinvar.py --release YYYYMMDD` | `variant_annotator` |
| Other resources | gnomAD v4.1 constraint (PP2, constraint channel), MANE v1.5 summary (PVS1 transcript gate), UCSC phyloP100way and simpleRepeat hg19 (BP7, PM4/BP3) | `scripts/download_*.py` |
| Not available from the cache | gnomAD AC, AN, nhomalt and grpmax, so VEP "popmax" is `MAX_AF` and site coverage is not checked | `info.txt` |

Not produced: full HGVS names (VEP runs without `--hgvs`/`--fasta`); for missense variants the
protein change is rebuilt from `Protein_position` and `Amino_acids`. On GRCh37 the MANE gate
matches the Ensembl transcript ID (without version) to the GRCh38 MANE summary.

**Provenance.** Each case's `variant_annotator` provenance records the VEP image and ID, cache
version and `info.txt` sources, the flag list, VEP's `##VEP` header, a configuration fingerprint,
the filter profile and thresholds, per-step filter counts, VCF-loading drop counts and the
ClinVar/REVEL/MANE/SpliceAI file versions. The VEP result cache is keyed by the fingerprint, so
annotations from another configuration are never reused. Table 10 runs write `run_manifest.json`
(code hash, git commit, VEP metadata, resource versions) and refuse to mix code versions.

## 2. Quality control and filtering

1. **Assembly** from the VCF header (`##contig` length of chromosome 1, then assembly keywords);
   annotation stops if it cannot be determined. GRCh37-only resources are not used for GRCh38.
2. **Records**: `PASS` only, primary contigs (1–22, X, Y, MT), QUAL ≥ 20 (`GENERESOLVER_MIN_QUAL`),
   multi-allelic records split and trimmed. Dropped and counted: no ALT allele in the first sample,
   REF or ALT > 50 bp, symbolic ALTs (SV/CNV). Only the first sample is analysed. GQ/DP filters
   exist (`GENERESOLVER_MIN_GQ`, `GENERESOLVER_MIN_DP`) but are off: the spiked causal records carry
   `GT` only, so they would remove background calls only. All eligible alleles are annotated
   (`GENERESOLVER_VCF_MAX_VARIANTS=0`). No reference FASTA was configured, so no left-alignment.
3. **Variant retention (OR)**: a variant is kept if it is **rare or relevant**, and dropped only if
   it is common and not relevant.
   - *Rare*: overall gnomAD v4.1 exome AF (`gnomADe_AF`) **< 0.01**, or no gnomAD exome record.
   - *Relevant*: VEP IMPACT HIGH (transcript_ablation, splice_acceptor/donor, stop_gained,
     frameshift, stop_lost, start_lost, transcript_amplification) or MODERATE (inframe
     insertion/deletion, missense, protein_altering), plus `splice_region_variant`.
4. **Variant-driven candidate genes (AND)**: a gene enters the candidate pool through its variants
   only with ≥ 1 variant that is HIGH/MODERATE **and** rare (`gene_aggregator`).
5. **Rationale**: 1 % is the conventional polymorphism cut-off and lies below the ACMG BA1
   threshold (5 %); the filter removes only variants that are both common and non-protein-altering
   and leaves pathogenicity to the ACMG engine. A missing frequency counts as rare, so this
   filter never confirms coverage.

**Revised profile** (`GENERESOLVER_VARIANT_FILTER_PROFILE=revised`; not used for the reported
results). Same 1 % threshold and OR/AND structure, but *rare* uses the highest continental gnomAD
v4.1 exome AF (AFR, AMR, EAS, NFE, SAS), `splice_region_variant` counts only with SpliceAI ≥ 0.2,
and variants passing on a missing frequency are flagged. Inheritance-specific thresholds are not
applied. On the shared template exome it reduces the variants passing the gene gate from 1,567 to
1,294; causal records passing go from 440 to 441 of 446 (none lost). Five causal records fail both
profiles: two common missense alleles (SPTA1 AF 0.27, NPHS1 AF 0.31), two intronic (WNK1, FGFR3)
and one 3′UTR (P3H1). (Supplementary analysis, run locally; not part of the repository.)

## 3. Ranking sequence

1. **Candidate genes** (`gene_aggregator`): genes of the top phenotype-scored diseases, genes with
   qualifying variants, Orphanet genes of diseases named in the narrative, Open Targets genes and
   the phenotype–disease shortlist (caps in `config.py`).
2. **Disease phenotype score** (`lr_score`, Eq. 1): sum of log likelihood ratios over the observed
   and absent HPO terms (unmatched term LR 0.2; ancestor-only annotation 0.5 × LR; excluded term
   0.001; absent terms (1 − f)/(1 − b), floored at 0.05).
3. **Disease → gene**: the gene takes the **maximum** score over its diseases (HPO-A, Orphanet
   fallback); the best disease is recorded.
4. **Variant triage** (Eq. 2): impact weight (HIGH 0.85, MODERATE 0.65, LOW 0.45, MODIFIER 0.30)
   × rarity weight (AF < 10⁻⁴ 1.00, < 10⁻³ 0.90, < 10⁻² 0.75, else 0.55), capped at 0.75; a ClinVar
   P/LP variant scores 0.95 (`GENERESOLVER_CLINVAR_TRIAGE=0` disables this). The gene takes its best variant.
5. **Fusion** (`gene_ranker`, Eq. 3): phenotype scores min-max normalized within the case; weights
   phenotype 0.30, variant 0.26, gene–disease 0.17, pathway 0.09, constraint 0.08, literature 0.07,
   GWAS 0.03. Without any variant signal: phenotype 0.80, constraint 0.10, gene–disease 0.05,
   pathway 0.05. The constraint weight is 0 for an inferred recessive model. With no phenotype input
   (variant interpretation mode) the phenotype term is 0 and the weights are not renormalized, so
   the score is at most 0.70.
6. **Bounded adjustments**: negated-phenotype penalty up to −0.05; −0.20 for Y-linked and −0.05 for
   X-linked recessive genes in female patients; +0.10 × disease-match strength. Clamped to [0, 1].
   Confidence: high ≥ 0.70, moderate ≥ 0.50, low ≥ 0.30.
7. **ACMG re-rank** (`acmg_rerank`): when the gene's best disease context is unambiguous, the ACMG
   class score (P 1.0, LP 0.8, VUS 0.5, LB 0.2, B 0.0) replaces the triage score, and the final
   score changes by (new − old) × 0.26. See `docs/ACMG_RULES.md`.
8. **LLM tie-break** (full runs only; the benchmarks stop at step 7): ±0.01 at most, only when
   grounded in the case facts, so it can reorder only genes within 0.02 of each other. Every
   adjustment is stored on the gene.

All weights, tiers and adjustment bounds were set manually during development; none were fitted.
ACMG thresholds come from the cited ACMG/ClinGen publications.
