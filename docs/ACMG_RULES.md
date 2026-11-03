# ACMG/AMP evidence engine: criteria, thresholds and combination

Implementation: `generesolver/components/acmg_engine.py` (ruleset `ACMG_AMP_2015` with ClinGen SVI
refinements). Every classification is labelled `provisional` and `requires_expert_review=True`.

## Status values

Each criterion returns one of:

| Status | Meaning | Effect on the class |
|---|---|---|
| `met` | Evidence present and the rule satisfied, at the stated strength | counted |
| `not_met` | Evidence present and the rule not satisfied | not counted |
| `not_applicable` | Rule does not apply to this variant type (e.g. PP2 for a frameshift) | not counted |
| `not_evaluated` | Evidence needed for the rule is missing; the missing input is listed in `missing_requirements` | not counted, **never treated as met** |

## Gene-disease context

A variant is assessed only inside a gene-disease-transcript context
(`variant_disease_context_builder.py`). Diseases come from Orphanet gene associations. The
disease mechanism is `loss_of_function` / `gain_of_function` from the Orphanet association type,
otherwise the ClinGen haploinsufficiency curation (HI score 3 = sufficient evidence → loss of
function; `scripts/download_clingen_haploinsufficiency.py`). With no gene-disease pair or no transcript the
result is `not_classifiable`.

## Criteria

| Code | Strength | Rule as implemented | Evidence source | When `not_evaluated` |
|---|---|---|---|---|
| PVS1 | Supporting→Very strong | Adapted from the ClinGen PVS1 decision tree (Abou Tayoun 2018), on GENCODE v19 structure; the frequency of LoF variants in the affected exon is not assessed. Stop-gained/frameshift: NMD predicted (> 50 nt upstream of the last exon-exon junction) → Very strong; NMD escape → Strong if the removed part overlaps a UniProt functional region or is > 10% of the protein, otherwise Moderate. Canonical splice (±1/2): the adjacent exon is assumed skipped; frameshifting skip with NMD → Very strong, otherwise as above; in-frame skip → Strong (critical region or > 10%) or Moderate. Start-lost → Supporting. Transcript ablation → Very strong. Transcript: MANE Select (VEP's, or the gene's MANE transcript when the variant lies in it), else a CCDS "basic" transcript one level lower. Mechanism, in order: Orphanet LoF or ClinGen HI 3; ClinVar gene-level evidence, i.e. ≥ 5 P/LP truncating (nonsense/frameshift) ClinVar variants in the gene besides the assessed one (v3; the assessed variant's own record cannot establish it); otherwise gnomAD v4.1 LOEUF < 0.45 (gnomAD's suggested threshold for Mendelian interpretation, ~15% most constrained genes) or recessive inheritance, one level lower | VEP consequence; GENCODE v19; MANE; UniProt; Orphanet; ClinGen HI; ClinVar variant_summary; gnomAD v4.1 constraint | gain-of-function or no LoF evidence; no MANE/CCDS transcript; skipped exon non-coding or first exon. *Cryptic splice sites and the new stop of a frameshift are not modelled; start-lost does not use upstream ClinVar variants (circular).* |
| PS1 | Strong | missense; the same amino-acid change is reported P/LP at this residue | ClinVar variant_summary + UniProt humsavar | reference data missing; protein change unknown. *Caveat: cannot tell whether the matching record is this same nucleotide variant.* |
| PS2 | Strong | narrative states parental/trio testing confirmed de novo | LLM-extracted family evidence (local model) | no confirmed-de-novo statement |
| PM1 | Moderate | missense inside a UniProt functional region (VEP domain as fallback) with ≥2 other known-P/LP missense positions within ±5 residues | UniProt features, ClinVar/humsavar | no region annotation, no reference data |
| PM2 | Supporting | gnomAD popmax AF < 1×10⁻⁵, **or the variant is absent from gnomAD** (site coverage not confirmed, disclosed as a warning) | VEP `--af_gnomad` / gnomAD API | frequency source unavailable or failed |
| PM3 | Supporting→Very strong | recessive disorders; ClinGen SVI points: homozygous 0.5 (0.25 with consanguinity), second P/LP allele phase-unknown 0.5/0.25; tiers 0.5/1/2/4 | zygosity, consanguinity | not recessive → `not_applicable`; *the second-allele branch is not populated by the current pipeline, so only the homozygous branch can fire* |
| PM4 | Moderate | stop-lost, or in-frame indel outside a UCSC simpleRepeat region; not applied when PVS1 is met | VEP, UCSC simpleRepeat | repeat track missing |
| PM5 | Moderate | missense; a different P/LP missense change exists at the same residue | ClinVar/humsavar | reference data missing |
| PM6 | Moderate | narrative states de novo without confirmation; not applied when PS2 is met | LLM-extracted family evidence | no de-novo statement |
| PP1 | Supporting | narrative states segregation with disease (no meiosis-based upgrade) | LLM-extracted family evidence | no segregation statement |
| PP2 | Supporting | missense in a gene with gnomAD missense Z ≥ 3.09 | gnomAD v4.1 constraint | no constraint data |
| PP3 | Supporting / Moderate / Strong | missense REVEL ≥ 0.644 / 0.773 / 0.932 (Pejaver 2022); splice-relevant SpliceAI Δ ≥ 0.20 (Supporting only) | REVEL, SpliceAI | no calibrated score (SIFT/PolyPhen deliberately not used) |
| PP4 | Supporting | phenotype-match score ≥ 0.5 for the gene-disease pair | HPO likelihood-ratio disease score, min-max normalized within the case | no phenotype score. *Because the score is normalized within each case, the best-matching disease always scores 1.0, so this is weaker than the guideline's "highly specific phenotype".* |
| BA1 | Stand-alone | AF > 5% in a general continental population of ≥ 2,000 alleles (Ghosh et al. 2018): the highest gnomAD v4.1 exome AF among AFR, AMR, EAS, NFE and SAS. The guideline's minimum of 2,000 observed alleles applies at the variant site in that population; the VEP cache has no per-site allele number, so this is **not verified** (a deviation from the full recommendation) unless the optional per-site index below is installed. Founder/small groups (ASJ, FIN, MID, remaining) and 1000 Genomes are excluded. The variant must not be on the ClinGen SVI BA1 exception list (9 variants in ACAD9, GJB2, HFE ×2, MEFV ×2, PIBF1, ACADS, BTD; matched by gene + protein change, `clingen_ba1_exceptions.py`); an exception-list variant is `not_met` | gnomAD v4.1 exomes (VEP cache 116). Optional: a per-site AF_popmax/AN_popmax pair from gnomAD v2.1.1 (`scripts/build_gnomad_an_index.py`) is used first when present | no continental-population AF (e.g. only excluded populations or 1000 Genomes exceed 5%). *Per-site allele number is not checked (not in the VEP cache); a poorly covered site could in principle fall below 2,000 alleles.* |
| BS1 | — | always `not_evaluated` (needs disease prevalence/penetrance) | — | always |
| BS2 | Strong | narrative reports an unaffected **adult** relative with the same genotype | LLM-extracted family evidence | no statement, or adult status not stated |
| BS4 | Strong | narrative explicitly states non-segregation | LLM-extracted family evidence | no segregation statement |
| BP1 | Supporting | missense in a gene where ≥ 90% of ≥ 5 classified ClinVar P/LP variants are truncating | ClinVar variant_summary | too few classified variants |
| BP3 | Supporting | in-frame indel inside a UCSC simpleRepeat region | UCSC simpleRepeat | repeat track missing |
| BP4 | Supporting→Very strong | missense REVEL ≤ 0.290 / 0.183 / 0.016 / 0.003; splice-relevant SpliceAI Δ ≤ 0.10; synonymous and intronic variants SpliceAI Δ ≤ 0.10 (Supporting; ClinGen SVI splicing, Walker 2023, together with BP7 for synonymous). Synonymous/intronic SpliceAI Δ ≥ 0.20 gives PP3 Supporting | REVEL, SpliceAI | no calibrated score; UTR/promoter variants are not given splicing BP4 |
| BP7 | Supporting | synonymous, not splice-relevant, SpliceAI ≤ 0.10 and phyloP ≤ 0 | SpliceAI, UCSC phyloP100way | conservation track missing |
| PP5, BP6 | — | `not_applicable` (deprecated by ClinGen SVI 2018) | — | — |
| PS3, PS4, BS3, BP2, BP5 | — | `not_evaluated` (not implemented: functional, case-control, phase and alternative-diagnosis data) | — | always |

## Combination

* Points model of Tavtigian et al. 2020: Supporting 1, Moderate 2, Strong 4, Very strong 8; benign
  criteria count negative.
* Class: ≥ 10 pathogenic; 6–9 likely pathogenic; 0–5 VUS; −1 to −6 likely benign; ≤ −7 benign.
* BA1 met with no Strong/Very-strong pathogenic criterion → benign (stand-alone).
* Two criteria that cite the same evidence item (and PVS1 + PM4) are not both counted; the
  stronger one is kept and the other is listed in `excluded_dependent_criteria`.
* If pathogenic and benign criteria are both met, the case is flagged `conflict_status=unresolved`
  and needs expert review. A P/LP class with BA1 counted is downgraded to VUS.

## Use in ranking

`acmg_rerank.py` maps the class to a variant score (P 1.0, LP 0.8, VUS 0.5, LB 0.2, B 0.0). The
score replaces the triage score only when the gene's best-matching disease context is unambiguous
and at least one criterion was counted. The gene's final score changes by
Δ = (new − old variant score) × 0.26.

## Why a separate implementation

Available implementations differ substantially in which criteria they automate, how they
evaluate them and which data they use (Costa et al., Brief Bioinform 2025,
doi:10.1093/bib/bbaf545). The pipeline needed four properties that no single available tool
provides together:

1. **Offline.** The best-performing tools in that review (Franklin, VarSome, GeneBe) are web
   services, so a hospital's patient variants would leave the server; InterVar runs locally but
   needs a licensed ANNOVAR installation.
2. **Case context.** PM3, PP4, PS2, PM6, PP1, BS2 and BS4 need the case's inheritance, phenotype
   match and family history, which the pipeline already holds; variant-only tools cannot see them
   (PP1/BS4 are automated only by ELLA and VarSome in the review).
3. **Machine-readable evidence for the ranker.** Every criterion returns status, strength, points,
   evidence items and missing inputs; the class feeds the gene score with the same disease context
   as the ranker.
4. **Current ClinGen SVI rules.** Points model (Tavtigian 2020), PM2 at Supporting, PP3/BP4 from
   calibrated REVEL thresholds (Pejaver 2022), BA1 from continental populations with the exception
   list, PP5/BP6 not used. GeneBe uses PP5 from ClinVar and scores PM2 at
   Moderate; InterVar follows the original 2015 combining rules.

20 of the 28 criteria have a rule (71%); 5 of those depend on narrative family evidence and are
not validated. Unevaluable criteria are `not_evaluated`, never met, so the engine leans towards VUS
rather than over-calling, the failure mode the review reports for most web tools (> 50 % of expert
VUS called P/LP).

## Engine versions

| Version | Change | PVS1 rule id |
|---|---|---|
| v1 | rules as in the table above, PVS1 without the decision tree (no canonical-splice PVS1) | `pvs1_v6.0` |
| v2 | ClinGen PVS1 decision tree on GENCODE v19; SpliceAI BP4/PP3 for synonymous and intronic variants (Walker 2023) | `pvs1_v7.0_clingen_tree` |
| v3 | v2 + LoF mechanism for PVS1 from ClinVar gene-level evidence (≥ 5 other P/LP truncating variants in the gene) | `pvs1_v8.0_clingen_tree_clinvar_lof` |
| v3.1 (current) | v3 with the PVS1 LOEUF fallback at < 0.45 (gnomAD's suggested threshold for Mendelian interpretation) instead of < 0.6; validation results identical to v3 | `pvs1_v8.1_clingen_tree_clinvar_lof` |

v2 and v3 need `python scripts/download_gencode_gtf.py` (run by `setup_quickstart.py`); without the
GENCODE index PVS1 falls back to the v1 rule.

**Table 10 was not rerun with the revised engine.** The published Table 10 run (August 2026) used an
earlier engine whose exact source was not preserved (`acmg_engine.py` was modified during that
run). Do not attribute the Table 10 ranking to the validated v3 engine.

## Validation against ClinVar

ClinVar ≥ 2-star Pathogenic, Likely pathogenic, Likely benign and Benign coding/splice variants
(ClinVar 2026-10-19), annotated exactly as in Table 10 (GRCh37, local VEP 116). Only variant-level
evidence is available (heterozygous, no phenotype or family data). Direction is correct when a
P/LP variant is called P/LP or a B/LB variant B/LB; VUS counts as not correct.

**Development set** (`data_creation/acmg_validation/dev_400.json`; `python -m evaluation.table8_acmg_validation.acmg_validation_balanced`): seeded sample
of 100 variants per class. v2 and v3 were designed on these variants, so their figures are
optimistic.

| | v1 | v3.1 |
|---|---|---|
| P/LP called P/LP (n = 200) | 41.0% | 72.0% |
| B/LB called B/LB (n = 200) | 62.5% | 87.0% |
| Overall direction correct (n = 400) | 51.8% | 79.5% |
| Opposite-direction calls | 0 | 0 |

**Held-out set** (Table 8; `data_creation/acmg_validation/heldout_214.json`; `python -m evaluation.table8_acmg_validation.eval_acmg_validation sample|score|genebe|report`):
214 variants not in the development set (81 P, 6 LP, 27 LB, 100 B; seed 20261102), drawn from the
ClinVar variants annotated in a Table 10 run, classified with v3 and v3.1 (identical results) and with GeneBe (public API;
public ClinVar variants only, never patient data). Classes as returned by each tool, all criteria;
Wilson 95% CIs. Output: `evaluation/paper_results/acmg_heldout/comparison.md`.

| | GeneResolver v3.1 | GeneBe |
|---|---|---|
| P/LP called P/LP (n = 87) | 67, 77.0% (67.1–84.6) | 86, 98.9% (93.8–99.8) |
| B/LB called B/LB (n = 127) | 120, 94.5% (89.1–97.3) | 127, 100% (97.1–100) |
| Overall direction correct (n = 214) | 187, 87.4% (82.3–91.2) | 213, 99.5% (97.4–99.9) |
| Opposite-direction calls | 0 | 0 |
| VUS / not classified | 27 | 1 |
| Exact 5-class agreement | 60.7% | 85.0% |

Direction agreement between the tools 87.9% (Cohen's κ 0.775). McNemar on correct direction: only
GeneResolver correct 0, only GeneBe correct 26 (exact p < 0.001); all 26 were VUS calls by
GeneResolver. By consequence (GeneResolver / GeneBe correct): synonymous 78/80 vs 80/80, missense
66/73 vs 72/73, stop-gained 15/24 vs 24/24, frameshift 15/20 vs 20/20, canonical splice 5/6 vs 6/6.

Both tools use ClinVar-derived evidence, so agreement with ClinVar is an upper bound, not an
independent accuracy estimate: GeneBe applied PP5 to 87/87 P/LP and BP6 to 126/127 B/LB variants
(criteria ClinGen SVI no longer recommends); GeneResolver's PS1, PM5, PM1 and the v3 PVS1 mechanism
draw on ClinVar. The result files also contain a scoring with these GeneResolver criteria removed.

Not validated: the narrative-derived criteria (PS2, PM6, PP1, BS2, BS4), PM3's second-allele
branch and PP4.
