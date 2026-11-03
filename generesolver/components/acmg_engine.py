from __future__ import annotations

import re

from generesolver.integrations import (
    clingen_ba1_exceptions, gencode_transcripts, gnomad_constraint, known_pathogenic_residues,
    mane_local, ucsc_repeats, uniprot_api, vep_api,
)
from generesolver.schemas.acmg_assessment import (
    ACMGAssessment,
    ACMGComparisonResult,
    ClinVarExternalAnnotation,
    CombinationResult,
    Conflict,
    CriterionResult,
    CriterionStatus,
    EvidenceDirection,
    EvidenceItem,
    EvidenceStrength,
)
from generesolver.schemas.variant import AnnotatedVariant
from generesolver.schemas.variant_disease_context import VariantDiseaseContext

_RULESET_FOUNDATION = "ACMG_AMP_2015"
_RULESET_GENERAL_REFINEMENTS = "ClinGen"

_TRUNCATING_NMD_CONSEQUENCES = {"stop_gained", "frameshift_variant"}
_TRANSCRIPT_ABLATION_CONSEQUENCE = "transcript_ablation"
_START_LOST_CONSEQUENCE = "start_lost"
_LOF_CONSEQUENCES = (
    _TRUNCATING_NMD_CONSEQUENCES
    | {_TRANSCRIPT_ABLATION_CONSEQUENCE, _START_LOST_CONSEQUENCE}
    | {"splice_acceptor_variant", "splice_donor_variant"}
)
_PM4_CONSEQUENCES = {"inframe_insertion", "inframe_deletion", "stop_lost"}
_SYNONYMOUS_CONSEQUENCES = {"synonymous_variant", "stop_retained_variant"}
# Scope of the SVI splicing BP4/PP3 rule: synonymous and intronic only. Promoter/UTR variants act
# through mechanisms SpliceAI does not model, so BP4 there would be unsafe.
_NON_PROTEIN_CHANGING = _SYNONYMOUS_CONSEQUENCES | {"intron_variant"}

_BA1_THRESHOLD = 0.05
# Extra variant_ids to treat as BA1 exceptions, on top of the ClinGen SVI list
# (clingen_ba1_exceptions.EXCEPTIONS, matched by gene + protein change).
_BA1_EXCEPTION_VARIANTS: set[str] = set()
_BA1_EXCEPTION_LIST_AVAILABLE = True

_MIN_ADEQUATE_ALLELE_NUMBER = 2000

_PM2_RESEARCH_THRESHOLD = 1e-5

_REVEL_PP3_TIERS = [
    (0.932, EvidenceStrength.STRONG),
    (0.773, EvidenceStrength.MODERATE),
    (0.644, EvidenceStrength.SUPPORTING),
]
_REVEL_BP4_TIERS = [
    (0.003, EvidenceStrength.VERY_STRONG),
    (0.016, EvidenceStrength.STRONG),
    (0.183, EvidenceStrength.MODERATE),
    (0.290, EvidenceStrength.SUPPORTING),
]

_PHYLOP_BP7_THRESHOLD = 0.0

_PM1_CLUSTER_WINDOW = 5
_PM1_CLUSTER_MIN_NEARBY = 2


def _uniprot_essential_region(gene: str, position: int) -> dict | None:
    for region in uniprot_api.get_functional_regions(gene):
        if region["start"] <= position <= region["end"]:
            return region
    return None

_PP2_MISSENSE_Z_THRESHOLD = 3.09

_PP4_PHENOTYPE_MATCH_THRESHOLD = 0.5

_SPLICEAI_PP3_THRESHOLD = 0.20
_SPLICEAI_BP4_THRESHOLD = 0.10
_CANONICAL_SPLICE_CONSEQUENCES = {"splice_acceptor_variant", "splice_donor_variant"}


def _gnomad_source_versions() -> dict[str, str]:
    return {"gnomad": "v4.1"}


def _vep_source_versions() -> dict[str, str]:
    cache_version = getattr(vep_api, "_VEP_CACHE_VERSION", None)
    return {"vep_cache_version": str(cache_version)} if cache_version else {}


def _repeat_overlap(chrom_pos: tuple[str, int], genome_build: object) -> bool | None:
    # The UCSC simpleRepeat track is hg19; never query it with GRCh38 coordinates.
    if str(getattr(genome_build, "value", genome_build)) != "GRCh37":
        return None
    return ucsc_repeats.in_repeat_region(*chrom_pos)


def _variant_chrom_pos(variant: AnnotatedVariant) -> tuple[str, int] | None:
    parts = variant.variant_id.split(":")
    if len(parts) < 2:
        return None
    try:
        return parts[0], int(parts[1])
    except ValueError:
        return None


def _result(
    code: str, status: CriterionStatus, direction: EvidenceDirection,
    rationale: str, *, strength: EvidenceStrength | None = None,
    evidence_items: list[EvidenceItem] | None = None,
    missing_requirements: list[str] | None = None,
    warnings: list[str] | None = None,
    evaluator_version: str = "unspecified_v1.0",
    source_versions: dict[str, str] | None = None,
) -> CriterionResult:
    return CriterionResult(
        code=code, status=status, direction=direction, strength=strength,
        points=None,
        rationale=rationale,
        evidence_items=evidence_items or [],
        missing_requirements=missing_requirements or [],
        warnings=warnings or [],
        evaluator_version=evaluator_version,
        source_versions=source_versions or {},
    )


def _not_evaluated(code: str, direction: EvidenceDirection, rationale: str,
                    missing: list[str], evaluator_version: str) -> CriterionResult:
    return _result(code, CriterionStatus.NOT_EVALUATED, direction, rationale,
                   missing_requirements=missing, evaluator_version=evaluator_version)


def _not_applicable(code: str, direction: EvidenceDirection, rationale: str,
                     evaluator_version: str = "unspecified_v1.0") -> CriterionResult:
    return _result(code, CriterionStatus.NOT_APPLICABLE, direction, rationale,
                    evaluator_version=evaluator_version)



def _pvs1_exon_evidence(variant: AnnotatedVariant) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"vep:{variant.variant_id}:consequence",
        evidence_type="molecular_consequence", source="VEP",
        value={"consequence": variant.consequence, "exon": variant.exon,
               "transcript": variant.transcript},
    )


_STRENGTH_ORDER = [EvidenceStrength.SUPPORTING, EvidenceStrength.MODERATE,
                   EvidenceStrength.STRONG, EvidenceStrength.VERY_STRONG]
# gnomAD v4 constraint: LOEUF upper bound below this marks a gene as LoF-intolerant.
_PVS1_LOEUF_THRESHOLD = 0.45  # gnomAD v4.1 suggestion for Mendelian interpretation (~15% most constrained)
_PVS1_MIN_REMOVED_FRACTION = 0.10
_RECESSIVE_FOR_PVS1 = {"autosomal_recessive", "x_linked_recessive"}
# PVS1 mechanism from ClinVar: other P/LP truncating variants required in the gene.
_PVS1_CLINVAR_MIN_LOF = 5
# Marks a PVS1 whose mechanism came from ClinVar, so ClinVar-blind scorings can drop it.
CLINVAR_GENE_LOF_MARKER = "LoF mechanism from ClinVar gene-level evidence"


def _downgrade(strength: EvidenceStrength, steps: int) -> EvidenceStrength:
    return _STRENGTH_ORDER[max(0, _STRENGTH_ORDER.index(strength) - steps)]


def _pvs1_mechanism(variant: AnnotatedVariant, context: VariantDiseaseContext) -> tuple[str | None, str]:
    """('established' | 'inferred' | None, explanation).

    Sources, in order: Orphanet association type / ClinGen HI 3; ClinVar gene-level evidence
    (ClinGen PVS1, Abou Tayoun 2018: LoF is a known mechanism when multiple pathogenic LoF
    variants are reported in the gene; at least _PVS1_CLINVAR_MIN_LOF others are required, so the
    assessed variant's own record cannot establish it); then gnomAD constraint or recessive
    inheritance (inferred, one level lower)."""
    if context.disease_mechanism == "loss_of_function":
        return "established", "LoF mechanism from Orphanet association type or ClinGen HI score 3"
    if context.disease_mechanism == "gain_of_function":
        return None, "disease mechanism is gain of function"
    gene = variant.gene or context.gene
    if gene and known_pathogenic_residues.is_available():
        n_lof = (known_pathogenic_residues.type_counts(gene) or {}).get("truncating", 0)
        # The count may include the assessed variant itself, so one is subtracted.
        if n_lof - 1 >= _PVS1_CLINVAR_MIN_LOF:
            return "established", (f"{CLINVAR_GENE_LOF_MARKER}: {n_lof} P/LP truncating variants in {gene} "
                                   f"(>= {_PVS1_CLINVAR_MIN_LOF} besides this one)")
    loeuf = (gnomad_constraint.get_constraint(variant.gene or context.gene) or {}).get("loeuf")
    if loeuf is not None and loeuf < _PVS1_LOEUF_THRESHOLD:
        return "inferred", f"gene is LoF-intolerant (gnomAD v4.1 LOEUF {loeuf:.2f} < {_PVS1_LOEUF_THRESHOLD})"
    if context.inheritance in _RECESSIVE_FOR_PVS1:
        return "inferred", f"recessive disorder ({context.inheritance}): biallelic LoF is the usual mechanism"
    return None, "no LoF mechanism from Orphanet, ClinGen HI or gnomAD constraint"


def _pvs1_transcript(variant: AnnotatedVariant):
    """(Transcript | None, relevance, explanation). Relevance: 'mane' | 'ccds' | None."""
    pos = (_variant_chrom_pos(variant) or (None, None))[1]
    if variant.mane_status in ("MANE_Select", "MANE_Plus_Clinical"):
        return gencode_transcripts.get(variant.transcript), "mane", "VEP transcript is MANE"
    mane_id = mane_local.mane_select_transcript(variant.gene)
    mane_tx = gencode_transcripts.get(mane_id)
    if mane_tx and pos is not None and (mane_tx.exon_index(pos) is not None
                                         or mane_tx.skipped_exon_for_splice_site(pos)):
        return mane_tx, "mane", f"variant also lies in the gene's MANE Select transcript {mane_id}"
    tx = gencode_transcripts.get(variant.transcript)
    if tx and tx.has_ccds and tx.is_basic:
        return tx, "ccds", f"transcript {tx.transcript_id} is not MANE but is a CCDS 'basic' transcript"
    return None, None, "transcript is neither MANE nor CCDS-supported"


def _pvs1_truncation_strength(tx, cds_pos: int, gene: str | None) -> tuple[EvidenceStrength, str]:
    """ClinGen PVS1 tree, NMD-escape branch: Strong if the removed region is critical
    (overlaps a UniProt functional region) or > 10% of the protein, otherwise Moderate."""
    first_aa = cds_pos // 3 + 1
    length = tx.protein_length or 1
    removed = max(0.0, (length - first_aa + 1) / length)
    critical = [r for r in (uniprot_api.get_functional_regions(gene) if gene else [])
                if r["end"] >= first_aa]
    if critical:
        return EvidenceStrength.STRONG, (f"NMD escape; removed C-terminal region (aa {first_aa}-{length}) "
                                         f"overlaps UniProt {critical[0]['type']} {critical[0]['start']}-{critical[0]['end']}")
    if removed > _PVS1_MIN_REMOVED_FRACTION:
        return EvidenceStrength.STRONG, f"NMD escape; removes {removed:.0%} of the protein (> 10%)"
    return EvidenceStrength.MODERATE, f"NMD escape; removes {removed:.0%} of the protein (<= 10%)"


def _pvs1(variant: AnnotatedVariant, context: VariantDiseaseContext) -> CriterionResult:
    """ClinGen SVI PVS1 decision tree (Abou Tayoun et al. 2018) on GENCODE v19 transcript
    structure. Deviations, all conservative: an LoF mechanism inferred from gnomAD constraint or
    recessive inheritance (instead of curated evidence) and a CCDS transcript that is not MANE
    each lower the strength one level; start-lost is Supporting because the 'pathogenic variants
    upstream of the next in-frame start' evidence would come from ClinVar."""
    code = "PVS1"
    version = "pvs1_v8.1_clingen_tree_clinvar_lof"
    if variant.consequence not in _LOF_CONSEQUENCES:
        return _not_applicable(code, EvidenceDirection.PATHOGENIC,
                                "Consequence is not a loss-of-function type.", version)
    mechanism, mech_note = _pvs1_mechanism(variant, context)
    if mechanism is None:
        return _not_evaluated(code, EvidenceDirection.PATHOGENIC,
                              f"Loss of function is not an established mechanism: {mech_note}.",
                              ["gene_disease_mechanism=loss_of_function"], version)
    if not gencode_transcripts.is_available():
        return _pvs1_v6(variant, context, version)
    tx, relevance, tx_note = _pvs1_transcript(variant)
    if tx is None:
        return _not_evaluated(code, EvidenceDirection.PATHOGENIC,
                              f"Biologically relevant transcript not confirmed: {tx_note}.",
                              ["MANE Select or CCDS transcript"], version)
    pos = _variant_chrom_pos(variant)[1]
    steps = (mechanism == "inferred") + (relevance == "ccds")
    warnings = [n for n, apply in ((mech_note, mechanism == "inferred"), (tx_note, relevance == "ccds")) if apply]
    if steps:
        warnings.append(f"strength lowered {steps} level(s) for the inferred mechanism/transcript")

    branch: str
    if variant.consequence == _TRANSCRIPT_ABLATION_CONSEQUENCE:
        strength, branch = EvidenceStrength.VERY_STRONG, "whole-transcript ablation"
    elif variant.consequence == _START_LOST_CONSEQUENCE:
        strength, branch = EvidenceStrength.SUPPORTING, ("initiation codon; upstream-pathogenic-variant "
                                                         "evidence not used (ClinVar), so Supporting")
    elif variant.consequence in _CANONICAL_SPLICE_CONSEQUENCES:
        hit = tx.skipped_exon_for_splice_site(pos)
        if hit is None:
            return _not_evaluated(code, EvidenceDirection.PATHOGENIC,
                                  f"Variant is not at ±1/2 of an exon boundary of {tx.transcript_id}.",
                                  ["canonical splice position in the relevant transcript"], version)
        idx, side = hit
        coding = tx.coding_length_of_exon(idx)
        prot = tx.protein_range_of_exon(idx)
        if not coding or prot is None or idx == 0:
            return _not_evaluated(code, EvidenceDirection.PATHOGENIC,
                                  f"{side} site of exon {idx + 1}: skipped exon is non-coding or holds the "
                                  "start codon, which the tree handles separately.",
                                  ["coding exon skip"], version)
        if coding % 3:
            exon_start_cds = tx.cds_end_of_exon(idx) - coding + 1
            if tx.nmd_predicted(exon_start_cds):
                strength, branch = EvidenceStrength.VERY_STRONG, (
                    f"{side} site; skipping exon {idx + 1} ({coding} nt) shifts the frame and NMD is predicted")
            else:
                strength, why = _pvs1_truncation_strength(tx, exon_start_cds, variant.gene)
                branch = f"{side} site; frameshifting skip of exon {idx + 1}, {why}"
        else:
            critical = [r for r in uniprot_api.get_functional_regions(variant.gene or "")
                        if r["start"] <= prot[1] and r["end"] >= prot[0]]
            frac = (prot[1] - prot[0] + 1) / (tx.protein_length or 1)
            if critical:
                strength, why = EvidenceStrength.STRONG, f"removes part of UniProt {critical[0]['type']}"
            elif frac > _PVS1_MIN_REMOVED_FRACTION:
                strength, why = EvidenceStrength.STRONG, f"removes {frac:.0%} of the protein"
            else:
                strength, why = EvidenceStrength.MODERATE, f"removes {frac:.0%} of the protein"
            branch = f"{side} site; in-frame skip of exon {idx + 1} (aa {prot[0]}-{prot[1]}), {why}"
        warnings.append("exon skipping assumed; cryptic splice-site use was not modelled")
    else:  # stop_gained / frameshift
        cds_pos = tx.cds_offset(pos)
        if cds_pos is None:
            return _not_evaluated(code, EvidenceDirection.PATHOGENIC,
                                  f"Variant is outside the coding sequence of {tx.transcript_id}.",
                                  ["CDS position in the relevant transcript"], version)
        if tx.nmd_predicted(cds_pos):
            strength, branch = EvidenceStrength.VERY_STRONG, (
                f"c.{cds_pos} is > 50 nt upstream of the last exon-exon junction: NMD predicted")
        else:
            strength, branch = _pvs1_truncation_strength(tx, cds_pos, variant.gene)
        if variant.consequence == "frameshift_variant":
            warnings.append("frameshift: NMD judged from the variant position, not the new stop codon")

    strength = _downgrade(strength, steps)
    return _result(
        code, CriterionStatus.MET, EvidenceDirection.PATHOGENIC,
        f"ClinGen PVS1 tree: {branch}; mechanism: {mech_note}; transcript {tx.transcript_id} ({relevance}).",
        strength=strength, warnings=warnings,
        evidence_items=[_pvs1_exon_evidence(variant), EvidenceItem(
            evidence_id=f"gencode:{tx.transcript_id}:{variant.variant_id}", evidence_type="transcript_structure",
            source=gencode_transcripts.SOURCE_VERSION,
            value={"transcript": tx.transcript_id, "relevance": relevance, "mechanism": mechanism})],
        evaluator_version=version,
        source_versions={**_vep_source_versions(), "gencode": "v19"},
    )


def _pvs1_v6(variant: AnnotatedVariant, context: VariantDiseaseContext, version: str) -> CriterionResult:
    """Previous rule (VEP exon field only), used when the GENCODE index is not installed."""
    code = "PVS1"
    if context.disease_mechanism != "loss_of_function":
        return _not_evaluated(code, EvidenceDirection.PATHOGENIC,
                              "Established loss-of-function disease mechanism is not confirmed.",
                              ["gene_disease_mechanism=loss_of_function"], version)
    if variant.mane_status not in ("MANE_Select", "MANE_Plus_Clinical"):
        return _not_evaluated(
            code, EvidenceDirection.PATHOGENIC,
            "Biological relevance of the transcript is not confirmed (ClinGen's "
            "PVS1 recommendations require assessing the biologically relevant "
            "transcript before PVS1 is applied) -- this variant's transcript is "
            "not annotated as MANE Select or MANE Plus Clinical.",
            ["mane_status=MANE_Select or MANE_Plus_Clinical"], version,
        )

    if variant.consequence == _START_LOST_CONSEQUENCE:
        return _not_evaluated(
            code, EvidenceDirection.PATHOGENIC,
            "Start-loss requires confirming whether a downstream in-frame "
            "alternative start codon rescues translation (ClinGen's PVS1 "
            "decision tree treats this as a separate path from truncating "
            "variants) -- no alternative-start-codon annotation is available "
            "in this pipeline.",
            ["alternative_start_codon_annotation"], version,
        )
    if variant.consequence in _CANONICAL_SPLICE_CONSEQUENCES:
        return _not_evaluated(
            code, EvidenceDirection.PATHOGENIC,
            "A canonical splice-site variant's true consequence (frameshifting "
            "intron retention or exon loss vs. an in-frame exon skip that may "
            "preserve function) depends on the predicted splicing outcome, not "
            "merely the ±1/2 annotation -- no splice-outcome predictor "
            "(distinct from a bare SpliceAI disruption-probability score) is "
            "available in this pipeline to confirm a true loss-of-function effect.",
            ["splice_outcome_prediction"], version,
        )

    if variant.consequence == _TRANSCRIPT_ABLATION_CONSEQUENCE:
        return _result(
            code, CriterionStatus.MET, EvidenceDirection.PATHOGENIC,
            "Whole-transcript ablation -- no transcript/protein product exists, "
            "so NMD/exon-position nuance does not apply.",
            strength=EvidenceStrength.VERY_STRONG,
            evidence_items=[_pvs1_exon_evidence(variant)],
            evaluator_version=version, source_versions=_vep_source_versions(),
        )

    exon = (variant.exon or "").split("/")
    if not (len(exon) == 2 and exon[0].isdigit() and exon[1].isdigit()):
        return _not_evaluated(
            code, EvidenceDirection.PATHOGENIC,
            "Exon position could not be determined, so it is not possible to "
            "tell which branch of the ClinGen PVS1 NMD decision tree applies.",
            ["exon position (e.g. VEP's exon field)"], version,
        )
    exon_n, exon_total = int(exon[0]), int(exon[1])
    if exon_n == exon_total:
        return _not_evaluated(
            code, EvidenceDirection.PATHOGENIC,
            "Variant is in the last exon -- NMD is predicted to be escaped, "
            "which selects ClinGen's non-NMD PVS1 branch. That branch requires "
            "evaluating critical functional-domain overlap and the proportion "
            "of the protein removed, neither of which is annotated in this "
            "pipeline, so the branch cannot be completed.",
            ["functional_domain_overlap", "proportion_of_protein_removed"], version,
        )
    return _result(
        code, CriterionStatus.MET, EvidenceDirection.PATHOGENIC,
        "Loss-of-function consequence in a gene-disease pair with an "
        "established LoF mechanism; variant is upstream of the last exon, so "
        "NMD is predicted and the ClinGen PVS1 decision tree supports full "
        "(Very Strong) strength for this branch.",
        strength=EvidenceStrength.VERY_STRONG,
        evidence_items=[_pvs1_exon_evidence(variant)],
        evaluator_version=version,
        source_versions=_vep_source_versions(),
    )


def _scope_matched_af_an(variant: AnnotatedVariant) -> tuple[float | None, int | None, str | None]:
    if variant.gnomad_af_popmax is not None and variant.gnomad_af_popmax_allele_number is not None:
        return variant.gnomad_af_popmax, variant.gnomad_af_popmax_allele_number, "popmax_population"
    return None, None, None


def _popmax_af_coverage_unconfirmed(variant: AnnotatedVariant) -> float | None:
    return variant.gnomad_af_popmax


# BA1 (Ghosh et al. 2018): AF > 5% "in any general continental population dataset of at
# least 2,000 observed alleles". These gnomAD v4.1 continental groups each contain thousands of
# individuals in the exomes (far over 1,000), so a frequency from them meets the 2,000-allele
# requirement by dataset size. Founder/bottlenecked or small groups (ASJ, FIN, MID, AMI,
# "remaining") and 1000 Genomes populations are excluded. Per-site AN is not available in the
# VEP cache; this is disclosed in docs/ACMG_RULES.md.
_BA1_CONTINENTAL_POPULATIONS = ("AFR", "AMR", "EAS", "NFE", "SAS")
_BA1_CONTINENTAL_KEY_RE = re.compile(
    r"^gnomAD[eg]_(" + "|".join(_BA1_CONTINENTAL_POPULATIONS) + r")_AF$")


def _continental_max_af(variant: AnnotatedVariant) -> tuple[float, str] | None:
    best: tuple[float, str] | None = None
    for key, af in (variant.gnomad_af_by_population or {}).items():
        if af is None or not _BA1_CONTINENTAL_KEY_RE.match(key):
            continue
        if best is None or af > best[0]:
            best = (af, key.removesuffix("_AF"))
    return best


def _pm2_supporting(variant: AnnotatedVariant, context: VariantDiseaseContext) -> CriterionResult:
    code = "PM2"
    version = "pm2_supporting_v7.0"
    caveat = (f"disclosed research rarity filter ({_PM2_RESEARCH_THRESHOLD}), "
              "not a guideline-derived universal PM2 threshold")
    if variant.gnomad_frequency_status == "absent":
        return _result(
            code, CriterionStatus.MET, EvidenceDirection.PATHOGENIC,
            "Variant is absent from gnomAD (the queried frequency source returned no "
            "record for this allele). Site-level coverage could not be confirmed, so "
            "absence is treated as a disclosed, lower-rigor rarity signal.",
            strength=EvidenceStrength.SUPPORTING,
            evidence_items=[EvidenceItem(
                evidence_id=f"gnomad:{context.genome_build}:{variant.variant_id}:pm2",
                evidence_type="population_frequency", source="gnomAD",
                value={"af": None, "status": "absent"},
            )],
            warnings=["gnomAD site coverage not confirmed for an absent variant."],
            evaluator_version=version, source_versions=_gnomad_source_versions(),
        )
    if variant.gnomad_frequency_status != "observed":
        return _not_evaluated(code, EvidenceDirection.PATHOGENIC,
                               "Population frequency annotation is unavailable.",
                               ["gnomad_frequency_status=observed"], version)
    af, an, scope = _scope_matched_af_an(variant)
    coverage_confirmed = an is not None and an >= _MIN_ADEQUATE_ALLELE_NUMBER
    if af is None:
        af = _popmax_af_coverage_unconfirmed(variant)
        scope = "popmax_coverage_unconfirmed"
    if af is None:
        return _not_evaluated(code, EvidenceDirection.PATHOGENIC,
                               "No gnomAD popmax AF is available from any local source "
                               "(an overall/pooled frequency is not used as a substitute).",
                               ["gnomad_af_popmax"], version)
    coverage_note = (
        "" if coverage_confirmed else
        " Population-specific allele number could not be confirmed for this "
        "frequency (no local source provides it) -- treated as a disclosed, "
        "lower-rigor rarity signal (common practice in rare-disease filtering "
        "pipelines), not a fully guideline-compliant adequately-covered check."
    )
    evidence = [EvidenceItem(
        evidence_id=f"gnomad:{context.genome_build}:{variant.variant_id}:pm2",
        evidence_type="population_frequency", source="gnomAD",
        value={"af": af, "allele_number": an, "scope": scope},
    )]
    if af < _PM2_RESEARCH_THRESHOLD:
        return _result(
            code, CriterionStatus.MET, EvidenceDirection.PATHOGENIC,
            f"Variant is absent/rare in gnomAD, below the {caveat}.{coverage_note}",
            strength=EvidenceStrength.SUPPORTING, evidence_items=evidence,
            evaluator_version=version, source_versions=_gnomad_source_versions(),
        )
    return _result(
        code, CriterionStatus.NOT_MET, EvidenceDirection.PATHOGENIC,
        f"Variant frequency exceeds the {caveat}.{coverage_note}",
        evidence_items=evidence, evaluator_version=version,
        source_versions=_gnomad_source_versions(),
    )


def _ba1(variant: AnnotatedVariant, context: VariantDiseaseContext) -> CriterionResult:
    code = "BA1"
    version = "ba1_v7.0"
    if variant.gnomad_frequency_status == "absent":
        return _result(code, CriterionStatus.NOT_MET, EvidenceDirection.BENIGN,
                        "Variant is absent from gnomAD, so the stand-alone benign "
                        "frequency threshold cannot be exceeded.",
                        evaluator_version=version, source_versions=_gnomad_source_versions())
    if variant.gnomad_frequency_status != "observed":
        return _not_evaluated(code, EvidenceDirection.BENIGN,
                               "Population frequency annotation is unavailable.",
                               ["gnomad_frequency_status=observed"], version)
    # 1) a popmax AF with its own allele number (optional gnomAD v2.1.1 index / online API);
    # 2) otherwise the highest gnomAD v4.1 continental-population AF from the VEP cache;
    # 3) otherwise a coverage-unconfirmed popmax (e.g. VEP MAX_AF incl. 1000 Genomes),
    #    which can never make BA1 met.
    af, an, scope = _scope_matched_af_an(variant)
    coverage_confirmed = an is not None and an >= _MIN_ADEQUATE_ALLELE_NUMBER
    if not coverage_confirmed:
        continental = _continental_max_af(variant)
        if continental is not None:
            af, scope = continental[0], f"gnomad_v4.1_continental:{continental[1]}"
            an, coverage_confirmed = None, True
    if af is None:
        af = _popmax_af_coverage_unconfirmed(variant)
        scope = "popmax_coverage_unconfirmed"
    if af is None:
        return _not_evaluated(code, EvidenceDirection.BENIGN,
                               "No gnomAD popmax AF is available from any local "
                               "source (an overall/pooled frequency is not used "
                               "as a substitute for BA1's continental-population "
                               "requirement).",
                               ["gnomad_af_popmax"], version)
    evidence = [EvidenceItem(
        evidence_id=f"gnomad:{context.genome_build}:{variant.variant_id}:ba1",
        evidence_type="population_frequency", source="gnomAD",
        value={"af": af, "allele_number": an, "scope": scope},
    )]
    if af <= _BA1_THRESHOLD:
        return _result(code, CriterionStatus.NOT_MET, EvidenceDirection.BENIGN,
                        "Population frequency does not exceed the stand-alone benign threshold.",
                        evidence_items=evidence, evaluator_version=version,
                        source_versions=_gnomad_source_versions())
    if not coverage_confirmed:
        return _not_evaluated(
            code, EvidenceDirection.BENIGN,
            f"Population frequency ({af}) would exceed the stand-alone benign "
            f"threshold ({_BA1_THRESHOLD}), but population-specific allele "
            "number could not be confirmed for this frequency -- BA1's "
            "Stand-Alone, classification-overriding strength must not be "
            "assigned from a coverage-unconfirmed frequency alone.",
            ["gnomad_af_popmax_allele_number (coverage confirmation)"], version,
        )
    if not _BA1_EXCEPTION_LIST_AVAILABLE:
        return _not_evaluated(
            code, EvidenceDirection.BENIGN,
            f"Population frequency ({af}) exceeds the stand-alone benign "
            f"threshold ({_BA1_THRESHOLD}), but the ClinGen BA1 exception list "
            "has not been loaded/checked in this pipeline -- an unchecked "
            "exception dataset must be treated as 'not checked', not 'no "
            "exception', so BA1 cannot be assigned MET.",
            ["ba1_exception_list_loaded"], version,
        )
    exception = clingen_ba1_exceptions.match(variant.gene, variant.hgvs_p, variant.variant_id)
    if exception or variant.variant_id in _BA1_EXCEPTION_VARIANTS:
        label = (f"{exception['gene']} {exception['transcript']}:{exception['hgvs_c']}"
                 if exception else variant.variant_id)
        return _result(
            code, CriterionStatus.NOT_MET, EvidenceDirection.BENIGN,
            f"This variant is on the {clingen_ba1_exceptions.SOURCE} ({label}); "
            "the general stand-alone threshold does not apply here.",
            evidence_items=evidence, evaluator_version=version,
            source_versions=_gnomad_source_versions(),
        )
    return _result(
        code, CriterionStatus.MET, EvidenceDirection.BENIGN,
        f"Population frequency ({af}, {scope}) exceeds the stand-alone benign threshold "
        f"({_BA1_THRESHOLD}) in a continental population of >= "
        f"{_MIN_ADEQUATE_ALLELE_NUMBER} alleles, and the variant is not on the "
        f"{clingen_ba1_exceptions.SOURCE}.",
        strength=EvidenceStrength.STAND_ALONE, evidence_items=evidence,
        evaluator_version=version, source_versions=_gnomad_source_versions(),
    )


def _bs1(variant: AnnotatedVariant, context: VariantDiseaseContext) -> CriterionResult:
    return _not_evaluated(
        "BS1", EvidenceDirection.BENIGN,
        "BS1 requires a disease-specific expected-frequency calculation "
        "(prevalence, penetrance, allelic heterogeneity) -- the base ACMG/AMP "
        "guideline gives no universal numeric threshold, and this pipeline has "
        "no disease-prevalence data to compute one.",
        ["disease_prevalence", "disease_penetrance", "allelic_heterogeneity"],
        "bs1_v4.0",
    )


def _pp3_bp4(variant: AnnotatedVariant, context: VariantDiseaseContext,
             pvs1_result: CriterionResult) -> CriterionResult:
    code = "PP3/BP4"
    version = "pp3_bp4_v3.0_splicing_svi2023"

    # ClinGen SVI splicing recommendations (Walker et al. 2023): for variants that do not change
    # the protein sequence (synonymous, intronic, UTR, non-coding) SpliceAI gives BP4 (<= 0.10)
    # or PP3 (>= 0.20); BP7 can be applied in addition for synonymous variants.
    if (not variant.is_splice_relevant and variant.spliceai_score is not None
            and variant.consequence in _NON_PROTEIN_CHANGING):
        evidence = [EvidenceItem(
            evidence_id=f"spliceai:{variant.variant_id}", evidence_type="computational_prediction",
            source="SpliceAI", value={"delta_score": variant.spliceai_score},
        )]
        if variant.spliceai_score <= _SPLICEAI_BP4_THRESHOLD:
            return _result("BP4", CriterionStatus.MET, EvidenceDirection.BENIGN,
                           f"{variant.consequence}: SpliceAI delta {variant.spliceai_score} <= "
                           f"{_SPLICEAI_BP4_THRESHOLD}, no predicted splicing impact (Walker 2023).",
                           strength=EvidenceStrength.SUPPORTING, evidence_items=evidence,
                           evaluator_version=version)
        if variant.spliceai_score >= _SPLICEAI_PP3_THRESHOLD:
            return _result("PP3", CriterionStatus.MET, EvidenceDirection.PATHOGENIC,
                           f"{variant.consequence}: SpliceAI delta {variant.spliceai_score} >= "
                           f"{_SPLICEAI_PP3_THRESHOLD}, predicted splicing impact (Walker 2023).",
                           strength=EvidenceStrength.SUPPORTING, evidence_items=evidence,
                           evaluator_version=version)
        return _result(code, CriterionStatus.NOT_MET, EvidenceDirection.PATHOGENIC,
                       "SpliceAI delta score falls in the indeterminate band (0.10-0.20).",
                       evidence_items=evidence, evaluator_version=version)

    if variant.is_splice_relevant and variant.spliceai_score is not None:
        if (variant.consequence in _CANONICAL_SPLICE_CONSEQUENCES
                and pvs1_result.status == CriterionStatus.MET):
            return _not_applicable(
                code, EvidenceDirection.PATHOGENIC,
                "Canonical splice-site consequence already captured by PVS1; "
                "PP3 must not count the same SpliceAI evidence twice.", version,
            )
        evidence = [EvidenceItem(
            evidence_id=f"spliceai:{variant.variant_id}", evidence_type="computational_prediction",
            source="SpliceAI", value={"delta_score": variant.spliceai_score},
        )]
        if variant.spliceai_score >= _SPLICEAI_PP3_THRESHOLD:
            return _result(
                "PP3", CriterionStatus.MET, EvidenceDirection.PATHOGENIC,
                f"SpliceAI delta score {variant.spliceai_score} >= {_SPLICEAI_PP3_THRESHOLD} "
                "predicts a splicing impact (ClinGen generic threshold).",
                strength=EvidenceStrength.SUPPORTING, evidence_items=evidence,
                warnings=["Applied at Supporting strength per ClinGen's generic/"
                          "conservative recommendation for automated use -- "
                          "Moderate/Strong require an applicable VCEP or a "
                          "separately validated calibration."],
                evaluator_version=version,
            )
        if variant.spliceai_score <= _SPLICEAI_BP4_THRESHOLD:
            return _result(
                "BP4", CriterionStatus.MET, EvidenceDirection.BENIGN,
                f"SpliceAI delta score {variant.spliceai_score} <= {_SPLICEAI_BP4_THRESHOLD} "
                "predicts an absence of splicing impact (ClinGen generic threshold).",
                strength=EvidenceStrength.SUPPORTING, evidence_items=evidence,
                evaluator_version=version,
            )
        return _result(
            code, CriterionStatus.NOT_MET, EvidenceDirection.PATHOGENIC,
            f"SpliceAI delta score falls in the indeterminate band "
            f"({_SPLICEAI_BP4_THRESHOLD}-{_SPLICEAI_PP3_THRESHOLD}).",
            evidence_items=evidence, evaluator_version=version,
        )

    if variant.consequence == "missense_variant" and variant.revel_score is not None:
        evidence = [EvidenceItem(
            evidence_id=f"revel:{variant.variant_id}", evidence_type="computational_prediction",
            source="REVEL", value={"score": variant.revel_score},
        )]
        for cutoff, strength in _REVEL_PP3_TIERS:
            if variant.revel_score >= cutoff:
                return _result(
                    "PP3", CriterionStatus.MET, EvidenceDirection.PATHOGENIC,
                    f"REVEL score {variant.revel_score} meets the Pejaver et al. "
                    f"calibrated pathogenic threshold for {strength.value}.",
                    strength=strength, evidence_items=evidence, evaluator_version=version,
                )
        for cutoff, strength in _REVEL_BP4_TIERS:
            if variant.revel_score <= cutoff:
                return _result(
                    "BP4", CriterionStatus.MET, EvidenceDirection.BENIGN,
                    f"REVEL score {variant.revel_score} meets the Pejaver et al. "
                    f"calibrated benign threshold for {strength.value}.",
                    strength=strength, evidence_items=evidence, evaluator_version=version,
                )
        return _result(
            code, CriterionStatus.NOT_MET, EvidenceDirection.PATHOGENIC,
            "REVEL score falls in the indeterminate band (0.290-0.644).",
            evidence_items=evidence, evaluator_version=version,
        )

    if variant.consequence != "missense_variant" and variant.revel_score is not None:
        return _not_applicable(
            code, EvidenceDirection.PATHOGENIC,
            "REVEL's Pejaver et al. calibration was established for missense "
            "variants only; not applied to this consequence type.", version,
        )

    return _not_evaluated(
        code, EvidenceDirection.PATHOGENIC,
        "No calibrated computational predictor (REVEL for missense variants, "
        "SpliceAI for splice-relevant variants) score is available -- VEP's "
        "SIFT/PolyPhen are not used as a substitute (correlated, uncalibrated "
        "predictors would violate the independence requirement).",
        ["revel_score (missense only) or spliceai_score"], version,
    )


def _pm4(variant: AnnotatedVariant, context: VariantDiseaseContext,
         pvs1_result: CriterionResult) -> CriterionResult:
    code = "PM4"
    version = "pm4_v3.0"
    if variant.consequence not in _PM4_CONSEQUENCES:
        return _not_applicable(code, EvidenceDirection.PATHOGENIC,
                                "Consequence is not an in-frame indel or stop-loss.", version)
    if pvs1_result.status == CriterionStatus.MET:
        return _not_applicable(
            code, EvidenceDirection.PATHOGENIC,
            "Consequence is already captured by PVS1; PM4 must not duplicate it.",
            version,
        )
    if variant.consequence == "stop_lost":
        return _result(
            code, CriterionStatus.MET, EvidenceDirection.PATHOGENIC,
            "Stop-loss (read-through) variant -- qualifies under PM4's "
            "stop-loss branch, which (unlike the in-frame-indel branch) "
            "carries no repeat-region qualifier in the base ACMG/AMP rule.",
            strength=EvidenceStrength.MODERATE,
            evidence_items=[_pvs1_exon_evidence(variant)],
            evaluator_version=version, source_versions=_vep_source_versions(),
        )
    chrom_pos = _variant_chrom_pos(variant)
    if chrom_pos is None:
        return _not_evaluated(
            code, EvidenceDirection.PATHOGENIC,
            "Variant position could not be parsed to check repeat-region overlap.",
            ["repeat_region_annotation"], version,
        )
    in_repeat = _repeat_overlap(chrom_pos, context.genome_build)
    if in_repeat is None:
        return _not_evaluated(
            code, EvidenceDirection.PATHOGENIC,
            "Repeat-region annotation is unavailable (UCSC simpleRepeat hg19 track "
            "not loaded, or the case is not GRCh37) -- cannot confirm this in-frame indel falls outside a "
            "repetitive region with no known functional significance.",
            ["repeat_region_annotation"], version,
        )
    evidence = [EvidenceItem(
        evidence_id=f"ucsc_simplerepeat:{variant.variant_id}:pm4", evidence_type="repeat_region",
        source="UCSC simpleRepeat", value={"in_repeat_region": in_repeat},
    )]
    if in_repeat:
        return _result(
            code, CriterionStatus.NOT_MET, EvidenceDirection.PATHOGENIC,
            "In-frame indel falls within a UCSC simpleRepeat (Tandem Repeats "
            "Finder) region -- PM4's nonrepeat-region requirement is not met.",
            evidence_items=evidence, evaluator_version=version,
        )
    return _result(
        code, CriterionStatus.MET, EvidenceDirection.PATHOGENIC,
        "In-frame indel confirmed outside any UCSC simpleRepeat region.",
        strength=EvidenceStrength.MODERATE, evidence_items=evidence,
        evaluator_version=version,
    )


def _bp3(variant: AnnotatedVariant, context: VariantDiseaseContext) -> CriterionResult:
    code = "BP3"
    version = "bp3_v1.0"
    if variant.consequence not in ("inframe_insertion", "inframe_deletion"):
        return _not_applicable(code, EvidenceDirection.BENIGN,
                                "Consequence is not an in-frame indel.", version)
    chrom_pos = _variant_chrom_pos(variant)
    if chrom_pos is None:
        return _not_evaluated(code, EvidenceDirection.BENIGN,
                               "Variant position could not be parsed to check repeat-region overlap.",
                               ["repeat_region_annotation"], version)
    in_repeat = _repeat_overlap(chrom_pos, context.genome_build)
    if in_repeat is None:
        return _not_evaluated(code, EvidenceDirection.BENIGN,
                               "Repeat-region annotation is unavailable (UCSC simpleRepeat hg19 "
                               "track not loaded, or the case is not GRCh37).",
                               ["repeat_region_annotation"], version)
    evidence = [EvidenceItem(
        evidence_id=f"ucsc_simplerepeat:{variant.variant_id}:bp3", evidence_type="repeat_region",
        source="UCSC simpleRepeat", value={"in_repeat_region": in_repeat},
    )]
    if in_repeat:
        return _result(
            code, CriterionStatus.MET, EvidenceDirection.BENIGN,
            "In-frame indel falls within a UCSC simpleRepeat region -- "
            "consistent with BP3's 'repetitive region with no known "
            "functional significance', though this pipeline cannot separately "
            "confirm the region truly has NO known functional significance "
            "beyond being repetitive.",
            strength=EvidenceStrength.SUPPORTING, evidence_items=evidence,
            evaluator_version=version,
        )
    return _result(
        code, CriterionStatus.NOT_MET, EvidenceDirection.BENIGN,
        "In-frame indel is outside any UCSC simpleRepeat region.",
        evidence_items=evidence, evaluator_version=version,
    )


def _bp7(variant: AnnotatedVariant, context: VariantDiseaseContext) -> CriterionResult:
    code = "BP7"
    version = "bp7_v7.0"
    if variant.consequence not in _SYNONYMOUS_CONSEQUENCES:
        return _not_applicable(code, EvidenceDirection.BENIGN,
                                "Consequence is not synonymous.", version)
    if variant.is_splice_relevant:
        return _result(code, CriterionStatus.NOT_MET, EvidenceDirection.BENIGN,
                        "Variant has an annotated splice-relevant consequence.",
                        evaluator_version=version)
    if variant.spliceai_score is not None and variant.spliceai_score > _SPLICEAI_BP4_THRESHOLD:
        evidence = [EvidenceItem(
            evidence_id=f"spliceai:{variant.variant_id}", evidence_type="computational_prediction",
            source="SpliceAI", value={"delta_score": variant.spliceai_score},
        )]
        return _result(code, CriterionStatus.NOT_MET, EvidenceDirection.BENIGN,
                        f"SpliceAI delta score {variant.spliceai_score} exceeds the "
                        f"no-splice-impact threshold ({_SPLICEAI_BP4_THRESHOLD}).",
                        evidence_items=evidence, evaluator_version=version)
    if variant.conservation_score is None:
        return _not_evaluated(
            code, EvidenceDirection.BENIGN,
            "No splice impact is predicted, but the phyloP conservation track "
            "is unavailable -- cannot confirm the nucleotide is not highly "
            "conserved (BP7 requires both).",
            ["conservation_score (phyloP)"], version,
        )
    cons_evidence = [EvidenceItem(
        evidence_id=f"phylop:{variant.variant_id}", evidence_type="conservation",
        source="UCSC phyloP100way", value={"score": variant.conservation_score},
    )]
    if variant.conservation_score > _PHYLOP_BP7_THRESHOLD:
        return _result(
            code, CriterionStatus.NOT_MET, EvidenceDirection.BENIGN,
            f"phyloP conservation score ({variant.conservation_score}) exceeds "
            f"the disclosed not-conserved threshold ({_PHYLOP_BP7_THRESHOLD}) -- "
            "position is under purifying selection.",
            evidence_items=cons_evidence, evaluator_version=version,
        )
    return _result(
        code, CriterionStatus.MET, EvidenceDirection.BENIGN,
        f"No splice impact predicted and phyloP conservation score "
        f"({variant.conservation_score}) does not exceed the disclosed "
        f"not-conserved threshold ({_PHYLOP_BP7_THRESHOLD}).",
        strength=EvidenceStrength.SUPPORTING, evidence_items=cons_evidence,
        evaluator_version=version,
    )


def _pp2(variant: AnnotatedVariant, context: VariantDiseaseContext) -> CriterionResult:
    code = "PP2"
    version = "pp2_v1.0"
    if variant.consequence != "missense_variant":
        return _not_applicable(code, EvidenceDirection.PATHOGENIC,
                                "PP2 applies to missense variants only.", version)
    gene = variant.gene or context.gene
    constraint = gnomad_constraint.get_constraint(gene) if gene else None
    mis_z = constraint.get("mis_z") if constraint else None
    if mis_z is None:
        return _not_evaluated(
            code, EvidenceDirection.PATHOGENIC,
            "gnomAD missense-constraint (Z-score) data is unavailable for this gene.",
            ["gnomad missense Z-score (gnomad.v4.1.constraint_metrics.tsv)"], version,
        )
    evidence = [EvidenceItem(
        evidence_id=f"gnomad_constraint:{gene}:pp2", evidence_type="gene_constraint",
        source="gnomAD", value={"missense_z": mis_z},
    )]
    if mis_z >= _PP2_MISSENSE_Z_THRESHOLD:
        return _result(
            code, CriterionStatus.MET, EvidenceDirection.PATHOGENIC,
            f"Gene missense Z-score ({mis_z}) meets the literature-cited "
            f"missense-constraint significance threshold ({_PP2_MISSENSE_Z_THRESHOLD}, "
            "Samocha et al. 2014 convention) -- not a guideline-derived universal PP2 number.",
            strength=EvidenceStrength.SUPPORTING, evidence_items=evidence,
            evaluator_version=version, source_versions=_gnomad_source_versions(),
        )
    return _result(
        code, CriterionStatus.NOT_MET, EvidenceDirection.PATHOGENIC,
        f"Gene missense Z-score ({mis_z}) does not meet the missense-constraint threshold.",
        evidence_items=evidence, evaluator_version=version,
        source_versions=_gnomad_source_versions(),
    )


_PP1_MIN_AFFECTED = 1


def _ps2(variant: AnnotatedVariant, context: VariantDiseaseContext) -> CriterionResult:
    code = "PS2"
    version = "ps2_v1.0"
    fe = context.family_evidence
    if fe is None or fe.de_novo_confirmed is None:
        return _not_evaluated(
            code, EvidenceDirection.PATHOGENIC,
            "No confirmed-de-novo statement is available from the clinical "
            "narrative (requires the note to state parental/trio testing "
            "confirmed the variant's absence in both parents).",
            ["family_evidence.de_novo_confirmed"], version,
        )
    evidence = [EvidenceItem(
        evidence_id=f"family_evidence:{context.variant_id}:ps2", evidence_type="family_history",
        source="clinical narrative (demographics, llm-extracted)",
        value={"de_novo_confirmed": fe.de_novo_confirmed},
    )]
    if fe.de_novo_confirmed:
        return _result(
            code, CriterionStatus.MET, EvidenceDirection.PATHOGENIC,
            "Clinical narrative states parental/trio testing confirmed this "
            "variant is de novo (absent in both parents).",
            strength=EvidenceStrength.STRONG, evidence_items=evidence,
            evaluator_version=version,
        )
    return _result(
        code, CriterionStatus.NOT_MET, EvidenceDirection.PATHOGENIC,
        "Clinical narrative states de novo occurrence was NOT confirmed by "
        "parental/trio testing.",
        evidence_items=evidence, evaluator_version=version,
    )


def _pm6(variant: AnnotatedVariant, context: VariantDiseaseContext) -> CriterionResult:
    code = "PM6"
    version = "pm6_v1.0"
    fe = context.family_evidence
    if fe is not None and fe.de_novo_confirmed:
        return _not_applicable(
            code, EvidenceDirection.PATHOGENIC,
            "De novo occurrence is confirmed by parental/trio testing -- PS2 "
            "applies instead; PM6 must not double-count the same de novo fact.",
            version,
        )
    if fe is None or fe.de_novo_stated is None:
        return _not_evaluated(
            code, EvidenceDirection.PATHOGENIC,
            "No de novo statement is available from the clinical narrative.",
            ["family_evidence.de_novo_stated"], version,
        )
    evidence = [EvidenceItem(
        evidence_id=f"family_evidence:{context.variant_id}:pm6", evidence_type="family_history",
        source="clinical narrative (demographics, llm-extracted)",
        value={"de_novo_stated": fe.de_novo_stated},
    )]
    if fe.de_novo_stated:
        return _result(
            code, CriterionStatus.MET, EvidenceDirection.PATHOGENIC,
            "Clinical narrative states the variant is de novo, without "
            "stating parental/trio testing confirmed it (assumed, not "
            "confirmed, de novo occurrence).",
            strength=EvidenceStrength.MODERATE, evidence_items=evidence,
            evaluator_version=version,
        )
    return _result(
        code, CriterionStatus.NOT_MET, EvidenceDirection.PATHOGENIC,
        "Clinical narrative does not state a de novo occurrence.",
        evidence_items=evidence, evaluator_version=version,
    )


def _pp1(variant: AnnotatedVariant, context: VariantDiseaseContext) -> CriterionResult:
    code = "PP1"
    version = "pp1_v1.0"
    fe = context.family_evidence
    if fe is None or fe.segregation_observed is None:
        return _not_evaluated(
            code, EvidenceDirection.PATHOGENIC,
            "No segregation statement is available from the clinical narrative.",
            ["family_evidence.segregation_observed"], version,
        )
    evidence = [EvidenceItem(
        evidence_id=f"family_evidence:{context.variant_id}:pp1", evidence_type="family_history",
        source="clinical narrative (demographics, llm-extracted)",
        value={"segregation_observed": fe.segregation_observed,
               "n_affected_segregating": fe.n_affected_segregating},
    )]
    if fe.segregation_observed:
        return _result(
            code, CriterionStatus.MET, EvidenceDirection.PATHOGENIC,
            "Clinical narrative states the variant/phenotype tracks with the "
            "disease across tested relatives -- applied at Supporting only "
            "(no quantitative meioses-based upgrade attempted).",
            strength=EvidenceStrength.SUPPORTING, evidence_items=evidence,
            evaluator_version=version,
        )
    return _result(
        code, CriterionStatus.NOT_MET, EvidenceDirection.PATHOGENIC,
        "Clinical narrative does not state segregation with the disease.",
        evidence_items=evidence, evaluator_version=version,
    )


def _bs4(variant: AnnotatedVariant, context: VariantDiseaseContext) -> CriterionResult:
    code = "BS4"
    version = "bs4_v1.0"
    fe = context.family_evidence
    if fe is None or fe.segregation_observed is None:
        return _not_evaluated(
            code, EvidenceDirection.BENIGN,
            "No segregation statement is available from the clinical narrative.",
            ["family_evidence.segregation_observed"], version,
        )
    evidence = [EvidenceItem(
        evidence_id=f"family_evidence:{context.variant_id}:bs4", evidence_type="family_history",
        source="clinical narrative (demographics, llm-extracted)",
        value={"segregation_observed": fe.segregation_observed,
               "n_unaffected_nonsegregating": fe.n_unaffected_nonsegregating},
    )]
    if fe.segregation_observed is False:
        return _result(
            code, CriterionStatus.MET, EvidenceDirection.BENIGN,
            "Clinical narrative explicitly states the variant/phenotype does "
            "NOT track with the disease in tested relatives.",
            strength=EvidenceStrength.STRONG, evidence_items=evidence,
            evaluator_version=version,
        )
    return _result(
        code, CriterionStatus.NOT_MET, EvidenceDirection.BENIGN,
        "Clinical narrative states segregation WAS observed (supports PP1, "
        "not BS4).",
        evidence_items=evidence, evaluator_version=version,
    )


def _bp1(variant: AnnotatedVariant, context: VariantDiseaseContext) -> CriterionResult:
    """BP1: missense variant in a gene where the gene's OWN known pathogenic variants
    are primarily truncating -- a curated fact about that gene's disease mechanism.
    Approximated by classifying the gene's ClinVar pathogenic/likely-pathogenic
    variants as truncating vs. missense (known_pathogenic_residues.truncating_fraction),
    NOT by general population missense-constraint (gnomAD Z-score), which measures
    genome-wide selective tolerance for missense change and is a different question
    from what this specific gene's disease-causing variants look like."""
    code = "BP1"
    version = "bp1_v2.0_truncating_fraction"
    if variant.consequence != "missense_variant":
        return _not_applicable(code, EvidenceDirection.BENIGN,
                                "BP1 applies to missense variants only.", version)
    if not known_pathogenic_residues.is_available():
        return _not_evaluated(
            code, EvidenceDirection.BENIGN,
            "Known-pathogenic-variant reference (ClinVar variant_summary) is not available.",
            ["known_pathogenic_residues data"], version,
        )
    gene = variant.gene or context.gene
    if not gene:
        return _not_evaluated(code, EvidenceDirection.BENIGN, "No gene symbol available.",
                               ["gene"], version)
    fraction, n = known_pathogenic_residues.truncating_fraction(gene)
    if fraction is None:
        return _not_evaluated(
            code, EvidenceDirection.BENIGN,
            f"Fewer than {known_pathogenic_residues.BP1_MIN_CLASSIFIED} of this "
            f"gene's ClinVar pathogenic/likely-pathogenic variants could be "
            f"classified as truncating or missense ({n} found) -- too few to "
            "establish whether this gene's known pathogenic variants are "
            "primarily truncating.",
            ["known_pathogenic_residues truncating/missense classification"], version,
        )
    evidence = [EvidenceItem(
        evidence_id=f"known_pathogenic_residues:{gene}:bp1", evidence_type="gene_mechanism",
        source="ClinVar variant_summary (truncating/missense classification)",
        value={"truncating_fraction": round(fraction, 3), "n_classified": n},
    )]
    threshold = known_pathogenic_residues.BP1_TRUNCATING_FRACTION_THRESHOLD
    if fraction >= threshold:
        return _result(
            code, CriterionStatus.MET, EvidenceDirection.BENIGN,
            f"{round(fraction * 100, 1)}% of this gene's {n} classified ClinVar "
            f"pathogenic/likely-pathogenic variants are truncating (nonsense, "
            f"frameshift, or canonical splice) -- at or above the disclosed "
            f"{threshold:.0%} threshold for 'primarily truncating', consistent "
            "with BP1's gene-mechanism premise.",
            strength=EvidenceStrength.SUPPORTING, evidence_items=evidence,
            evaluator_version=version,
        )
    return _result(
        code, CriterionStatus.NOT_MET, EvidenceDirection.BENIGN,
        f"Only {round(fraction * 100, 1)}% of this gene's {n} classified ClinVar "
        f"pathogenic/likely-pathogenic variants are truncating -- missense "
        "variants ARE an established part of this gene's disease mechanism, so "
        "BP1 does not apply.",
        evidence_items=evidence, evaluator_version=version,
        source_versions=_gnomad_source_versions(),
    )


_OWN_MISSENSE_HGVS_P_RE = re.compile(r"p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})")


def _own_missense_change(variant: AnnotatedVariant) -> tuple[int, str, str] | None:
    if not variant.hgvs_p:
        return None
    m = _OWN_MISSENSE_HGVS_P_RE.search(variant.hgvs_p)
    if not m:
        return None
    return int(m.group(2)), m.group(1), m.group(3)


def _ps1(variant: AnnotatedVariant, context: VariantDiseaseContext) -> CriterionResult:
    code = "PS1"
    version = "ps1_v1.0"
    if variant.consequence != "missense_variant":
        return _not_applicable(code, EvidenceDirection.PATHOGENIC,
                                "PS1 applies to missense variants only.", version)
    if not known_pathogenic_residues.is_available():
        return _not_evaluated(
            code, EvidenceDirection.PATHOGENIC,
            "Known-pathogenic-residue reference (ClinVar variant_summary + "
            "UniProt humsavar) is not available.",
            ["known_pathogenic_residues data"], version,
        )
    own = _own_missense_change(variant)
    if own is None:
        return _not_evaluated(
            code, EvidenceDirection.PATHOGENIC,
            "This variant's own amino-acid change could not be determined.",
            ["protein position + amino acid change"], version,
        )
    position, ref_aa, alt_aa = own
    gene = variant.gene or context.gene
    known = known_pathogenic_residues.known_pathogenic_at_residue(gene, position)
    match = any(r == ref_aa and a == alt_aa for r, a in known)
    evidence = [EvidenceItem(
        evidence_id=f"known_pathogenic_residues:{gene}:{position}:ps1", evidence_type="known_variant",
        source="ClinVar variant_summary + UniProt humsavar",
        value={"position": position, "change": f"{ref_aa}->{alt_aa}", "known_at_residue": known},
    )]
    if match:
        return _result(
            code, CriterionStatus.MET, EvidenceDirection.PATHOGENIC,
            f"The SAME amino acid change ({ref_aa}{position}{alt_aa}) is already "
            "reported pathogenic/likely pathogenic at this residue. Caveat: this "
            "does not separately confirm the matching record is a DIFFERENT "
            "nucleotide-level variant (the base guideline's 'regardless of "
            "nucleotide change' framing implies independent observations) -- if "
            "this variant IS itself that record, this is not independent evidence.",
            strength=EvidenceStrength.STRONG, evidence_items=evidence,
            evaluator_version=version,
        )
    return _result(
        code, CriterionStatus.NOT_MET, EvidenceDirection.PATHOGENIC,
        f"No known-pathogenic entry for the SAME amino acid change "
        f"({ref_aa}{position}{alt_aa}) at this residue.",
        evidence_items=evidence, evaluator_version=version,
    )


def _pm5(variant: AnnotatedVariant, context: VariantDiseaseContext) -> CriterionResult:
    code = "PM5"
    version = "pm5_v1.0"
    if variant.consequence != "missense_variant":
        return _not_applicable(code, EvidenceDirection.PATHOGENIC,
                                "PM5 applies to missense variants only.", version)
    if not known_pathogenic_residues.is_available():
        return _not_evaluated(
            code, EvidenceDirection.PATHOGENIC,
            "Known-pathogenic-residue reference (ClinVar variant_summary + "
            "UniProt humsavar) is not available.",
            ["known_pathogenic_residues data"], version,
        )
    own = _own_missense_change(variant)
    if own is None:
        return _not_evaluated(
            code, EvidenceDirection.PATHOGENIC,
            "This variant's own amino-acid change could not be determined.",
            ["protein position + amino acid change"], version,
        )
    position, ref_aa, alt_aa = own
    gene = variant.gene or context.gene
    known = known_pathogenic_residues.known_pathogenic_at_residue(gene, position)
    different = [(r, a) for r, a in known if a != alt_aa]
    evidence = [EvidenceItem(
        evidence_id=f"known_pathogenic_residues:{gene}:{position}:pm5", evidence_type="known_variant",
        source="ClinVar variant_summary + UniProt humsavar",
        value={"position": position, "this_change": f"{ref_aa}->{alt_aa}", "other_known_changes": different},
    )]
    if different:
        return _result(
            code, CriterionStatus.MET, EvidenceDirection.PATHOGENIC,
            f"A DIFFERENT missense change at this same residue ({ref_aa}{position}) "
            f"is already reported pathogenic/likely pathogenic: {different}. Novel "
            "amino acid changes are not per se equivalent in effect to a "
            "previously reported one -- ClinGen SVI caution applies (not applied "
            "as a separate downgrade check here, disclosed as a known simplification).",
            strength=EvidenceStrength.MODERATE, evidence_items=evidence,
            evaluator_version=version,
        )
    return _result(
        code, CriterionStatus.NOT_MET, EvidenceDirection.PATHOGENIC,
        f"No known-pathogenic entry for a DIFFERENT amino acid change at this "
        f"residue ({ref_aa}{position}).",
        evidence_items=evidence, evaluator_version=version,
    )


def _pm1(variant: AnnotatedVariant, context: VariantDiseaseContext) -> CriterionResult:
    code = "PM1"
    version = "pm1_v2.0_uniprot_region"
    if variant.consequence != "missense_variant":
        return _not_applicable(code, EvidenceDirection.PATHOGENIC,
                                "PM1 applies to missense variants only.", version)
    if not known_pathogenic_residues.is_available():
        return _not_evaluated(
            code, EvidenceDirection.PATHOGENIC,
            "Known-pathogenic-residue reference (ClinVar variant_summary + "
            "UniProt humsavar) is not available.",
            ["known_pathogenic_residues data"], version,
        )
    own = _own_missense_change(variant)
    if own is None:
        return _not_evaluated(
            code, EvidenceDirection.PATHOGENIC,
            "This variant's own protein position could not be determined.",
            ["protein position"], version,
        )
    position, _ref_aa, _alt_aa = own
    gene = variant.gene or context.gene

    region = _uniprot_essential_region(gene, position)
    if region is not None:
        region_label = f"UniProt {region['type']}" + (
            f" ({region['description']})" if region["description"] else "")
        region_source = "UniProt curated feature"
    elif variant.domains:
        region_label = f"a VEP-annotated domain ({', '.join(variant.domains[:2])})"
        region_source = "VEP DOMAINS (generic Pfam/InterPro, uncurated fallback)"
    else:
        return _not_evaluated(
            code, EvidenceDirection.PATHOGENIC,
            "No curated UniProt functional region and no VEP domain annotation "
            "are available for this variant's position.",
            ["UniProt functional-region data", "VEP DOMAINS annotation"], version,
        )

    nearby = known_pathogenic_residues.pathogenic_residues_near(gene, position, _PM1_CLUSTER_WINDOW)
    evidence = [EvidenceItem(
        evidence_id=f"pm1_cluster:{gene}:{position}", evidence_type="mutational_cluster",
        source=f"{region_source} + ClinVar/UniProt known-pathogenic residues",
        value={"region": region_label, "nearby_pathogenic_positions": nearby,
               "window": _PM1_CLUSTER_WINDOW},
    )]
    if nearby >= _PM1_CLUSTER_MIN_NEARBY:
        return _result(
            code, CriterionStatus.MET, EvidenceDirection.PATHOGENIC,
            f"Variant falls within {region_label} and {nearby} other distinct "
            f"known-pathogenic missense position(s) fall within {_PM1_CLUSTER_WINDOW} "
            "residues -- a disclosed local-clustering proxy for a mutational "
            "hotspot, not a determination confirming the absence of benign "
            "variation at this position.",
            strength=EvidenceStrength.MODERATE, evidence_items=evidence,
            evaluator_version=version,
        )
    return _result(
        code, CriterionStatus.NOT_MET, EvidenceDirection.PATHOGENIC,
        f"Variant is within {region_label}, but only {nearby} nearby known-"
        f"pathogenic position(s) were found (threshold {_PM1_CLUSTER_MIN_NEARBY}) "
        "-- insufficient local clustering for this proxy.",
        evidence_items=evidence, evaluator_version=version,
    )


_RECESSIVE_INHERITANCE_MODELS = {"autosomal_recessive", "x_linked_recessive"}


def _bs2(variant: AnnotatedVariant, context: VariantDiseaseContext) -> CriterionResult:
    """BS2: observed in a healthy adult individual under the genotype state relevant to
    the disease's inheritance pattern (homozygous/hemizygous/heterozygous), with full
    penetrance expected at an early age. Evaluated from an explicit clinical-narrative
    statement about an unaffected carrier -- NOT from population-database presence,
    which cannot confirm the observed individual's health/disease status and is
    therefore not evidence of this criterion (see known_pathogenic_residues module
    docstrings for the analogous BP1 reasoning)."""
    code = "BS2"
    version = "bs2_v3.0_narrative"
    fe = context.family_evidence
    if fe is None or fe.unaffected_carrier_observed is None:
        return _not_evaluated(
            code, EvidenceDirection.BENIGN,
            "No statement is available from the clinical narrative about an "
            "unaffected relative carrying this variant's genotype state.",
            ["family_evidence.unaffected_carrier_observed"], version,
        )
    evidence = [EvidenceItem(
        evidence_id=f"family_evidence:{context.variant_id}:bs2", evidence_type="family_history",
        source="clinical narrative (demographics, llm-extracted)",
        value={"unaffected_carrier_observed": fe.unaffected_carrier_observed,
               "unaffected_carrier_adult": fe.unaffected_carrier_adult},
    )]
    if not fe.unaffected_carrier_observed:
        return _result(
            code, CriterionStatus.NOT_MET, EvidenceDirection.BENIGN,
            "Clinical narrative does not report an unaffected relative carrying "
            "this genotype state.",
            evidence_items=evidence, evaluator_version=version,
        )
    if fe.unaffected_carrier_adult is not True:
        return _not_evaluated(
            code, EvidenceDirection.BENIGN,
            "An unaffected carrier is reported, but the narrative does not "
            "confirm they are an adult / past the disease's expected age of "
            "onset -- BS2's 'full penetrance expected at an early age' premise "
            "cannot be confirmed without this, so MET cannot be assigned.",
            ["family_evidence.unaffected_carrier_adult"], version,
        )
    return _result(
        code, CriterionStatus.MET, EvidenceDirection.BENIGN,
        "Clinical narrative reports an unaffected adult relative carrying the "
        "same genotype state (homozygous/hemizygous/heterozygous as relevant to "
        "the inheritance pattern) without manifesting the disease past its "
        "expected age of onset.",
        strength=EvidenceStrength.STRONG, evidence_items=evidence,
        evaluator_version=version,
    )


_PM3_POINTS = {
    "second_pathogenic_phase_unknown": 0.5,
    "second_likely_pathogenic_phase_unknown": 0.25,
    "second_vus_phase_unknown": 0.0,
    "homozygous": 0.5,
    "homozygous_consanguineous": 0.25,
}

_PM3_THRESHOLDS: list[tuple[float, EvidenceStrength]] = [
    (4.0, EvidenceStrength.VERY_STRONG),
    (2.0, EvidenceStrength.STRONG),
    (1.0, EvidenceStrength.MODERATE),
    (0.5, EvidenceStrength.SUPPORTING),
]


def _pm3_strength(points: float) -> EvidenceStrength | None:
    for threshold, strength in _PM3_THRESHOLDS:
        if points >= threshold:
            return strength
    return None


def _pm3(variant: AnnotatedVariant, context: VariantDiseaseContext) -> CriterionResult:
    code = "PM3"
    version = "pm3_v1.0_points_phase_unknown_only"
    if context.inheritance not in _RECESSIVE_INHERITANCE_MODELS:
        return _not_applicable(
            code, EvidenceDirection.PATHOGENIC,
            "PM3 applies to recessive (or X-linked recessive) disorders only.",
            version,
        )

    points = 0.0
    evidence: list[EvidenceItem] = []
    contributed: list[str] = []
    missing: list[str] = []

    consang = context.family_evidence.consanguinity if context.family_evidence else None
    if context.zygosity == "homozygous":
        if consang:
            points += _PM3_POINTS["homozygous_consanguineous"]
            contributed.append(
                f"target variant homozygous, consanguinity stated "
                f"(+{_PM3_POINTS['homozygous_consanguineous']})"
            )
        else:
            points += _PM3_POINTS["homozygous"]
            contributed.append(f"target variant homozygous (+{_PM3_POINTS['homozygous']})")
        evidence.append(EvidenceItem(
            evidence_id=f"pm3_homozygosity:{context.variant_id}", evidence_type="zygosity",
            source="variant annotation + clinical narrative (consanguinity)",
            value={"zygosity": context.zygosity, "consanguinity": consang},
        ))
    elif context.zygosity is None:
        missing.append("zygosity")

    scored_others = [a for a in context.other_alleles if a.classification_support]
    if scored_others:
        best_label = None
        best_points = 0.0
        for allele in scored_others:
            cls = allele.classification_support.lower()
            if cls == "pathogenic":
                p = _PM3_POINTS["second_pathogenic_phase_unknown"]
            elif cls == "likely_pathogenic":
                p = _PM3_POINTS["second_likely_pathogenic_phase_unknown"]
            elif cls in ("vus", "uncertain_significance"):
                p = _PM3_POINTS["second_vus_phase_unknown"]
            else:
                continue
            if p >= best_points:
                best_points, best_label = p, (allele.variant_id, cls)
        if best_label is not None:
            points += best_points
            if best_points:
                contributed.append(
                    f"second variant {best_label[0]} ({best_label[1]}, phase unknown) "
                    f"(+{best_points})"
                )
            evidence.append(EvidenceItem(
                evidence_id=f"pm3_trans_partner:{context.variant_id}:{best_label[0]}",
                evidence_type="allelic_evidence",
                source="variant_disease_context.other_alleles (phase unknown)",
                value={"variant_id": best_label[0], "classification_support": best_label[1]},
            ))
    else:
        missing.append("second-allele classification_support (never populated by the "
                        "current pipeline -- see code comment)")

    if not contributed:
        return _not_evaluated(
            code, EvidenceDirection.PATHOGENIC,
            "No PM3-scorable evidence is available for this variant-disease pair "
            "(missing: " + "; ".join(missing) + ").",
            missing, version,
        )

    strength = _pm3_strength(points)
    if strength is None:
        return _result(
            code, CriterionStatus.NOT_MET, EvidenceDirection.PATHOGENIC,
            f"PM3 point total ({points}) is below the 0.5 threshold for any "
            f"strength tier ({'; '.join(contributed)}).",
            evidence_items=evidence, evaluator_version=version,
        )
    return _result(
        code, CriterionStatus.MET, EvidenceDirection.PATHOGENIC,
        f"PM3 point total {points} ({'; '.join(contributed)}) reaches the "
        f"{strength.value} tier of the ClinGen SVI PM3 points framework -- "
        "phase-unknown branch only, single-case (not multi-proband-aggregated).",
        strength=strength, evidence_items=evidence, evaluator_version=version,
    )


def _pp4(context: VariantDiseaseContext) -> CriterionResult:
    code = "PP4"
    version = "pp4_v1.0"
    if context.phenotype_match_score is None:
        return _not_evaluated(
            code, EvidenceDirection.PATHOGENIC,
            "No phenotype-match score is available for this gene-disease pair.",
            ["phenotype_match_score"], version,
        )
    evidence = [EvidenceItem(
        evidence_id=f"phenotype:{context.disease_id}:pp4", evidence_type="phenotype_match",
        source="GeneResolver phenotype matcher",
        value={"phenotype_match_score": context.phenotype_match_score},
    )]
    if context.phenotype_match_score >= _PP4_PHENOTYPE_MATCH_THRESHOLD:
        return _result(
            code, CriterionStatus.MET, EvidenceDirection.PATHOGENIC,
            f"Phenotype-match score ({context.phenotype_match_score}) meets the "
            f"'strong phenotype overlap' threshold ({_PP4_PHENOTYPE_MATCH_THRESHOLD}) "
            "already used elsewhere in this pipeline (gene_ranker.py) -- a "
            "best-effort phenotype-fit heuristic, not the guideline's stricter "
            "\"highly specific for a single diagnosis\" bar.",
            strength=EvidenceStrength.SUPPORTING, evidence_items=evidence,
            evaluator_version=version,
        )
    return _result(
        code, CriterionStatus.NOT_MET, EvidenceDirection.PATHOGENIC,
        f"Phenotype-match score ({context.phenotype_match_score}) does not meet "
        f"the strong-overlap threshold ({_PP4_PHENOTYPE_MATCH_THRESHOLD}).",
        evidence_items=evidence, evaluator_version=version,
    )



_DEPRECATED_CODES = {
    "PP5": "Deprecated per ClinGen SVI (2018): a ClinVar-reported reviewer "
           "assessment is not independent evidence and must not be used as a "
           "stand-alone criterion.",
    "BP6": "Deprecated per ClinGen SVI (2018): a ClinVar-reported reviewer "
           "assessment is not independent evidence and must not be used as a "
           "stand-alone criterion.",
}

_NOT_YET_IMPLEMENTED: dict[str, tuple[EvidenceDirection, str]] = {
    "PS3": (EvidenceDirection.PATHOGENIC, "well-established functional study data (functional_evidence)"),
    "PS4": (EvidenceDirection.PATHOGENIC, "case-control prevalence data (case_control_evidence)"),
    "BS3": (EvidenceDirection.BENIGN, "well-established functional study showing no damaging effect (functional_evidence)"),
    "BP2": (EvidenceDirection.BENIGN, "observed in trans/cis with a pathogenic variant (other_alleles + phase)"),
    "BP5": (EvidenceDirection.BENIGN, "an alternate molecular basis for disease in this case (alternative_diagnoses)"),
}

def _placeholder_results() -> list[CriterionResult]:
    results = []
    for code, rationale in _DEPRECATED_CODES.items():
        results.append(_not_applicable(code, EvidenceDirection.PATHOGENIC
                                        if code == "PP5" else EvidenceDirection.BENIGN,
                                        rationale, "deprecated_v1.0"))
    for code, (direction, requirement) in _NOT_YET_IMPLEMENTED.items():
        results.append(_not_evaluated(
            code, direction, f"Not yet implemented in this engine: requires {requirement}.",
            [requirement], "not_implemented_v1.0",
        ))
    return results



def check_independence(criteria: list[CriterionResult]) -> list[Conflict]:
    met = [c for c in criteria if c.status == CriterionStatus.MET]
    conflicts: list[Conflict] = []

    evidence_owner: dict[str, str] = {}
    for c in met:
        for ei in c.evidence_items:
            owner = evidence_owner.get(ei.evidence_id)
            if owner and owner != c.code:
                conflicts.append(Conflict(
                    criteria=[owner, c.code],
                    reason=f"both criteria cite the same evidence_id '{ei.evidence_id}'",
                ))
            evidence_owner.setdefault(ei.evidence_id, c.code)

    codes_met = {c.code for c in met}
    if "PVS1" in codes_met and "PM4" in codes_met:
        conflicts.append(Conflict(
            criteria=["PVS1", "PM4"],
            reason="PM4 must not duplicate a consequence already captured by PVS1",
        ))
    return conflicts


_POINTS = {
    EvidenceStrength.SUPPORTING: 1,
    EvidenceStrength.MODERATE: 2,
    EvidenceStrength.STRONG: 4,
    EvidenceStrength.VERY_STRONG: 8,
}


def _signed_points(c: CriterionResult) -> int:
    base = _POINTS.get(c.strength, 0) if c.strength else 0
    return base if c.direction == EvidenceDirection.PATHOGENIC else -base


def classify_points(points: int) -> str:
    if points >= 10:
        return "pathogenic"
    if 6 <= points <= 9:
        return "likely_pathogenic"
    if 0 <= points <= 5:
        return "vus"
    if -6 <= points <= -1:
        return "likely_benign"
    return "benign"


def _valid_ba1_stand_alone(met: list[CriterionResult]) -> bool:
    ba1 = next((c for c in met if c.code == "BA1"), None)
    if ba1 is None:
        return False
    strong_pathogenic = [
        c for c in met
        if c.direction == EvidenceDirection.PATHOGENIC
        and c.strength in (EvidenceStrength.STRONG, EvidenceStrength.VERY_STRONG)
    ]
    return not strong_pathogenic


def _exclude_duplicated_evidence(
    met: list[CriterionResult], conflicts: list[Conflict],
) -> list[CriterionResult]:
    excluded: set[str] = set()
    for conflict in conflicts:
        pair = [c for c in met if c.code in conflict.criteria and c.code not in excluded]
        if len(pair) < 2:
            continue
        pair.sort(key=lambda c: _POINTS.get(c.strength, 0), reverse=True)
        for c in pair[1:]:
            excluded.add(c.code)
    return [c for c in met if c.code not in excluded]


def combine(criteria: list[CriterionResult]) -> CombinationResult:
    met = [c for c in criteria if c.status == CriterionStatus.MET]
    conflicts = check_independence(met)

    if _valid_ba1_stand_alone(met):
        return CombinationResult(
            classification="benign", total_points=-8, basis="BA1_stand_alone",
            conflicts=conflicts,
            counted_criteria=[c.code for c in met if c.code == "BA1"],
            excluded_criteria=[c.code for c in met if c.code != "BA1"],
        )

    counted = _exclude_duplicated_evidence(met, conflicts)
    counted_codes = {c.code for c in counted}
    points = sum(_signed_points(c) for c in counted)
    classification = classify_points(points)

    pathogenic_met = [c for c in counted if c.direction == EvidenceDirection.PATHOGENIC]
    benign_met = [c for c in counted if c.direction == EvidenceDirection.BENIGN]
    conflict_status = "unresolved" if (pathogenic_met and benign_met) else None

    if classification in ("pathogenic", "likely_pathogenic") and any(
        c.code == "BA1" for c in counted
    ):
        classification = "vus"
        conflict_status = "unresolved"

    return CombinationResult(
        classification=classification, total_points=points, basis="Tavtigian_2020",
        conflicts=conflicts, conflict_status=conflict_status,
        counted_criteria=[c.code for c in counted],
        excluded_criteria=[c.code for c in met if c.code not in counted_codes],
    )


def _valid_gene_disease_context(context: VariantDiseaseContext) -> bool:
    return bool(context.disease_id and context.transcript and context.gene)


def _context_integrity_failed(variant: AnnotatedVariant, context: VariantDiseaseContext) -> bool:
    if variant.variant_id != context.variant_id:
        return True
    if variant.gene and context.gene and variant.gene != context.gene:
        return True
    if variant.transcript and context.transcript and variant.transcript != context.transcript:
        return True
    return False


def _technical_annotation_failed(variant: AnnotatedVariant) -> bool:
    if variant.consequence is None and not variant.all_consequence_terms:
        return True
    return False


def _not_classifiable(interpretation_id: str, context: VariantDiseaseContext, reason: str) -> ACMGAssessment:
    return ACMGAssessment(
        interpretation_id=interpretation_id,
        variant_id=context.variant_id, gene=context.gene, disease_id=context.disease_id,
        transcript=context.transcript, inheritance=context.inheritance,
        disease_mechanism=context.disease_mechanism,
        classification="not_classifiable", total_points=None,
        classification_status="not_classifiable",
        ruleset={"foundation": _RULESET_FOUNDATION, "general_refinements": _RULESET_GENERAL_REFINEMENTS,
                 "vcep_specification": context.vcep_specification},
        resource_versions=dict(context.source_versions),
        warnings=[reason],
        requires_expert_review=True,
    )


def assess(variant_annotation: AnnotatedVariant, context: VariantDiseaseContext) -> ACMGAssessment:
    interpretation_id = f"{context.variant_id}|{context.gene}|{context.disease_id}|{context.transcript}"

    if not _valid_gene_disease_context(context):
        return _not_classifiable(
            interpretation_id, context,
            "Required gene-disease-transcript interpretation context is not established.",
        )
    if _context_integrity_failed(variant_annotation, context):
        return _not_classifiable(
            interpretation_id, context,
            f"Context integrity check failed: the annotation "
            f"(variant_id={variant_annotation.variant_id!r}, gene={variant_annotation.gene!r}, "
            f"transcript={variant_annotation.transcript!r}) does not match the "
            f"interpretation context (variant_id={context.variant_id!r}, "
            f"gene={context.gene!r}, transcript={context.transcript!r}) -- "
            "refusing to continue with a mismatched variant/gene/transcript.",
        )
    if _technical_annotation_failed(variant_annotation):
        return _not_classifiable(
            interpretation_id, context,
            "Required technical annotation (a molecular consequence) is missing.",
        )

    pvs1_result = _pvs1(variant_annotation, context)
    criteria = [
        pvs1_result,
        _pm2_supporting(variant_annotation, context),
        _ba1(variant_annotation, context),
        _bs1(variant_annotation, context),
        _pp3_bp4(variant_annotation, context, pvs1_result),
        _pm4(variant_annotation, context, pvs1_result),
        _bp3(variant_annotation, context),
        _bp7(variant_annotation, context),
        _pp2(variant_annotation, context),
        _bp1(variant_annotation, context),
        _ps2(variant_annotation, context),
        _pm6(variant_annotation, context),
        _pp1(variant_annotation, context),
        _bs4(variant_annotation, context),
        _ps1(variant_annotation, context),
        _pm5(variant_annotation, context),
        _pm1(variant_annotation, context),
        _bs2(variant_annotation, context),
        _pm3(variant_annotation, context),
        _pp4(context),
        *_placeholder_results(),
    ]

    combination = combine(criteria)

    pathogenic_criteria = [c.code for c in criteria
                           if c.status == CriterionStatus.MET and c.direction == EvidenceDirection.PATHOGENIC]
    benign_criteria = [c.code for c in criteria
                       if c.status == CriterionStatus.MET and c.direction == EvidenceDirection.BENIGN]
    criteria_not_met = [c.code for c in criteria if c.status == CriterionStatus.NOT_MET]
    criteria_not_applicable = [c.code for c in criteria if c.status == CriterionStatus.NOT_APPLICABLE]
    criteria_not_evaluated = [c.code for c in criteria if c.status == CriterionStatus.NOT_EVALUATED]
    criteria_conflicting = [c.code for c in criteria if c.status == CriterionStatus.CONFLICTING]

    warnings: list[str] = []
    if combination.conflict_status == "unresolved":
        warnings.append(
            "Pathogenic and benign evidence both met and were not cancelled -- "
            "requires expert review to resolve."
        )
    for conflict in combination.conflicts:
        warnings.append(f"Evidence-independence conflict: {conflict.reason} ({', '.join(conflict.criteria)}).")

    resource_versions: dict[str, str] = dict(context.source_versions)
    for c in criteria:
        resource_versions.update(c.source_versions)

    return ACMGAssessment(
        interpretation_id=interpretation_id,
        variant_id=context.variant_id, gene=context.gene, disease_id=context.disease_id,
        transcript=context.transcript, inheritance=context.inheritance,
        disease_mechanism=context.disease_mechanism,
        classification=combination.classification, total_points=combination.total_points,
        classification_status="provisional",
        criteria=criteria,
        pathogenic_criteria=pathogenic_criteria, benign_criteria=benign_criteria,
        criteria_not_met=criteria_not_met, criteria_not_applicable=criteria_not_applicable,
        criteria_not_evaluated=criteria_not_evaluated, criteria_conflicting=criteria_conflicting,
        counted_criteria=combination.counted_criteria,
        excluded_dependent_criteria=combination.excluded_criteria,
        combination_model=combination.basis,
        ruleset={"foundation": _RULESET_FOUNDATION, "general_refinements": _RULESET_GENERAL_REFINEMENTS,
                 "vcep_specification": context.vcep_specification},
        resource_versions=resource_versions,
        warnings=warnings,
        requires_expert_review=True,
    )


def _clinvar_significance_bucket(label: str) -> str | None:
    label = label.lower()
    if "conflicting" in label:
        return "conflicting"
    if "uncertain" in label or "vus" in label:
        return "vus"
    if "likely pathogenic" in label or "likely_pathogenic" in label:
        return "likely_pathogenic"
    if "pathogenic" in label:
        return "pathogenic"
    if "likely benign" in label or "likely_benign" in label:
        return "likely_benign"
    if "benign" in label:
        return "benign"
    return None


_CONDITION_STOPWORDS = {"disease", "disorder", "syndrome", "deficiency", "the", "of", "type", "and"}


def _condition_names_correspond(disease_name: str | None, clinvar_condition: str | None) -> bool:
    if not disease_name or not clinvar_condition:
        return False
    def _tokens(s: str) -> set[str]:
        return {
            w for w in re.findall(r"[a-z0-9]+", s.lower())
            if w not in _CONDITION_STOPWORDS and len(w) > 2
        }
    disease_tokens = _tokens(disease_name)
    if not disease_tokens:
        return False
    for condition in clinvar_condition.split("; "):
        condition_tokens = _tokens(condition)
        if not condition_tokens:
            continue
        overlap = disease_tokens & condition_tokens
        if len(overlap) >= min(2, len(disease_tokens)):
            return True
    return False


def _concordance(
    classification: str, clinvar: ClinVarExternalAnnotation | None, context: VariantDiseaseContext,
) -> tuple[str, str]:
    if clinvar is None or not clinvar.significance:
        return "unknown", "no_clinvar_data"
    if not _condition_names_correspond(context.disease_name, clinvar.condition_name):
        return "unknown", "condition_not_confirmed"

    clinvar_bucket = _clinvar_significance_bucket(clinvar.significance)
    if clinvar_bucket is None:
        return "unknown", "clinvar_significance_unparseable"
    if clinvar_bucket == "conflicting":
        return "unknown", "clinvar_conflicting_interpretations"
    return ("concordant" if classification == clinvar_bucket else "discordant"), "significance_compared"


def assess_with_clinvar(
    variant_annotation: AnnotatedVariant, context: VariantDiseaseContext,
) -> ACMGComparisonResult:
    computed = assess(variant_annotation, context)

    clinvar_ext: ClinVarExternalAnnotation | None = None
    if variant_annotation.clinvar_significance:
        clinvar_ext = ClinVarExternalAnnotation(
            significance=variant_annotation.clinvar_significance,
            review_status=variant_annotation.clinvar_review_status,
            condition_name=variant_annotation.clinvar_condition,
            variation_id=variant_annotation.clinvar_id,
            snapshot=None,
        )

    concordance, concordance_basis = _concordance(computed.classification, clinvar_ext, context)
    return ACMGComparisonResult(
        computed_acmg_assessment=computed,
        clinvar_external_annotation=clinvar_ext,
        concordance=concordance,
        concordance_basis=concordance_basis,
    )
