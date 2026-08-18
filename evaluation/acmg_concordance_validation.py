from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from generesolver.components import acmg_engine
from generesolver.components.variant_disease_context_builder import (
    _gene_disease_pairs,
)
from generesolver.integrations import gnomad_api, vep_api
from generesolver.schemas.variant import AnnotatedVariant
from generesolver.schemas.variant_disease_context import VariantDiseaseContext

BENCHMARK_PATH = Path(__file__).parent.parent / "external" / "davp" / "benchmarks" / "clinvar.jsonl"
RESULTS_PATH = Path(__file__).parent / "acmg_concordance_results.json"

_GOLD_MAP = {
    "pathogenic": "pathogenic",
    "likely_pathogenic": "likely_pathogenic",
    "uncertain_significance": "vus",
    "likely_benign": "likely_benign",
    "benign": "benign",
}

_REVSTAT_STARS = {1: "criteria_provided_single_submitter",
                   2: "criteria_provided_multiple_submitters_no_conflicts",
                   3: "reviewed_by_expert_panel",
                   4: "practice_guideline"}

_CRITERIA_OF_INTEREST = ["PVS1", "PM2", "BA1", "BS1", "PP3/BP4", "PP3", "BP4", "PM4", "BP7"]


def _load_benchmark(limit: int | None) -> list[dict]:
    records = []
    with BENCHMARK_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            clnsig = (r.get("clnsig") or [None])[0]
            gold = _GOLD_MAP.get(clnsig)
            if gold is None or not r.get("genes"):
                continue
            records.append({
                "variant_id": f"{r['contig']}:{r['pos']}:{r['ref']}:{r['alt']}",
                "chrom": r["contig"], "pos": r["pos"], "ref": r["ref"], "alt": r["alt"],
                "gene": r["genes"][0],
                "diseases": r.get("diseases") or [],
                "gold": gold,
                "revstat": _REVSTAT_STARS.get(r.get("clnrevstat"), "unknown"),
            })
            if limit and len(records) >= limit:
                break
    return records


def _pick_disease_context(gene: str, gold_diseases: list[str]) -> tuple[str, str | None, str | None]:
    pairs = _gene_disease_pairs(gene)
    if not pairs:
        return f"unmatched:{gene}", (gold_diseases[0].replace("_", " ") if gold_diseases else None), None
    gold_text = "; ".join(d.replace("_", " ") for d in gold_diseases)
    for disease_id, disease_name, mechanism in pairs:
        if acmg_engine._condition_names_correspond(disease_name, gold_text):
            return disease_id, disease_name, mechanism
    disease_id, disease_name, mechanism = pairs[0]
    return disease_id, disease_name, mechanism


def _build_variant(rec: dict, vep: dict) -> AnnotatedVariant:
    gnomad_af = vep.get("gnomad_af")
    gnomad_af_popmax = vep.get("gnomad_af_popmax")
    gnomad_af_popmax_allele_number = vep.get("gnomad_af_popmax_allele_number")
    gnomad_frequency_status = vep.get("gnomad_frequency_status") or "not_evaluated"
    if gnomad_af_popmax_allele_number is None:
        gn = gnomad_api.get_allele_frequency(rec["chrom"], rec["pos"], rec["ref"], rec["alt"], "GRCh38")
        if gnomad_af is None:
            gnomad_af = gn.get("af")
            gnomad_frequency_status = "observed" if gnomad_af is not None else gnomad_frequency_status
        if gn.get("af_popmax_allele_number") is not None:
            gnomad_af_popmax = gn.get("af_popmax")
            gnomad_af_popmax_allele_number = gn.get("af_popmax_allele_number")

    return AnnotatedVariant(
        variant_id=rec["variant_id"],
        gene=vep.get("gene") or rec["gene"],
        transcript=vep.get("transcript"),
        hgvs_c=vep.get("hgvs_c"), hgvs_p=vep.get("hgvs_p"),
        consequence=vep.get("consequence") or vep.get("most_severe_consequence"),
        all_consequence_terms=vep.get("all_consequence_terms") or [],
        is_splice_relevant=bool(vep.get("is_splice_relevant")),
        mane_status=vep.get("mane_status"),
        protein_position=vep.get("protein_position"),
        exon=vep.get("exon"), intron=vep.get("intron"),
        impact=vep.get("impact"),
        clinvar_significance=rec["gold"],
        clinvar_condition="; ".join(d.replace("_", " ") for d in rec["diseases"]),
        clinvar_review_status=rec["revstat"],
        clinvar_source="davp_clinvar_benchmark_gold",
        gnomad_af=gnomad_af,
        gnomad_af_popmax=gnomad_af_popmax,
        gnomad_af_by_population=vep.get("gnomad_af_by_population") or {},
        gnomad_allele_count=vep.get("gnomad_allele_count"),
        gnomad_allele_number=vep.get("gnomad_allele_number"),
        gnomad_af_popmax_allele_number=gnomad_af_popmax_allele_number,
        gnomad_homozygote_count=vep.get("gnomad_homozygote_count"),
        gnomad_filter_status=vep.get("gnomad_filter_status"),
        gnomad_coverage=vep.get("gnomad_coverage"),
        gnomad_frequency_status=gnomad_frequency_status,
        sift_prediction=vep.get("sift"), sift_score=vep.get("sift_score"),
        polyphen_prediction=vep.get("polyphen"), polyphen_score=vep.get("polyphen_score"),
        revel_score=vep.get("revel_score"), spliceai_score=vep.get("spliceai_score"),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None, help="Only score the first N gold variants.")
    args = p.parse_args()

    records = _load_benchmark(args.limit)
    print(f"Loaded {len(records)} gold-labeled ClinVar variant(s) from {BENCHMARK_PATH}")
    print("Gold label distribution:", dict(Counter(r["gold"] for r in records)))
    print("NOTE: gold set is 100% Pathogenic/Likely Pathogenic -- this run measures "
          "sensitivity/recall only, not specificity.\n")

    print(f"Annotating {len(records)} variant(s) via Ensembl VEP REST...")
    vep_results = vep_api.annotate_variants(
        [(r["chrom"], r["pos"], r["ref"], r["alt"]) for r in records]
    )

    classification_counts = Counter()
    criterion_status_counts = {c: Counter() for c in _CRITERIA_OF_INTEREST}
    pvs1_not_evaluated_reasons = Counter()
    engine_concordance_counts = Counter()
    raw_agree = raw_disagree_benign_direction = raw_abstain = 0
    per_variant_results = []

    for rec in records:
        key1 = f"{rec['chrom'].lstrip('chr')}:{rec['pos']}:{rec['ref']}:{rec['alt']}"
        vep = vep_results.get(key1) or vep_results.get(rec["variant_id"]) or {}
        variant = _build_variant(rec, vep)

        if not variant.gene or not variant.transcript:
            classification_counts["skipped_no_annotation"] += 1
            per_variant_results.append({**rec, "computed": "skipped_no_annotation"})
            continue

        disease_id, disease_name, mechanism = _pick_disease_context(variant.gene, rec["diseases"])
        context = VariantDiseaseContext(
            variant_id=variant.variant_id, gene=variant.gene,
            disease_id=disease_id, disease_name=disease_name,
            transcript=variant.transcript, genome_build="GRCh38",
            ruleset_version="ACMG-AMP-2015+ClinGen-SVI",
            disease_mechanism=mechanism,
        )

        comparison = acmg_engine.assess_with_clinvar(variant, context)
        computed = comparison.computed_acmg_assessment
        classification_counts[computed.classification] += 1
        engine_concordance_counts[comparison.concordance] += 1

        for c in computed.criteria:
            if c.code in criterion_status_counts:
                criterion_status_counts[c.code][c.status.value] += 1
            if c.code == "PVS1" and c.status.value == "not_evaluated":
                for req in (c.missing_requirements or ["(unspecified)"]):
                    pvs1_not_evaluated_reasons[req] += 1

        if computed.classification in ("pathogenic", "likely_pathogenic"):
            raw_agree += 1
        elif computed.classification in ("likely_benign", "benign"):
            raw_disagree_benign_direction += 1
        else:
            raw_abstain += 1

        per_variant_results.append({
            **rec, "computed": computed.classification,
            "counted_criteria": computed.counted_criteria,
            "engine_concordance": comparison.concordance,
            "engine_concordance_basis": comparison.concordance_basis,
            "gene_matched": variant.gene, "disease_context": disease_name,
        })

    n_scored = sum(classification_counts.values()) - classification_counts.get("skipped_no_annotation", 0)

    print("\n" + "=" * 78)
    print("COMPUTED CLASSIFICATION DISTRIBUTION (all gold-P/LP variants)")
    print("=" * 78)
    for cls, n in classification_counts.most_common():
        print(f"  {cls:<25} {n:>4}")

    print("\n" + "=" * 78)
    print("RAW SENSITIVITY READ (direction-only, ignores disease-context matching)")
    print("=" * 78)
    print(f"  n scored                                    : {n_scored}")
    print(f"  computed pathogenic/likely_pathogenic (agree): {raw_agree}  "
          f"({raw_agree / n_scored:.1%})" if n_scored else "")
    print(f"  computed vus/not_classifiable (abstained)    : {raw_abstain}  "
          f"({raw_abstain / n_scored:.1%})" if n_scored else "")
    print(f"  computed likely_benign/benign (CONTRADICTS)  : {raw_disagree_benign_direction}  "
          f"({raw_disagree_benign_direction / n_scored:.1%})" if n_scored else "")

    print("\n" + "=" * 78)
    print("ENGINE'S OWN concordance FIELD (condition-match-gated, from assess_with_clinvar)")
    print("=" * 78)
    for k, n in engine_concordance_counts.most_common():
        print(f"  {k:<15} {n:>4}")

    print("\n" + "=" * 78)
    print("PER-CRITERION STATUS DISTRIBUTION (why classifications land where they do)")
    print("=" * 78)
    for code in _CRITERIA_OF_INTEREST:
        counts = criterion_status_counts[code]
        total = sum(counts.values())
        if total == 0:
            continue
        parts = "  ".join(f"{status}={n}" for status, n in counts.most_common())
        print(f"  {code:<8} (n={total:<4}): {parts}")

    if pvs1_not_evaluated_reasons:
        print("\nPVS1 NOT_EVALUATED breakdown (missing_requirements, one variant can hit >1):")
        for reason, n in pvs1_not_evaluated_reasons.most_common():
            print(f"  {n:>4}  {reason}")

    RESULTS_PATH.write_text(json.dumps({
        "n_gold": len(records), "n_scored": n_scored,
        "classification_distribution": dict(classification_counts),
        "raw_sensitivity": {
            "agree_pathogenic_direction": raw_agree,
            "abstained": raw_abstain,
            "contradicts_benign_direction": raw_disagree_benign_direction,
        },
        "engine_concordance": dict(engine_concordance_counts),
        "criterion_status_distribution": {
            c: dict(criterion_status_counts[c]) for c in _CRITERIA_OF_INTEREST
        },
        "per_variant": per_variant_results,
    }, indent=2), encoding="utf-8")
    print(f"\nFull per-variant results written to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
