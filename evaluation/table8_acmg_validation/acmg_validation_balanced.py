"""ACMG/AMP engine validation against ClinVar: development set and re-scoring (Table 8).

This script samples expert-reviewed (>=2-star) Pathogenic,
Likely pathogenic, Likely benign and Benign coding/splice variants from the local ClinVar
VCF and reports:

  * the 4x5 confusion matrix (gold ClinVar class x engine class, VUS included),
  * direction-level agreement (P/LP->P/LP, B/LB->B/LB), wrong-direction calls and abstention,
  * per-criterion MET rates by gold class (criterion calibration),
  * a second scoring that removes PS1, PM5 and PM1. Those three draw on ClinVar itself, so
    a ClinVar variant can count as evidence for itself.

Only variant-intrinsic evidence is available here (no phenotype, segregation, de novo or
functional data), so the result describes the evidence layer the pipeline can compute
automatically, not a full clinical classification.

Variants are annotated exactly as variant_annotator does in the evaluated (Stage 5)
configuration: GRCh37, local Docker VEP, REVEL, slivar gnomAD-popmax/SpliceAI, phyloP and the
local ClinVar VCF, but without the candidate pre-filter (common benign variants must be kept).

    set GENERESOLVER_LOCAL_MODE=1
    python -m evaluation.table8_acmg_validation.acmg_validation_balanced --per-class 100   # sample + score
    python -m evaluation.table8_acmg_validation.acmg_validation_balanced --rescore data_creation/acmg_validation/dev_400.json --tag <tag>
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from generesolver.components import acmg_engine
from generesolver.components import variant_annotator as va
from generesolver.components.variant_disease_context_builder import _gene_disease_pairs
from generesolver.components.vcf_input import NormalizedVariant, _trim_variant
from generesolver.integrations import conservation_local, gnomad_popmax_local, vep_api
from generesolver.schemas.acmg_assessment import CriterionStatus
from generesolver.schemas.variant_disease_context import VariantDiseaseContext

CLINVAR_VCF = ROOT / "data" / "clinvar" / "clinvar_GRCh38.vcf.gz"  # set from --build in main()
OUT = ROOT / "evaluation" / "paper_results" / "acmg_validation"


def _pick_disease_context(gene: str, gold_diseases: list[str]) -> tuple[str, str | None, str | None]:
    """The gene-disease pair whose name matches the ClinVar condition, else the gene's first pair."""
    pairs = _gene_disease_pairs(gene)
    if not pairs:
        return f"unmatched:{gene}", (gold_diseases[0].replace("_", " ") if gold_diseases else None), None
    gold_text = "; ".join(d.replace("_", " ") for d in gold_diseases)
    for disease_id, disease_name, mechanism in pairs:
        if acmg_engine._condition_names_correspond(disease_name, gold_text):
            return disease_id, disease_name, mechanism
    return pairs[0]

GOLD = {"Pathogenic": "pathogenic", "Likely_pathogenic": "likely_pathogenic",
        "Likely_benign": "likely_benign", "Benign": "benign"}
TWO_STAR_PLUS = {"criteria_provided,_multiple_submitters,_no_conflicts",
                 "reviewed_by_expert_panel", "practice_guideline"}
CODING_MC = {"missense_variant", "nonsense", "stop_gained", "frameshift_variant", "splice_donor_variant",
             "splice_acceptor_variant", "inframe_deletion", "inframe_insertion", "inframe_indel",
             "synonymous_variant", "start_lost", "stop_lost", "initiator_codon_variant"}
CLINVAR_DERIVED = {"PS1", "PM5", "PM1"}
ENGINE_CLASSES = ("pathogenic", "likely_pathogenic", "vus", "likely_benign", "benign", "not_classifiable")


def _info(field: str) -> dict[str, str]:
    return dict(kv.split("=", 1) for kv in field.split(";") if "=" in kv)


def sample_clinvar(per_class: int, seed: int, exclude: set[str] = frozenset()) -> list[dict]:
    rng = random.Random(seed)
    reservoirs: dict[str, list[dict]] = {g: [] for g in GOLD.values()}
    seen: Counter = Counter()
    with gzip.open(CLINVAR_VCF, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            chrom, pos, vid, ref, alt, _, _, info = line.rstrip("\n").split("\t")[:8]
            if len(ref) > 50 or len(alt) > 50 or alt == ".":
                continue
            if f"{chrom}:{pos}:{ref}:{alt}" in exclude:
                continue
            d = _info(info)
            gold = GOLD.get(d.get("CLNSIG", ""))
            if not gold or d.get("CLNREVSTAT") not in TWO_STAR_PLUS:
                continue
            genes = [g.split(":")[0] for g in d.get("GENEINFO", "").split("|") if g]
            if len(genes) != 1:
                continue
            mcs = {m.split("|")[-1] for m in d.get("MC", "").split(",") if "|" in m}
            if not mcs & CODING_MC:
                continue
            rec = {"variant_id": f"{chrom}:{pos}:{ref}:{alt}", "chrom": chrom, "pos": int(pos),
                   "ref": ref, "alt": alt, "gene": genes[0],
                   "diseases": [x for x in d.get("CLNDN", "").split("|") if x and x != "not_provided"],
                   "gold": gold, "revstat": d["CLNREVSTAT"], "clinvar_id": vid,
                   "clinvar_mc": sorted(mcs & CODING_MC)}
            seen[gold] += 1
            res = reservoirs[gold]
            if len(res) < per_class * 3:
                res.append(rec)
            else:
                j = rng.randrange(seen[gold])
                if j < len(res):
                    res[j] = rec
    out = []
    for gold, res in reservoirs.items():
        rng.shuffle(res)
        kept = [r for r in res if _gene_disease_pairs(r["gene"])][:per_class]
        out.extend(kept)
        print(f"  {gold:18s} eligible={seen[gold]:>7}  sampled={len(kept)}")
    return out


def annotate_like_pipeline(records: list[dict], build: str) -> dict[str, "va.AnnotatedVariant"]:
    raws = []
    for r in records:
        pos, ref, alt, changed = _trim_variant(r["pos"], r["ref"], r["alt"])
        raws.append((r["variant_id"], NormalizedVariant(
            chrom=r["chrom"], pos=pos, ref=ref, alt=alt, genotype="0/1", zygosity="heterozygous",
            filter_status="PASS", was_indel_normalized=changed)))
    keys = [(v.chrom, v.pos, v.ref, v.alt) for _, v in raws]
    vep_results = vep_api.annotate_variants(keys, genome_build=build)
    slivar_results = gnomad_popmax_local.annotate_batch(keys, build)
    conservation = conservation_local.get_conservation_batch(
        va._extract_synonymous_positions([v for _, v in raws], vep_results))
    out = {}
    for vid, rv in raws:
        vep = va._get_vep_result(rv, vep_results)
        slivar = slivar_results.get(va._normalize_variant_key(rv.chrom, rv.pos, rv.ref, rv.alt)) or {}
        out[vid] = va._create_annotated_variant(rv, vep, slivar, conservation, build)
    return out


def _clinvar_derived(c) -> bool:
    """PS1/PM5/PM1, and PVS1 when its LoF mechanism came from ClinVar gene-level evidence."""
    return c.code in CLINVAR_DERIVED or (c.code == "PVS1"
                                          and acmg_engine.CLINVAR_GENE_LOF_MARKER in (c.rationale or ""))


def direction(cls: str) -> str:
    return {"pathogenic": "P", "likely_pathogenic": "P", "benign": "B", "likely_benign": "B"}.get(cls, "U")


def summarize(results: list[dict], key: str) -> dict:
    confusion = {g: Counter() for g in GOLD.values()}
    for r in results:
        confusion[r["gold"]][r[key]] += 1
    by_dir = defaultdict(Counter)
    for r in results:
        by_dir[direction(r["gold"])][direction(r[key])] += 1
    p, b = by_dir["P"], by_dir["B"]
    return {
        "confusion": {g: {c: confusion[g][c] for c in ENGINE_CLASSES} for g in confusion},
        "pathogenic_gold": {"n": sum(p.values()), "called_P_LP": p["P"], "called_B_LB": p["B"], "vus_or_nc": p["U"]},
        "benign_gold": {"n": sum(b.values()), "called_B_LB": b["B"], "called_P_LP": b["P"], "vus_or_nc": b["U"]},
        "sensitivity_P_LP": p["P"] / sum(p.values()) if p else None,
        "specificity_B_LB": b["B"] / sum(b.values()) if b else None,
        "wrong_direction_rate": (p["B"] + b["P"]) / max(1, sum(p.values()) + sum(b.values())),
        "exact_class_agreement": sum(r[key] == r["gold"] for r in results) / max(1, len(results)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--build", choices=["GRCh37", "GRCh38"], default="GRCh37",
                        help="GRCh37 = the evaluated Stage 5 configuration (needs GENERESOLVER_LOCAL_MODE=1)")
    parser.add_argument("--exclude", type=Path, nargs="*", default=[],
                        help="earlier validation JSONs whose variants must not be resampled (held-out set)")
    parser.add_argument("--tag", default="",
                        help="output suffix, e.g. 'v2_heldout' -> acmg_validation_balanced_GRCh37_v2_heldout.json")
    parser.add_argument("--rescore", type=Path,
                        help="re-run the current engine on the variants of an earlier validation JSON")
    args = parser.parse_args()

    global CLINVAR_VCF
    CLINVAR_VCF = ROOT / "data" / "clinvar" / f"clinvar_{args.build}.vcf.gz"
    out_path = OUT / f"acmg_validation_balanced_{args.build}{'_' + args.tag if args.tag else ''}.json"
    if out_path.exists():
        sys.exit(f"{out_path} exists; choose another --tag (earlier results are never overwritten)")
    exclude = {r["variant_id"] for p in args.exclude
               for r in json.loads(p.read_text(encoding="utf-8"))["per_variant"]}
    if args.rescore:
        keep = ("variant_id", "chrom", "pos", "ref", "alt", "gene", "diseases", "gold", "revstat",
                "clinvar_id", "clinvar_mc")
        records = [{k: r[k] for k in keep}
                   for r in json.loads(args.rescore.read_text(encoding="utf-8"))["per_variant"]]
        print(f"Re-scoring {len(records)} variants from {args.rescore.name} with the current engine")
    else:
        print(f"Sampling >=2-star coding/splice ClinVar variants from {CLINVAR_VCF.name} "
              f"(seed {args.seed}, {len(exclude)} excluded) ...")
        records = sample_clinvar(args.per_class, args.seed, exclude)
    print(f"Annotating {len(records)} variant(s) as variant_annotator does ({args.build}) ...")
    annotated = annotate_like_pipeline(records, args.build)

    results = []
    assessed = []
    criterion_met = defaultdict(Counter)
    for rec in records:
        variant = annotated[rec["variant_id"]]
        if not variant.gene or not variant.transcript:
            results.append({**rec, "computed": "not_classifiable", "computed_no_clinvar_derived": "not_classifiable",
                            "consequence": None, "counted": []})
            continue
        disease_id, disease_name, mechanism = _pick_disease_context(variant.gene, rec["diseases"])
        context = VariantDiseaseContext(
            variant_id=variant.variant_id, gene=variant.gene, disease_id=disease_id, disease_name=disease_name,
            transcript=variant.transcript, genome_build=args.build,
            ruleset_version="ACMG-AMP-2015+ClinGen-SVI", disease_mechanism=mechanism,
        )
        a = acmg_engine.assess(variant, context)
        assessed.append(a)
        reduced = (acmg_engine.combine([c for c in a.criteria if not _clinvar_derived(c)]).classification
                   if a.criteria else a.classification)
        for c in a.criteria:
            if c.status == CriterionStatus.MET:
                criterion_met[c.code][rec["gold"]] += 1
        results.append({**rec, "computed": a.classification, "computed_no_clinvar_derived": reduced,
                        "consequence": variant.consequence, "counted": a.counted_criteria,
                        "points": a.total_points, "gnomad_status": variant.gnomad_frequency_status})

    gold_n = Counter(r["gold"] for r in results)
    report = {
        "n": len(results), "gold_distribution": dict(gold_n),
        "full_engine": summarize(results, "computed"),
        "without_PS1_PM5_PM1": summarize(results, "computed_no_clinvar_derived"),
        "criterion_met_rate_by_gold": {
            code: {g: round(criterion_met[code][g] / gold_n[g], 3) for g in GOLD.values() if gold_n[g]}
            for code in sorted(criterion_met)},
        "by_consequence": {
            cons: summarize([r for r in results if r["consequence"] == cons], "computed_no_clinvar_derived")
            for cons in sorted({r["consequence"] for r in results if r["consequence"]})
            if sum(r["consequence"] == cons for r in results) >= 10},
        "per_variant": results,
    }
    report["engine_versions"] = sorted({c.evaluator_version for a in assessed for c in a.criteria})
    report["sample"] = {"seed": args.seed, "excluded_from": [p.name for p in args.exclude],
                        "rescored_from": args.rescore.name if args.rescore else None}
    OUT.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name in ("full_engine", "without_PS1_PM5_PM1"):
        s = report[name]
        print(f"\n== {name} ==")
        print("  gold \\ engine      " + " ".join(f"{c[:6]:>7s}" for c in ENGINE_CLASSES))
        for g, row in s["confusion"].items():
            print(f"  {g:18s} " + " ".join(f"{row[c]:7d}" for c in ENGINE_CLASSES))
        print(f"  sensitivity P/LP={s['sensitivity_P_LP']:.3f}  specificity B/LB={s['specificity_B_LB']:.3f}  "
              f"wrong-direction={s['wrong_direction_rate']:.3f}  exact={s['exact_class_agreement']:.3f}")
    print("\nMET rate by gold class:")
    for code, rates in report["criterion_met_rate_by_gold"].items():
        print(f"  {code:8s} " + "  ".join(f"{g}={v:.2f}" for g, v in rates.items()))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
