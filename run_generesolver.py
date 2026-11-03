from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generesolver.env import load_project_env

load_project_env(Path(__file__).parent)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("GENERESOLVER_LOCAL_MODE", "1")

from generesolver.pipeline import run_case
from generesolver.schemas.case import CaseInput

EXAMPLES: dict[str, dict] = {
    "single_gene": {
        "case_id": "DEMO_CASE_SINGLE_GENE",
        "note": (
            "A 45-year-old man was admitted with kidney failure requiring hemodialysis, "
            "presenting with painful urination, anemia, proteinuria, elevated blood urea "
            "nitrogen and creatinine, high potassium and phosphate, low calcium, and muscle "
            "pains. He had a 10-year history of high blood pressure treated with medication "
            "and a gradual rise in kidney markers with sub-nephrotic proteinuria over the "
            "prior two years, but had not seen a kidney specialist. From an early age he had "
            "uncontrolled burning pain in his limbs, intermittent abdominal pain, and "
            "intolerance to both heat and cold. Kidney ultrasound showed small kidneys with "
            "increased echogenicity, and an ECG showed a fast heart rate with inverted T "
            "waves in the chest leads. He was initially discharged with a diagnosis of "
            "kidney failure requiring dialysis. Over the following months, while being "
            "evaluated for a living-donor kidney transplant, further history revealed "
            "similar symptoms (burning limb pain, abdominal cramps, and reduced sweating) "
            "in his daughter and in a maternal cousin who was also a kidney transplant "
            "recipient. Cardiac MRI showed concentric thickening of the heart consistent "
            "with hypertrophic cardiomyopathy."
        ),
        "sex": "male",
    },
    "multifactorial": {
        "case_id": "DEMO_CASE_MULTIFACTORIAL",
        "note": (
            "A 49-year-old woman presented with recent-onset tingling in one finger along "
            "with excessive thirst, most pronounced at night, drinking far more water than "
            "usual and waking six to eight times nightly to urinate. She also noted feeling "
            "unusually hungry after dinner, without nausea or vomiting. She had no family "
            "history of diabetes but did have a personal history of a pituitary tumor "
            "diagnosed in her teens without progression, and a thyroid condition treated "
            "with ablation a decade earlier; she was on a variable levothyroxine dose and "
            "noticed her heart raced on the higher-dose days. She was very physically "
            "active as a fitness instructor and reported near-fainting sensations on "
            "standing from a kneeling position. On exam her BMI was 21.7, she had lost a "
            "small amount of weight in the preceding ten days, her heart rate was mildly "
            "elevated, and the rest of her exam was unremarkable. An initial random blood "
            "glucose was significantly elevated. At follow-up five days later, basic "
            "metabolic testing ruled out a significant acid-base disturbance, urinalysis "
            "showed sugar and ketones in the urine, and a bedside serum ketone test was "
            "negative; thyroid testing showed a high TSH with low-normal free T4, "
            "consistent with under-replacement of her thyroid medication."
        ),
        "sex": "female",
    },
    "vcf_lirical": {
        "case_id": "DEMO_CASE_VCF_LIRICAL",
        "note": (
            "A 14-year-old girl presented with macrocephaly noted since infancy, followed "
            "by an acute encephalopathic crisis with dystonia and subsequent cerebral "
            "atrophy on imaging; a subdural hemorrhage was also identified. Urine organic "
            "acid analysis showed glutaric aciduria."
        ),
        "vcf": "examples/data/GCDH_Abdul_Wahab_2016_Patient5.vcf.gz",
        "sex": "female",
    },
}

IDE_EXAMPLE: str | None = "single_gene"

CUSTOM_CASE = {
    "case_id":            None,
    "note":               None,
    "note_file":          None,
    "vcf":                None,
    "dragen_dir":         None,
    "candidate_gene":     None,
    "candidate_variant":  None,
    "rsid":               None,
    "disease":            None,
    "sex":                None,
    "family_history":     None,
    "consanguinity":      False,
    "no_save":            False,
    "quiet":              False,
}


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the GeneResolver pipeline on a single case.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--example", choices=sorted(EXAMPLES), default=None,
                   help="Run one of the bundled examples instead of a custom case "
                        "(single_gene / multifactorial / vcf_lirical).")
    p.add_argument("--case-id", default=None,
                   help="Unique case identifier. Auto-generated UUID if omitted.")

    note_group = p.add_mutually_exclusive_group()
    note_group.add_argument("--note", default=None,
                            help="Clinical note as a literal string.")
    note_group.add_argument("--note-file", type=Path, default=None,
                            help="Path to a text file containing the clinical note.")

    p.add_argument("--vcf", type=Path, default=None,
                   help="Path to a VCF or VCF.GZ file (GRCh37 or GRCh38).")
    p.add_argument("--dragen-dir", type=Path, default=None,
                   help="DRAGEN output directory — auto-locates the hard-filtered VCF.")

    p.add_argument("--candidate-gene", default=None,
                   help="HGNC gene symbol to focus on (e.g. SCN1A).")
    p.add_argument("--candidate-variant", default=None,
                   help="Specific variant ID / HGVS notation.")
    p.add_argument("--rsid", default=None,
                   help="dbSNP rsID for single-locus GWAS lookup.")
    p.add_argument("--disease", default=None,
                   help="Suspected disease label.")

    p.add_argument("--family-history", default=None,
                   help="Free-text family history.")
    p.add_argument("--sex", choices=("male", "female", "unknown"), default=None)
    p.add_argument("--consanguinity", action="store_true",
                   help="Parents are consanguineous.")

    p.add_argument("--no-save", action="store_true",
                   help="Do not write output files to outputs/<case_id>/.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress info-level pipeline logging.")
    return p.parse_args()


def _config_to_namespace(cfg: dict) -> argparse.Namespace:
    get = cfg.get
    return argparse.Namespace(
        case_id=get("case_id"),
        note=get("note"),
        note_file=Path(get("note_file")) if get("note_file") else None,
        vcf=Path(get("vcf")) if get("vcf") else None,
        dragen_dir=Path(get("dragen_dir")) if get("dragen_dir") else None,
        candidate_gene=get("candidate_gene"),
        candidate_variant=get("candidate_variant"),
        rsid=get("rsid"),
        disease=get("disease"),
        family_history=get("family_history"),
        sex=get("sex"),
        consanguinity=bool(get("consanguinity")),
        no_save=bool(get("no_save")),
        quiet=bool(get("quiet")),
    )


def _check_backend() -> None:
    import os
    from generesolver.integrations.medgemma_client import DEFAULT_MODEL
    model = os.environ.get("GENERESOLVER_MEDGEMMA_MODEL") or DEFAULT_MODEL
    print(f"LLM: local MedGemma ({model})")
    if not os.environ.get("GENERESOLVER_MEDGEMMA_BASE_URL"):
        print("  warning: GENERESOLVER_MEDGEMMA_BASE_URL is not set — point it at your local "
              "OpenAI-compatible MedGemma server, e.g. http://localhost:8000/v1.")


def _resolve_args() -> argparse.Namespace:
    if len(sys.argv) > 1:
        args = _parse_cli()
        if args.example:
            print(f"Running bundled example: {args.example!r}")
            return _config_to_namespace(EXAMPLES[args.example])
        return args

    if IDE_EXAMPLE:
        print(f"[IDE mode] No CLI arguments — running bundled example {IDE_EXAMPLE!r} "
              f"(change IDE_EXAMPLE in run_generesolver.py to switch).")
        _check_backend()
        return _config_to_namespace(EXAMPLES[IDE_EXAMPLE])

    print("[IDE mode] No CLI arguments — using CUSTOM_CASE from run_generesolver.py")
    _check_backend()
    return _config_to_namespace(CUSTOM_CASE)


def _load_note(args: argparse.Namespace) -> str | None:
    if args.note:
        return args.note
    if args.note_file:
        if not args.note_file.exists():
            print(f"ERROR: note_file does not exist: {args.note_file}",
                  file=sys.stderr)
            sys.exit(2)
        return args.note_file.read_text(encoding="utf-8").strip()
    return None


def _build_input(args: argparse.Namespace) -> CaseInput:
    case_id = args.case_id or f"CASE_{uuid.uuid4().hex[:10]}"
    note = _load_note(args)

    if not any([note, args.vcf, args.dragen_dir, args.candidate_gene,
                args.candidate_variant, args.rsid]):
        print(
            "ERROR: provide at least one input. CLI: --example / --note / --note-file / "
            "--vcf / --dragen-dir / --candidate-gene / --candidate-variant / "
            "--rsid. IDE: set IDE_EXAMPLE, or set IDE_EXAMPLE=None and fill in CUSTOM_CASE.",
            file=sys.stderr,
        )
        sys.exit(2)

    return CaseInput(
        case_id=case_id,
        clinical_note=note,
        vcf_path=str(args.vcf) if args.vcf else None,
        dragen_dir=str(args.dragen_dir) if args.dragen_dir else None,
        candidate_gene=args.candidate_gene,
        candidate_variant=args.candidate_variant,
        rsid=args.rsid,
        disease_label=args.disease,
        family_history=args.family_history,
        sex=args.sex,
        consanguinity=args.consanguinity if args.consanguinity else None,
    )


def _sep(title: str = "") -> None:
    if title:
        pad = (68 - len(title)) // 2
        print(f"\n{'─' * pad}  {title}  {'─' * pad}")
    else:
        print("\n" + "─" * 70)


def _print_summary(state, elapsed: float, save_dir: Path | None) -> None:
    print("\n" + "=" * 70)
    print(f"  GeneResolver  —  {state.case_id}")
    print("=" * 70)

    _sep("PIPELINE SUMMARY")
    print(f"  Elapsed          : {elapsed:.1f}s")
    print(f"  Task mode        : {state.task_mode}")
    print(f"  Steps completed  : {len(state.completed_steps)}")
    print(f"  Confidence level : {state.confidence_level}")
    if state.requires_human_review:
        print(f"  Human review     : REQUIRED")

    if state.safety_flags:
        _sep(f"SAFETY FLAGS  ({len(state.safety_flags)})")
        for f in state.safety_flags:
            print(f"  [SAFETY]  {f}")

    if state.warnings:
        _sep(f"WARNINGS  ({len(state.warnings)})")
        for w in state.warnings[:10]:
            print(f"  [!]  {w}")
        if len(state.warnings) > 10:
            print(f"  ... and {len(state.warnings) - 10} more")

    if state.hpo_profile.terms:
        _sep(f"HPO PROFILE  ({len(state.hpo_profile.terms)} terms)")
        for t in state.hpo_profile.terms[:10]:
            mark = "+" if t.status.value == "present" else "-"
            print(f"  [{mark}]  {t.hpo_id:<12}  {t.label}")
        if len(state.hpo_profile.terms) > 10:
            print(f"  ... and {len(state.hpo_profile.terms) - 10} more")

    if state.vcf_context:
        ctx = state.vcf_context
        _sep("VCF CONTEXT")
        print(f"  Build    : {ctx.genome_build}")
        print(f"  Variants : {ctx.variant_count} total | {ctx.pass_variant_count} PASS")
        print(f"  QC       : {ctx.qc_status}")

    if state.ranked_genes:
        _sep(f"RANKED GENES  ({len(state.ranked_genes)} candidates)")
        print(f"  {'#':<3} {'Gene':<10} {'Score':>6}  {'Conf':<10} Reason")
        for gr in state.ranked_genes[:10]:
            print(f"  {gr.rank:<3} {gr.gene:<10} {gr.final_score:>6.3f}  "
                  f"{gr.confidence:<10} {gr.prioritization_reason[:50]}")

    if state.acmg_assessments:
        _sep(f"ACMG ASSESSMENTS  ({len(state.acmg_assessments)})")
        for comp in state.acmg_assessments[:8]:
            a = comp.computed_acmg_assessment
            print(f"  {a.variant_id:<35} {a.gene or '?':<10} "
                  f"{a.classification} (disease={a.disease_id})")

    if state.doctor_report:
        dr = state.doctor_report
        _sep("DOCTOR REPORT")
        print(f"  Summary    : {dr.case_summary}")
        print(f"  Etiology   : {dr.classification}")
        if dr.classification_scores:
            print("  Class weights (after retry, if any):")
            for cls, score in sorted(dr.classification_scores.items(),
                                     key=lambda kv: -kv[1]):
                print(f"    {cls:<16} {score:.2f}")
        if dr.top_candidate_genes:
            print(f"  Top genes  : {', '.join(dr.top_candidate_genes)}")
        if dr.mechanism_explanation:
            print(f"  Mechanism  : {dr.mechanism_explanation[:200]}")
        if dr.uncertainty_statement:
            print(f"  Caveats    : {dr.uncertainty_statement[:200]}")
        if dr.suggested_review_points:
            print("  Review pts :")
            for rp in dr.suggested_review_points[:5]:
                print(f"    - {rp}")

    if save_dir and save_dir.exists():
        _sep("OUTPUT FILES")
        print(f"  Output dir : {save_dir}")
        for f in sorted(save_dir.glob("*")):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name:<40} ({size_kb:.1f} KB)")

    print("\n" + "=" * 70)
    print(
        "  Reminder: GeneResolver is decision support only — every output must be "
        "reviewed by a qualified clinician or clinical geneticist before any "
        "clinical action."
    )
    print("=" * 70)


def main() -> int:
    args = _resolve_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    inp = _build_input(args)
    print(f"Running GeneResolver on case {inp.case_id!r}...")

    t0 = time.time()
    try:
        state = run_case(inp, save_output=not args.no_save)
    except Exception as exc:
        print(f"\n[CRASH] Pipeline raised: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    elapsed = time.time() - t0

    save_dir: Path | None = None
    if not args.no_save:
        from generesolver.config import OUTPUTS_DIR
        save_dir = OUTPUTS_DIR / state.case_id

    _print_summary(state, elapsed, save_dir)
    if state.diagnostics.get("execution_failures"):
        print("Execution failed in: " + "; ".join(state.diagnostics["execution_failures"]),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
