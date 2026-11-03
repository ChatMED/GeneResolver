"""Build the manuscript's evaluation tables (Tables 4-10) from per-stage result files.

Every table is written separately (CSV + Markdown) and also collected into one Markdown file
and one Excel workbook. The builders only read the stage result dictionaries produced by
``reproduce_paper_results.py`` (``stage1.json`` ... ``stage6.json``), so tables can be rebuilt
from saved results without rerunning any model.

    Table 4  de-identification                    stage1.json
    Table 5  phenotype (HPO) extraction            stage2.json
    Table 6  etiology confusion matrix (No/Raw)    stage3.json
    Table 7  etiology few-shot x RAG ablation      stage3.json
    Table 8  ACMG/AMP validation vs GeneBe         stage6.json
    Table 9  phenotype-only ranking, PhEval        stage4.json
    Table 10 VCF ablation, LIRICAL spiked exomes   stage5.json
"""
from __future__ import annotations

import random
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd

from evaluation.table6_7_etiology_classification.eval_etiology_classification_score import (
    CONDITIONS as SCORER_CONDITIONS, LABELS, bootstrap_ci, confusion_matrix, exact_mcnemar_p, macro_f1,
)

BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20260731

# Published PhEval statistics for the comparison systems on the same 4,677-case corpus
# (Table 9). They were not rerun in this study.
PHEVAL_BASELINES: dict[str, dict[str, float]] = {
    "Exomiser": {
        "mrr": 0.458, "precision@1": 0.391, "precision@3": 0.164, "precision@5": 0.104, "precision@10": 0.057,
        "MAP@1": 0.391, "MAP@3": 0.436, "MAP@5": 0.442, "MAP@10": 0.448,
        "f_beta@1": 0.391, "f_beta@3": 0.231, "f_beta@5": 0.164, "f_beta@10": 0.099,
        "NDCG@3": 0.260, "NDCG@5": 0.249, "NDCG@10": 0.221},
    "PhenoGenius": {
        "mrr": 0.263, "precision@1": 0.162, "precision@3": 0.097, "precision@5": 0.075, "precision@10": 0.049,
        "MAP@1": 0.162, "MAP@3": 0.217, "MAP@5": 0.237, "MAP@10": 0.252,
        "f_beta@1": 0.162, "f_beta@3": 0.121, "f_beta@5": 0.102, "f_beta@10": 0.075,
        "NDCG@3": 0.136, "NDCG@5": 0.165, "NDCG@10": 0.184},
    "Phen2Gene": {
        "mrr": 0.180, "precision@1": 0.096, "precision@3": 0.071, "precision@5": 0.052, "precision@10": 0.033,
        "MAP@1": 0.096, "MAP@3": 0.150, "MAP@5": 0.160, "MAP@10": 0.169,
        "f_beta@1": 0.096, "f_beta@3": 0.082, "f_beta@5": 0.067, "f_beta@10": 0.049,
        "NDCG@3": 0.151, "NDCG@5": 0.163, "NDCG@10": 0.172},
    "GADO": {
        "mrr": 0.023, "precision@1": 0.006, "precision@3": 0.005, "precision@5": 0.004, "precision@10": 0.004,
        "MAP@1": 0.006, "MAP@3": 0.010, "MAP@5": 0.012, "MAP@10": 0.014,
        "f_beta@1": 0.006, "f_beta@3": 0.006, "f_beta@5": 0.005, "f_beta@10": 0.005,
        "NDCG@3": 0.015, "NDCG@5": 0.016, "NDCG@10": 0.022},
}

_RANKING_ROWS = (
    [("MRR", "mrr")]
    + [(f"Precision@{k}", f"precision@{k}") for k in (1, 3, 5, 10)]
    + [(f"MAP@{k}", f"MAP@{k}") for k in (1, 3, 5, 10)]
    + [(f"Fβ@{k}", f"f_beta@{k}") for k in (1, 3, 5, 10)]
    + [(f"NDCG@{k}", f"NDCG@{k}") for k in (3, 5, 10)]
)
_LIRICAL_ROWS = (
    [(f"Percentage@{k} (%)", f"percentage@{k}") for k in (1, 3, 5, 10)]
    + [r for r in _RANKING_ROWS if r[0] != "MRR"] + [("MRR", "mrr")]
)

_DEID_DISPLAY = {
    "OpenMed-privacy-filter": "OpenMed privacy filter",
    "stanford-deidentifier-base": "Stanford de-identifier",
    "deid_roberta_i2b2": "deid roberta (i2b2-2014)",
}
_HPO_DISPLAY = {"FastHPOCR": "FastHPOCR (selected)", "ClinPhen": "ClinPhen"}
_CLASS_DISPLAY = {"single_gene": "Single-gene", "chromosomal": "Chromosomal",
                  "multifactorial": "Multifactorial", "non_genetic": "Non-genetic"}
# (few_shot, rag) in the paper's row order; (False, False) is the production configuration.
_CONDITIONS = [(False, False), (True, False), (True, True), (False, True)]


def _f(x, nd: int) -> str:
    return "" if x is None else f"{x:.{nd}f}"


def _ci(ci, nd: int = 4) -> str:
    return f"({ci[0]:.{nd}f}-{ci[1]:.{nd}f})" if ci and ci[0] is not None else ""


# --------------------------------------------------------------------------- Table 4
def table4_deidentification(stage1: dict) -> tuple[pd.DataFrame, str]:
    rows = []
    total = None
    for key, m in (stage1.get("metrics") or {}).items():
        leaks, _, tokens = str(m.get("leaks", "")).partition("/")
        total = int(tokens) if tokens.isdigit() else total
        rows.append({"Model": _DEID_DISPLAY.get(key, key),
                     "Recall (95% CI)": f"{_f(m['recall'], 3)} {_ci(m.get('recall_ci95'))}".strip(),
                     "Leaks": f"{int(leaks):,}/{int(tokens):,}" if leaks.isdigit() and tokens.isdigit() else m.get("leaks"),
                     "_r": m["recall"]})
    rows.sort(key=lambda r: -r.pop("_r"))
    caption = (f"Table 4. De-identification model comparison on a held-out set of {stage1.get('n')} synthetic "
               f"clinical notes ({total:,} PHI tokens). Recall is the fraction of PHI tokens removed (the "
               "safety-critical metric); leaks are tokens that survived masking." if total else
               "Table 4. De-identification model comparison.")
    return pd.DataFrame(rows), caption


# --------------------------------------------------------------------------- Table 5
def table5_phenotype_extraction(stage2: dict) -> tuple[pd.DataFrame, str]:
    metrics = stage2.get("metrics") or {}
    order = [t for t in ("FastHPOCR", "ClinPhen") if t in metrics]
    rows = [{"Method": _HPO_DISPLAY.get(t, t), "Precision": _f(metrics[t]["precision"], 3),
             "Recall": _f(metrics[t]["recall"], 3),
             "F1 (95% CI)": f"{_f(metrics[t]['F1'], 3)} {_ci(metrics[t].get('F1_ci95'))}".strip()}
            for t in order]
    caption = (f"Table 5. Phenotype-extraction methods on the GSC+ corpus ({stage2.get('n')} documents), "
               "document-level micro-averaged.")
    return pd.DataFrame(rows), caption


# --------------------------------------------------------------------------- Stage 3 scoring
def score_etiology(rows: list[dict], bootstraps: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> dict:
    """Score every (few-shot, RAG) configuration over all of its runs.

    Only cases with a valid label in every run of every configuration are scored, so all
    rows of Table 7 are paired on the same cases. Each run gives accuracy, macro-F1 and
    branch separation; the table reports their mean and SD over runs. The 95% CI is a
    percentile bootstrap over cases of the per-case correctness averaged over runs (with one
    run this is the usual bootstrap CI). Table 6 uses the per-case majority label over runs
    (ties go to the earliest run).
    """
    by_cond: dict[tuple[bool, bool], dict[int, dict[str, dict]]] = {}
    for r in rows:
        by_cond.setdefault((bool(r["few_shot"]), bool(r["rag"])), {}).setdefault(int(r.get("repeat", 0)), {})[r["case_id"]] = r
    conditions = [c for c in _CONDITIONS if c in by_cond]
    case_sets = [set(run) for c in conditions for run in by_cond[c].values()]
    common = sorted(set.intersection(*case_sets)) if case_sets else []
    all_cases = sorted(set.union(*case_sets)) if case_sets else []
    if not common:
        raise ValueError("no case has a valid label in every run of every configuration")

    out: dict = {"n": len(common), "n_excluded": len(all_cases) - len(common),
                 "excluded_cases": sorted(set(all_cases) - set(common)), "conditions": {}}
    # One RNG, configurations in the Stage 3 scorer's order, accuracy CI before branch CI:
    # with a single run this reproduces the CIs of eval_etiology_classification_score exactly.
    rng = random.Random(seed)
    for cond in [c for c in SCORER_CONDITIONS if c in conditions]:
        runs = [by_cond[cond][k] for k in sorted(by_cond[cond])]
        gold = [runs[0][c]["gold_label"] for c in common]
        preds = [[run[c]["prediction"] for c in common] for run in runs]
        per_run = []
        for pred in preds:
            per_run.append({
                "accuracy": sum(g == p for g, p in zip(gold, pred)) / len(common),
                "macro_f1": macro_f1(gold, pred),
                "branch": sum((g == "single_gene") == (p == "single_gene") for g, p in zip(gold, pred)) / len(common),
            })
        acc_case = [statistics.mean(int(g == pred[i]) for pred in preds) for i, g in enumerate(gold)]
        br_case = [statistics.mean(int((g == "single_gene") == (pred[i] == "single_gene")) for pred in preds)
                   for i, g in enumerate(gold)]
        majority = [Counter(pred[i] for pred in preds).most_common(1)[0][0] for i in range(len(common))]
        summary = {k: {"mean": statistics.mean(r[k] for r in per_run),
                       "sd": statistics.stdev(r[k] for r in per_run) if len(per_run) > 1 else 0.0}
                   for k in ("accuracy", "macro_f1", "branch")}
        sg = [i for i, g in enumerate(gold) if g == "single_gene"]
        out["conditions"][f"{int(cond[0])}{int(cond[1])}"] = {
            "few_shot": cond[0], "rag": cond[1], "runs": len(runs), "per_run": per_run, **summary,
            "accuracy_ci95": list(bootstrap_ci(acc_case, rng, bootstraps)),
            "branch_ci95": list(bootstrap_ci(br_case, rng, bootstraps)),
            "label_changed_share": sum(len({pred[i] for pred in preds}) > 1 for i in range(len(common))) / len(common),
            "routing_changed_share": sum(len({pred[i] == "single_gene" for pred in preds}) > 1
                                         for i in range(len(common))) / len(common),
            "majority_confusion": confusion_matrix(gold, majority),
            "majority_correct": sum(g == p for g, p in zip(gold, majority)),
            "single_gene_correct": sum(majority[i] == "single_gene" for i in sg),
            "single_gene_total": len(sg),
            "_majority": dict(zip(common, majority)), "_gold": dict(zip(common, gold)),
        }
    # Table 6 describes the production configuration alone, so it uses every case that
    # configuration labelled in all of its runs (no pairing with the other configurations).
    if (False, False) in by_cond:
        runs = [by_cond[(False, False)][k] for k in sorted(by_cond[(False, False)])]
        cases = sorted(set.intersection(*[set(r) for r in runs]))
        gold = [runs[0][c]["gold_label"] for c in cases]
        majority = [Counter(r[c]["prediction"] for r in runs).most_common(1)[0][0] for c in cases]
        sg_tp = sum(g == p == "single_gene" for g, p in zip(gold, majority))
        sg_called = sum(p == "single_gene" for p in majority)
        out["production_all_cases"] = {
            "n": len(cases), "runs": len(runs),
            "excluded_cases": sorted(set(all_cases) - set(cases)),
            "majority_confusion": confusion_matrix(gold, majority),
            "majority_correct": sum(g == p for g, p in zip(gold, majority)),
            "single_gene_correct": sg_tp, "single_gene_total": gold.count("single_gene"),
            "single_gene_precision": sg_tp / sg_called if sg_called else None,
            "branch": sum((g == "single_gene") == (p == "single_gene") for g, p in zip(gold, majority)) / len(cases),
        }
    prod = out["conditions"].get("00")
    if prod:
        for key, c in out["conditions"].items():
            if key == "00":
                continue
            b = sum(prod["_majority"][i] == prod["_gold"][i] != c["_majority"][i] for i in common)
            cc = sum(c["_majority"][i] == c["_gold"][i] != prod["_majority"][i] for i in common)
            c["vs_production_mcnemar"] = {"production_only_correct": b, "this_only_correct": cc,
                                          "p_value": exact_mcnemar_p(b, cc)}
    for c in out["conditions"].values():
        c.pop("_majority"), c.pop("_gold")
    return out


# --------------------------------------------------------------------------- Table 6
def table6_etiology_confusion(stage3: dict) -> tuple[pd.DataFrame, str]:
    prod = stage3.get("production_all_cases") or (stage3.get("conditions") or {}).get("00")
    if not prod:
        return pd.DataFrame(), ""
    conf = prod["majority_confusion"]
    rows = [{"True class": _CLASS_DISPLAY[g], **{_CLASS_DISPLAY[p]: conf[g][p] for p in LABELS}} for g in LABELS]
    n, runs = prod.get("n", stage3["n"]), prod["runs"]
    basis = ("" if runs == 1 else f" Each case is assigned its majority label over the {runs} runs.")
    caption = (f"Table 6. Confusion matrix for raw-narrative etiology classification (production configuration) "
               f"on {n} cases.{basis} Rows show the true class and columns the predicted class. "
               f"{prod['majority_correct']} of {n} cases were classified correctly; single-gene sensitivity "
               f"{prod['single_gene_correct']}/{prod['single_gene_total']}")
    if prod.get("single_gene_precision") is not None:
        caption += (f", single-gene precision {prod['single_gene_precision']:.3f}, "
                    f"branch separation {100 * prod['branch']:.1f}%")
    caption += "."
    if prod.get("excluded_cases"):
        caption += (f" {len(prod['excluded_cases'])} case(s) without a valid label in every run were excluded: "
                    f"{', '.join(prod['excluded_cases'])}.")
    return pd.DataFrame(rows), caption


# --------------------------------------------------------------------------- Table 7
def table7_etiology_ablation(stage3: dict) -> tuple[pd.DataFrame, str]:
    conds = stage3.get("conditions") or {}
    rows = []
    for few, rag in _CONDITIONS:
        c = conds.get(f"{int(few)}{int(rag)}")
        if not c:
            continue
        multi = c["runs"] > 1

        def cell(key: str, pct: bool, ci_key: str | None = None) -> str:
            scale, nd = (100, 1) if pct else (1, 3)
            text = f"{c[key]['mean'] * scale:.{nd}f}{'%' if pct else ''}"
            if multi:
                text += f" ± {c[key]['sd'] * scale:.{nd}f}"
            if ci_key:
                lo, hi = c[ci_key]
                text += f" ({lo * scale:.{nd}f}-{hi * scale:.{nd}f}{'%' if pct else ''})"
            return text

        rows.append({"Few-shot": "Yes" if few else "No", "Context": "RAG" if rag else "Raw", "Runs": c["runs"],
                     "4-way acc. (95% CI)": cell("accuracy", True, "accuracy_ci95"),
                     "Macro F1": cell("macro_f1", False),
                     "Branch sep. (95% CI)": cell("branch", True, "branch_ci95")})
    runs = sorted({r["Runs"] for r in rows})
    stat = ("mean ± SD over runs; 95% CI is a percentile bootstrap over cases of per-case "
            "correctness averaged over runs" if runs != [1] else "single run; 95% percentile bootstrap CI")
    caption = (f"Table 7. Etiology classification ablation across {stage3.get('n')} dataset case narratives: "
               f"few-shot guidance and RAG-enriched context, independently varied ({stat}; "
               f"B={BOOTSTRAP_B:,}). Branch separation is the accuracy of the single-gene-versus-rest routing decision.")
    if stage3.get("n_excluded"):
        caption += (f" {stage3['n_excluded']} case(s) without a valid label in every run were excluded: "
                    f"{', '.join(stage3['excluded_cases'])}.")
    return pd.DataFrame(rows), caption


def table7_run_stability(stage3: dict) -> tuple[pd.DataFrame, str]:
    rows = []
    for few, rag in _CONDITIONS:
        c = (stage3.get("conditions") or {}).get(f"{int(few)}{int(rag)}")
        if not c or c["runs"] < 2:
            continue
        mc = c.get("vs_production_mcnemar") or {}
        rows.append({"Few-shot": "Yes" if few else "No", "Context": "RAG" if rag else "Raw", "Runs": c["runs"],
                     "Accuracy range": f"{min(r['accuracy'] for r in c['per_run']):.3f}-"
                                       f"{max(r['accuracy'] for r in c['per_run']):.3f}",
                     "Cases with a label change": f"{c['label_changed_share']:.1%}",
                     "Cases with a routing change": f"{c['routing_changed_share']:.1%}",
                     "McNemar vs No/Raw (b/c, p)": (f"{mc['production_only_correct']}/{mc['this_only_correct']}, "
                                                    f"p={mc['p_value']:.3f}") if mc else ""})
    return pd.DataFrame(rows), ("Supplementary to Table 7: run-to-run stability per configuration, and paired "
                                "McNemar tests of majority labels against the production configuration "
                                "(b = only No/Raw correct, c = only this configuration correct).")


# --------------------------------------------------------------------------- Table 8
def _pct_ci(x, ci) -> str:
    return f"{100 * x:.1f}% ({100 * ci[0]:.1f}–{100 * ci[1]:.1f})" if x is not None else "–"


def table8_acmg(stage6: dict) -> tuple[pd.DataFrame, str]:
    a, b = stage6.get("generesolver") or {}, stage6.get("genebe") or {}
    if not a:
        return pd.DataFrame(), ""
    rows = [
        ("P/LP classified P/LP (95% CI)", lambda r: _pct_ci(r["P_LP_called_P_LP"], r["P_LP_ci95"])),
        ("B/LB classified B/LB (95% CI)", lambda r: _pct_ci(r["B_LB_called_B_LB"], r["B_LB_ci95"])),
        ("Overall direction correct (95% CI)", lambda r: _pct_ci(r["overall_direction_correct"], r["overall_ci95"])),
        ("Opposite-direction calls", lambda r: str(r["opposite_direction"])),
        ("VUS / not classified", lambda r: str(r["vus_or_not_classified"])),
        ("Exact 5-class agreement", lambda r: f"{100 * r['exact_class_agreement']:.1f}%"),
    ]
    df = pd.DataFrame([{"Metric": m, "GeneResolver": f(a), "GeneBe": f(b) if b else "–"} for m, f in rows])
    g, mc = stage6.get("gold_counts") or {}, stage6.get("mcnemar_correct_direction") or {}
    caption = (f"Table 8. ACMG/AMP classification of {stage6.get('n')} held-out ClinVar variants "
               f"({g.get('pathogenic')} P, {g.get('likely_pathogenic')} LP, {g.get('likely_benign')} LB, "
               f"{g.get('benign')} B; ≥2 stars, GRCh37) by GeneResolver and GeneBe, as returned by each tool "
               f"(all criteria). Wilson 95% CIs. Direction agreement between the tools "
               f"{100 * stage6.get('agreement_direction', 0):.1f}% (κ {stage6.get('kappa_direction')}); variants "
               f"correct only by GeneResolver {mc.get('only_generesolver')}, only by GeneBe {mc.get('only_genebe')}.")
    return df, caption


# --------------------------------------------------------------------------- Table 9
def table9_pheval(stage4: dict) -> tuple[pd.DataFrame, str]:
    s = stage4.get("summary") or {}
    if not s:
        return pd.DataFrame(), ""
    rows = [{"Metric": label, "GeneResolver": _f(s.get(key), 4),
             **{name: _f(vals[key], 3) for name, vals in PHEVAL_BASELINES.items()}}
            for label, key in _RANKING_ROWS]
    n = s.get("number_of_samples") or s.get("total")
    return pd.DataFrame(rows), (f"Table 9. Phenotype-only ranking and binary-classification metrics on the PhEval "
                                f"corpus ({n} cases). Baselines are the published PhEval statistics and were not rerun.")


# --------------------------------------------------------------------------- Table 10
def table10_vcf_ablation(stage5: dict) -> tuple[pd.DataFrame, str]:
    ph = (stage5.get("phenotype_only") or {}).get("summary") or {}
    vcf = (stage5.get("phenotype_vcf") or {}).get("summary") or {}
    if not ph and not vcf:
        return pd.DataFrame(), ""
    rows = [{"Metric": label, "Phenotype-only": _f(ph.get(key), 2 if "%" in label else 4),
             "Phenotype + VCF": _f(vcf.get(key), 2 if "%" in label else 4)} for label, key in _LIRICAL_ROWS]
    n = ph.get("number_of_samples") or vcf.get("number_of_samples")
    return pd.DataFrame(rows), f"Table 10. VCF ablation: gene ranking on the LIRICAL spiked-exome benchmark ({n} cases), phenotype only vs phenotype + VCF."


TABLES = {
    "table4_deidentification": (1, table4_deidentification),
    "table5_phenotype_extraction": (2, table5_phenotype_extraction),
    "table6_etiology_confusion": (3, table6_etiology_confusion),
    "table7_etiology_ablation": (3, table7_etiology_ablation),
    "table7_supp_run_stability": (3, table7_run_stability),
    "table8_acmg_validation": (6, table8_acmg),
    "table9_pheval_phenotype_only": (4, table9_pheval),
    "table10_vcf_ablation": (5, table10_vcf_ablation),
}


def build_tables(results: dict[int, dict], out_dir: Path) -> dict[str, tuple[pd.DataFrame, str]]:
    """Write each table as CSV + Markdown, plus paper_tables.md and paper_tables.xlsx."""
    out_dir.mkdir(parents=True, exist_ok=True)
    built: dict[str, tuple[pd.DataFrame, str]] = {}
    for name, (stage, builder) in TABLES.items():
        if not results.get(stage):
            continue
        df, caption = builder(results[stage])
        if df.empty:
            continue
        built[name] = (df, caption)
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
        (out_dir / f"{name}.md").write_text(f"{caption}\n\n{df.to_markdown(index=False)}\n", encoding="utf-8")

    (out_dir / "paper_tables.md").write_text(
        "# GeneResolver evaluation tables\n\n"
        + "\n".join(f"## {caption.split('.')[0]}\n\n{caption}\n\n{df.to_markdown(index=False)}\n"
                    for df, caption in built.values()),
        encoding="utf-8")
    if built:
        with pd.ExcelWriter(out_dir / "paper_tables.xlsx", engine="openpyxl") as writer:
            for name, (df, caption) in built.items():
                df.to_excel(writer, sheet_name=name[:31], index=False, startrow=1)
                writer.sheets[name[:31]].cell(row=1, column=1, value=caption)
    for df, caption in built.values():
        print(f"\n{caption}\n{df.to_string(index=False)}")
    return built
