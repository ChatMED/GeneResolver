from __future__ import annotations

from pathlib import Path

import pandas as pd

_PHEVAL_BASELINES = {
    "Exomiser":    {"Recall@1": 0.391, "Recall@3": 0.491, "Recall@5": 0.518, "Recall@10": 0.566, "MRR": 0.458},
    "PhenoGenius": {"Recall@1": 0.162, "Recall@3": 0.290, "Recall@5": 0.374, "Recall@10": 0.486, "MRR": 0.263},
    "Phen2Gene":   {"Recall@1": 0.096, "Recall@3": 0.214, "Recall@5": 0.258, "Recall@10": 0.331, "MRR": 0.180},
    "GADO":        {"Recall@1": 0.006, "Recall@3": 0.015, "Recall@5": 0.022, "Recall@10": 0.039, "MRR": 0.023},
}

_DEID_DISPLAY = {
    "deid_roberta_i2b2": "deid_roberta (i2b2-2014)",
    "stanford-deidentifier-base": "Stanford de-identifier",
    "OpenMed-privacy-filter": "Multilingual privacy filter",
}

_HPO_DISPLAY = {"FastHPOCR": "FastHPOCR (selected)", "ClinPhen": "ClinPhen"}
_HPO_ORDER = ["FastHPOCR", "ClinPhen"]

_CLASSIFIER_ROW_ORDER = [
    ("yes_raw", "Yes", "Raw"),
    ("yes_rag", "Yes", "RAG"),
    ("no_raw", "No", "Raw"),
    ("no_rag", "No", "RAG"),
]


def _fmt(x, nd: int = 3):
    return round(x, nd) if isinstance(x, (int, float)) else x



def _table_deid(stage1: dict) -> tuple[pd.DataFrame, str]:
    n = stage1.get("n")
    rows = []
    for model, m in (stage1.get("metrics") or {}).items():
        recall = m.get("recall")
        ci = m.get("recall_ci95") or [None, None]
        recall_ci = f"{recall:.3f} ({ci[0]:.3f}-{ci[1]:.3f})" if recall is not None and ci[0] is not None else (
            f"{recall:.3f}" if recall is not None else None)
        rows.append({
            "Model": _DEID_DISPLAY.get(model, model),
            "Recall (95% CI)": recall_ci,
            "Leaks": m.get("leaks"),
            "_sort": recall if recall is not None else -1,
        })
    rows.sort(key=lambda r: -r["_sort"])
    for r in rows:
        r.pop("_sort")
    df = pd.DataFrame(rows)
    caption = (
        f"Table 3. De-identification model comparison on a held-out set of {n} synthetic "
        "clinical notes. Recall is the fraction of PHI tokens removed (the safety-critical "
        "metric); leaks are tokens that survived masking." if n else
        "Table 3. De-identification model comparison (not run)."
    )
    return df, caption



def _table_hpo(stage2: dict) -> tuple[pd.DataFrame, str]:
    metrics = stage2.get("metrics") or {}
    order = _HPO_ORDER + [t for t in metrics if t not in _HPO_ORDER]
    rows = []
    for tool in order:
        m = metrics.get(tool)
        if not m:
            continue
        rows.append({
            "Method": _HPO_DISPLAY.get(tool, tool),
            "Precision": _fmt(m.get("precision")),
            "Recall": _fmt(m.get("recall")),
            "F1": _fmt(m.get("F1")),
        })
    df = pd.DataFrame(rows)
    n = stage2.get("n")
    caption = (
        f"Table 4. Phenotype-extraction methods on the GSC+ corpus ({n} documents), "
        "document-level micro-averaged." if n else
        "Table 4. Phenotype-extraction methods on the GSC+ corpus (not run)."
    )
    return df, caption



def _table_classifier(stage3: dict) -> tuple[pd.DataFrame, str]:
    metrics = stage3.get("metrics") or {}
    combos = metrics.get("combos") or {}
    ci_by_combo = (metrics.get("significance") or {}).get("ci_by_combo") or {}
    rows = []
    for key, fewshot_label, context_label in _CLASSIFIER_ROW_ORDER:
        m = combos.get(key)
        if not m:
            continue
        ci = ci_by_combo.get(key) or {}
        acc_ci = ci.get("accuracy_ci95")
        br_ci = ci.get("branch_separation_ci95")
        rows.append({
            "Few-shot": fewshot_label, "Context": context_label,
            "4-way acc.": _fmt(m.get("accuracy")),
            "4-way acc. 95% CI": f"{acc_ci[0]:.3f}-{acc_ci[1]:.3f}" if acc_ci else None,
            "Macro F1": _fmt(m.get("macro_f1")),
            "Branch sep.": _fmt(m.get("branch_separation")),
            "Branch sep. 95% CI": f"{br_ci[0]:.3f}-{br_ci[1]:.3f}" if br_ci else None,
        })
    df = pd.DataFrame(rows)
    n = stage3.get("n")
    caption = (
        f"Table 5. Etiology classification ablation across the {n} scored case reports "
        "(local MedGemma backend): few-shot guidance and RAG-enriched context, "
        "independently varied. Branch separation: accuracy of the single-gene-vs-rest "
        "routing decision. 95% CIs are percentile bootstrap (B=2,000); see the "
        "classifier_significance sheet for paired McNemar tests between configurations." if n else
        "Table 5. Etiology classification ablation (not run)."
    )
    return df, caption


def _table_classifier_significance(stage3: dict) -> tuple[pd.DataFrame, str]:
    pairwise = ((stage3.get("metrics") or {}).get("significance") or {}).get("pairwise") or []
    if not pairwise:
        return pd.DataFrame(), ""
    rows = []
    for p in pairwise:
        m4, mb = p["mcnemar_4way"], p["mcnemar_branch_separation"]
        rows.append({
            "Comparison": p["label"], "A": p["a"], "B": p["b"], "n paired": p["n_paired"],
            "4-way: b/c": f"{m4['b']}/{m4['c']}", "4-way p-value": m4["p_value"],
            "Branch sep.: b/c": f"{mb['b']}/{mb['c']}", "Branch sep. p-value": mb["p_value"],
        })
    df = pd.DataFrame(rows)
    caption = (
        "Paired McNemar tests between etiology-classifier configurations (same 92 cases). "
        "b = A correct & B incorrect; c = A incorrect & B correct."
    )
    return df, caption


def _table_classifier_per_class(stage3: dict) -> tuple[pd.DataFrame, str]:
    per_class = (stage3.get("metrics") or {}).get("per_class") or {}
    if not per_class:
        return pd.DataFrame(), ""
    rows = [
        {"Class": cls, "Precision": _fmt(v.get("precision")),
         "Recall": _fmt(v.get("recall")), "F1": _fmt(v.get("f1"))}
        for cls, v in per_class.items()
    ]
    return pd.DataFrame(rows), (
        "Per-class precision/recall/F1 for the selected production configuration "
        "(few-shot, raw narrative — Table 5's \"Yes / Raw\" row)."
    )


def _table_classifier_confusion(stage3: dict) -> tuple[pd.DataFrame, str]:
    combos = (stage3.get("metrics") or {}).get("combos") or {}
    m = combos.get("yes_raw") or {}
    conf = m.get("confusion") or {}
    if not conf:
        return pd.DataFrame(), ""
    labels = list(conf.keys())
    rows = []
    for gold in labels:
        row = {"gold \\ pred": gold}
        row.update({p: conf[gold].get(p, 0) for p in labels})
        row["<missing>"] = conf[gold].get("<missing>", 0)
        rows.append(row)
    return pd.DataFrame(rows), (
        "Confusion matrix for the selected production configuration "
        "(few-shot, raw narrative — Table 5's \"Yes / Raw\" row)."
    )



def _table_pheval(stage4: dict) -> tuple[pd.DataFrame, str]:
    summary = stage4.get("summary") or {}
    rows = []
    n_cases = None
    ci_note = ""
    if summary:
        n_cases = summary.get("total") or summary.get("number_of_samples")
        n = n_cases or 1
        rows.append({
            "System": "GeneResolver",
            "Recall@1": _fmt(summary.get("top1", 0) / n), "Recall@3": _fmt(summary.get("top3", 0) / n),
            "Recall@5": _fmt(summary.get("top5", 0) / n), "Recall@10": _fmt(summary.get("top10", 0) / n),
            "MRR": _fmt(summary.get("mrr", 0)),
        })
        ci = stage4.get("recall_at_1_ci95") or [None, None]
        if ci[0] is not None:
            n_ci = stage4.get("n_scored_for_ci", n_cases)
            ci_note = (f" GeneResolver recall@1 has a percentile-bootstrap 95% CI of "
                       f"{ci[0]:.3f}-{ci[1]:.3f} (B=2,000 resamples over the {n_ci} cases).")
    else:
        rows.append({"System": "GeneResolver (not run)", "Recall@1": None, "Recall@3": None,
                     "Recall@5": None, "Recall@10": None, "MRR": None})
    for name, vals in _PHEVAL_BASELINES.items():
        rows.append({"System": name, **vals})
    df = pd.DataFrame(rows)
    caption = (
        f"Table 6. Phenotype-only gene ranking on the PhEval corpus "
        f"({n_cases if n_cases else 'N'} cases). Baseline values are the paper's cited "
        "PhEval ranking statistics on the identical corpus; recall@k is the fraction of "
        "cases with the causal gene ranked within the top k." + ci_note
    )
    return df, caption



_VCF_ROW_ORDER = [
    ("Recall@1", "top1", True), ("Recall@3", "top3", True),
    ("Recall@5", "top5", True), ("Recall@10", "top10", True),
    ("MRR", "mrr", False), ("Found-rate", "found", True),
]


def _table_vcf(stage5: dict) -> tuple[pd.DataFrame, str]:
    ph = (stage5.get("phenotype_only") or {}).get("summary") or {}
    vcf = (stage5.get("phenotype_vcf") or {}).get("summary") or {}
    if not ph and not vcf:
        return pd.DataFrame(), "Table 7. Gene ranking on the LIRICAL spiked-exome corpus (not run)."

    n_ph = ph.get("total") or ph.get("number_of_samples") or 1
    n_vcf = vcf.get("total") or vcf.get("number_of_samples") or 1

    def _val(summary: dict, key: str, is_ratio: bool, n: int):
        v = summary.get(key)
        if v is None:
            return None
        return _fmt(v / n) if is_ratio else _fmt(v)

    rows = []
    for label, key, is_ratio in _VCF_ROW_ORDER:
        rows.append({
            "Metric": label,
            "Phenotype-only": _val(ph, key, is_ratio, n_ph),
            "Phenotype + VCF": _val(vcf, key, is_ratio, n_vcf),
        })
    df = pd.DataFrame(rows)

    mc = stage5.get("mcnemar_recall_at_1") or {}
    n_cases = ph.get("total") or ph.get("number_of_samples") or vcf.get("total") or vcf.get("number_of_samples")
    caption = (
        f"Table 7. Gene ranking on the LIRICAL spiked-exome corpus ({n_cases if n_cases else 'N'} "
        "cases): phenotype-only versus the full phenotype+VCF pipeline. Recall@k is the fraction "
        "of cases with the causal gene ranked within the top k."
    )
    if mc:
        caption += (
            f" Paired McNemar test on recall@1 (same {mc.get('n_paired_cases')} cases): the genome "
            f"corrects {mc.get('b')} cases and regresses {mc.get('c')}, p = {mc.get('p_value')}."
        )
    return df, caption


_SHEET_BUILDERS = {
    1: lambda r: {"table3_deid": _table_deid(r)},
    2: lambda r: {"table4_hpo_extraction": _table_hpo(r)},
    3: lambda r: {"table5_classifier": _table_classifier(r),
                  "classifier_significance": _table_classifier_significance(r),
                  "classifier_per_class": _table_classifier_per_class(r),
                  "classifier_confusion": _table_classifier_confusion(r)},
    4: lambda r: {"table6_pheval": _table_pheval(r)},
    5: lambda r: {"table7_vcf_ablation": _table_vcf(r)},
}


def build_workbook(results: dict[int, dict], out_path: Path) -> dict[str, tuple[pd.DataFrame, str]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheets: dict[str, tuple[pd.DataFrame, str]] = {}
    for stage, r in results.items():
        builder = _SHEET_BUILDERS.get(stage)
        if builder is None:
            continue
        for name, (df, caption) in builder(r).items():
            if not df.empty:
                sheets[name] = (df, caption)

    if not sheets:
        sheets["empty"] = (pd.DataFrame([{"note": "no stages produced results"}]), "")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, (df, _caption) in sheets.items():
            sheet_name = name[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)
            ws = writer.sheets[sheet_name]
            if _caption:
                ws.cell(row=1, column=1, value=_caption)

    print("\n--- Tables (paper format) ---")
    for name, (df, caption) in sheets.items():
        if caption:
            print(f"\n{caption}")
        print(df.to_string(index=False))

    return sheets
