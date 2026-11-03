from __future__ import annotations

import logging
import os

from generesolver.integrations.llm_client import complete_json, active_backend
from generesolver.schemas.case import CaseState, TaskMode
from generesolver.schemas.report import DoctorReport, GeneticistAppendix, ProvenanceItem

logger = logging.getLogger(__name__)


def run(state: CaseState) -> CaseState:
    if state.task_mode == TaskMode.LOCUS_LOOKUP:
        state.doctor_report = _locus_report(state)
        state.geneticist_appendix = _build_geneticist_appendix(state)
        state.provenance.append(ProvenanceItem(
            component="report_generator", tool_name="deterministic ClinVar record summary"
        ))
        state.mark_done("report_generator")
        return state

    generated = False
    try:
        state.doctor_report = _generate_doctor_report(state)
        generated = True
    except Exception as exc:
        logger.error("Doctor report generation failed: %s", exc)
        state.add_warning(f"Doctor report model generation failed; deterministic fallback used: {exc}")
        state.diagnostics.setdefault("execution_failures", []).append(
            f"report_generator model generation: {exc}"
        )
        state.doctor_report = _fallback_doctor_report(state)

    state.geneticist_appendix = _build_geneticist_appendix(state)

    backend = active_backend()
    from generesolver.integrations import medgemma_client
    model = os.environ.get("GENERESOLVER_MEDGEMMA_MODEL") or medgemma_client.DEFAULT_MODEL
    state.provenance.append(
        ProvenanceItem(
            component="report_generator",
            tool_name=(f"{backend} ({model})" if generated else "deterministic fallback report"),
            tool_version=(f"generation={'model' if generated else 'fallback'}; "
                          f"backend={backend}; model={model}; "
                          f"temperature=0; max_tokens=1024; "
                          f"cache_disabled={os.environ.get('GENERESOLVER_LLM_CACHE_DISABLE', '0')}; "
                          f"local_annotation={os.environ.get('GENERESOLVER_LOCAL_MODE', '0')}"),
        )
    )
    state.mark_done("report_generator")
    return state


def _locus_report(state: CaseState) -> DoctorReport:
    lookup = state.branch_result.get("locus_lookup") or {}
    rsid = lookup.get("rsid") or state.input.rsid or "the requested rsID"
    records = lookup.get("clinvar_records") or []
    genes = list(dict.fromkeys(
        gene for record in records for gene in record.get("gene_symbols", []) if gene
    ))
    if records:
        descriptions = [
            f"{record.get('title') or record.get('clinvar_id')}: "
            f"{record.get('clinical_significance') or 'significance unavailable'}"
            for record in records[:5]
        ]
        summary = f"ClinVar returned {len(records)} record(s) for {rsid}. " + "; ".join(descriptions)
    else:
        summary = f"No ClinVar record was retrieved for {rsid}."
    report = DoctorReport(
        case_id=state.case_id,
        case_summary=summary,
        top_candidate_genes=genes,
        top_candidate_variants=[rsid] if records else [],
        evidence_strength="ClinVar record lookup only; no patient-level evidence",
        mechanism_explanation="",
        uncertainty_statement=(
            "An rsID identifies a locus or allele in a database, not the patient's genotype. "
            "This lookup does not establish that the patient carries the variant or has a diagnosis."
        ),
        suggested_review_points=[
            "Confirm the exact allele, genome build, and patient genotype before clinical interpretation."
        ],
    )
    _attach_case_record(state, report)
    report.classification = "Not assessed (locus lookup only)"
    return report


def _phenotype_tie_note(state: CaseState) -> str:
    ranked = getattr(state, "ranked_diseases", None) or []
    if not ranked or not isinstance(ranked[0], dict):
        return ""
    top = ranked[0]
    ties = top.get("phenotype_tie_ids") or []
    if not ties:
        return ""
    pairs = top.get("candidate_disease_gene_pairs") or []
    names = [p.get("name") for p in pairs if p.get("name")][:4]
    spectrum = top.get("name", "the leading disease")
    listed = "; ".join(names) if names else f"{len(ties) + 1} subtypes"
    return (
        f"The phenotype supports the {spectrum} spectrum, but the available HPO "
        f"evidence does not clearly separate the candidate subtypes ({listed}). "
        "See the suggested confirmatory tests for what would resolve this."
    )


def _generate_doctor_report(state: CaseState) -> DoctorReport:
    prompt = _build_prompt(state)
    data = complete_json(system="", user=prompt, max_tokens=1024)
    required = {"case_summary": str, "top_candidate_genes": list,
                "top_candidate_variants": list, "evidence_strength": str,
                "mechanism_explanation": str, "uncertainty_statement": str,
                "suggested_review_points": list}
    if not isinstance(data, dict) or any(not isinstance(data.get(k), t)
                                         for k, t in required.items()):
        raise ValueError("Report model JSON did not match the required report fields")

    known_genes = {g.gene for g in state.ranked_genes}
    genes = [g for g in data["top_candidate_genes"] if isinstance(g, str) and g in known_genes]
    d = getattr(state, "differential_decision", None)
    if d and d.most_likely_gene:
        ordered = [d.most_likely_gene] + [g for g in genes if g != d.most_likely_gene]
        genes = [g for g in ordered if g in known_genes]
    known_variants = {c.computed_acmg_assessment.variant_id for c in state.acmg_assessments}
    data["top_candidate_variants"] = [v for v in data["top_candidate_variants"]
                                      if isinstance(v, str) and v in known_variants]
    review = list(data.get("suggested_review_points", []) or [])
    if d and d.decisive_tests:
        for t in d.decisive_tests:
            if t and t not in review:
                review.append(t)

    tie_note = _phenotype_tie_note(state)
    if tie_note:
        unc = (data.get("uncertainty_statement") or "").strip()
        data["uncertainty_statement"] = (unc + " " + tie_note).strip()

    mechanism = (state.branch_result.get("mechanism") or "").strip() \
        or data.get("mechanism_explanation", "")

    report = DoctorReport(
        case_id=state.case_id,
        case_summary=data.get("case_summary", ""),
        top_candidate_genes=genes,
        top_candidate_variants=data.get("top_candidate_variants", []),
        evidence_strength=data.get("evidence_strength", state.confidence_level),
        mechanism_explanation=mechanism,
        uncertainty_statement=data.get("uncertainty_statement", ""),
        suggested_review_points=review,
    )
    _attach_case_record(state, report)
    return report


def _possible_conditions(state: CaseState) -> list[dict]:
    if state.branch_result.get("conditions"):
        return [
            {
                "name": c.get("name", ""),
                "likelihood": c.get("likelihood", ""),
                "supporting": c.get("supporting", []),
                "against": c.get("against", []),
            }
            for c in state.branch_result["conditions"][:8]
            if isinstance(c, dict)
        ]

    out: list[dict] = []
    for d in (state.ranked_diseases or [])[:8]:
        if isinstance(d, dict):
            out.append({
                "name": d.get("name", d.get("disease_id", "")),
                "disease_id": d.get("disease_id", ""),
                "likelihood": d.get("posttest_prob", ""),
                "genes": d.get("genes", []),
            })
    return out


def _attach_case_record(state: CaseState, report: DoctorReport) -> None:
    cc = state.case_class
    if cc:
        report.classification = cc.decision.value if cc.decision else "uncertain"
        report.classification_reasoning = cc.conclusion or cc.tiebreak_reasoning
        report.classification_scores = {a.case_class.value: a.score for a in cc.advocates}

    report.possible_conditions = _possible_conditions(state)

    report.suggested_next_steps = [
        {
            "test": t.test,
            "rationale": t.rationale,
            "confirms": t.confirms,
            "refutes": t.refutes,
            "power": t.power.value,
        }
        for t in state.next_steps.tests
    ]
    report.next_steps_reasoning = state.next_steps.reasoning

    report.retrieval_trace = [
        {
            "source": r.source,
            "query": r.query,
            "n_results": r.n_results,
            "results": r.results,
            "failed": r.failed,
            "error": r.error,
        }
        for r in state.trace.retrievals
    ]
    report.reasoning_trace = [
        {"step": s.step, "claim": s.claim, "grounded_in": s.grounded_in,
         "confidence": s.confidence}
        for s in state.trace.steps
    ]
    report.sources_consulted = state.trace.sources_used()
    report.investigation_limitations = list(state.branch_result.get("limitations") or [])


def _verdict_block(state: CaseState) -> str:
    d = getattr(state, "differential_decision", None)
    if not d or not d.most_likely_gene:
        return ""
    conf = getattr(d.confidence_band, "value", d.confidence_band)
    lines = [
        f"  PRIMARY: {d.most_likely_gene}"
        + (f" — {d.most_likely_disease_label}" if d.most_likely_disease_label else "")
        + (f" ({d.most_likely_orpha})" if d.most_likely_orpha else "")
        + f"  [confidence: {conf}]",
        f"  Reasoning: {d.primary_reasoning}" if d.primary_reasoning else "",
    ]
    for a in (d.alternatives or [])[:3]:
        lines.append(
            f"  ALTERNATIVE: {a.gene}"
            + (f" — {a.disease_label}" if a.disease_label else "")
            + (f": {a.brief_reasoning}" if a.brief_reasoning else "")
        )
    if d.decisive_tests:
        lines.append("  Decisive tests: " + "; ".join(d.decisive_tests[:5]))
    return "\n".join(l for l in lines if l)


def _build_prompt(state: CaseState) -> str:
    hpo_lines = [
        f"  • {t.hpo_id} ({t.label}) — {'PRESENT' if t.status.value == 'present' else 'ABSENT'}"
        for t in state.hpo_profile.terms[:15]
    ]
    phenotype_block = "\n".join(hpo_lines) or "  No HPO terms mapped."

    gene_lines = []
    for gr in state.ranked_genes[:5]:
        bio = next((b for b in state.biological_evidence if b.gene == gr.gene), None)
        mechanism = bio.mechanism_summary_doctor if bio else ""
        gene_lines.append(
            f"  • {gr.gene} (score={gr.final_score:.2f}, confidence={gr.confidence}): "
            f"{gr.prioritization_reason}. {mechanism}"
        )
    gene_block = "\n".join(gene_lines) or "  No candidate genes identified."

    variant_lines = []
    for comp in state.acmg_assessments[:5]:
        a = comp.computed_acmg_assessment
        met = a.pathogenic_criteria + a.benign_criteria
        variant_lines.append(
            f"  • {a.variant_id} in {a.gene} — {a.classification} [{', '.join(met)}]"
        )
    variant_block = "\n".join(variant_lines) or "  No variants assessed."

    safety_block = (
        "\n".join(f"  ⚠ {f}" for f in state.safety_flags)
        or "  No safety flags."
    )

    verdict_block = _verdict_block(state)
    verdict_section = (
        f"Differential reasoner's verdict (LEAD THE SUMMARY WITH THIS — it is the "
        f"holistic, evidence-grounded clinical conclusion; the candidate-gene list "
        f"below is supporting evidence):\n{verdict_block}\n\n"
        if verdict_block else ""
    )

    return (
        "You are a clinical genomics expert preparing a report for a physician "
        "who is NOT a genetic specialist.\n\n"
        f"Case ID: {state.case_id}\n"
        f"Overall evidence confidence: {state.confidence_level}\n\n"
        f"Patient phenotype (HPO terms):\n{phenotype_block}\n\n"
        f"{verdict_section}"
        f"Top candidate genes (supporting evidence):\n{gene_block}\n\n"
        f"Variant assessments:\n{variant_block}\n\n"
        f"Safety flags:\n{safety_block}\n\n"
        "Write a concise clinical decision-support summary. Center it on the "
        "reasoner's verdict above (most likely gene + reasoning, then alternatives); "
        "use the candidate-gene list as supporting detail, not as the headline. "
        "Do NOT make a definitive diagnosis. Respond with JSON only (no markdown fences):\n"
        '{"case_summary": "<2-3 sentences>", '
        '"top_candidate_genes": ["GENE1", ...], '
        '"top_candidate_variants": ["variant_id", ...], '
        '"evidence_strength": "<brief quality statement>", '
        '"mechanism_explanation": "<2-3 plain-English sentences>", '
        '"uncertainty_statement": "<clear limitations statement>", '
        '"suggested_review_points": ["<point 1>", "<point 2>", ...]}'
    )


def _fallback_doctor_report(state: CaseState) -> DoctorReport:
    d = getattr(state, "differential_decision", None)
    top_genes = [gr.gene for gr in state.ranked_genes[:3]]
    if d and d.most_likely_gene:
        top_genes = [d.most_likely_gene] + [g for g in top_genes if g != d.most_likely_gene]
    top_variants = [c.computed_acmg_assessment.variant_id for c in state.acmg_assessments[:3]]

    mechanism_parts = []
    for gr in state.ranked_genes[:3]:
        bio = next((b for b in state.biological_evidence if b.gene == gr.gene), None)
        if bio and bio.mechanism_summary_doctor:
            mechanism_parts.append(f"{gr.gene}: {bio.mechanism_summary_doctor}")
    mechanism_explanation = "  ".join(mechanism_parts)

    report = DoctorReport(
        case_id=state.case_id,
        case_summary=(
            (f"Available analysis identified {len(state.ranked_genes)} candidate gene(s). "
             if state.ranked_genes else "No candidate genes are available from completed steps. ")
            + f"Confidence: {state.confidence_level}. Expert review is required."
        ),
        top_candidate_genes=top_genes,
        top_candidate_variants=top_variants,
        evidence_strength=state.confidence_level,
        mechanism_explanation=(
            (state.branch_result.get("mechanism") or "").strip() or mechanism_explanation
        ),
        uncertainty_statement=(
            "Report generation encountered an error. "
            "All findings must be reviewed by a clinical geneticist."
        ),
        suggested_review_points=state.human_review_reasons[:5],
    )
    _attach_case_record(state, report)
    return report


def _build_geneticist_appendix(state: CaseState) -> GeneticistAppendix:
    hpo_terms = [
        f"{t.hpo_id} ({t.label}) [{t.status.value}]"
        for t in state.hpo_profile.terms
    ]

    vcf_summary = ""
    if state.vcf_context:
        vcf_summary = (
            f"Build: {state.vcf_context.genome_build or 'unknown'}, "
            f"Total variants: {state.vcf_context.variant_count}, "
            f"PASS: {state.vcf_context.pass_variant_count}, "
            f"QC: {state.vcf_context.qc_status.value}"
        )

    candidate_variants = [
        {
            "variant_id": v.variant_id,
            "gene": v.gene,
            "consequence": v.consequence,
            "impact": v.impact,
            "gnomad_af": v.gnomad_af,
            "clinvar": v.clinvar_significance,
            "hgvs_p": v.hgvs_p,
        }
        for v in state.annotated_variants[:20]
    ]

    acmg_evidence = [
        {
            "variant_id": comp.computed_acmg_assessment.variant_id,
            "gene": comp.computed_acmg_assessment.gene,
            "disease_id": comp.computed_acmg_assessment.disease_id,
            "classification": comp.computed_acmg_assessment.classification,
            "classification_status": comp.computed_acmg_assessment.classification_status,
            "pathogenic_criteria": comp.computed_acmg_assessment.pathogenic_criteria,
            "benign_criteria": comp.computed_acmg_assessment.benign_criteria,
            "criteria_not_evaluated": comp.computed_acmg_assessment.criteria_not_evaluated,
            "clinvar_significance": (
                comp.clinvar_external_annotation.significance
                if comp.clinvar_external_annotation else None
            ),
            "clinvar_concordance": comp.concordance,
        }
        for comp in state.acmg_assessments
    ]

    inheritance_assessment = [
        {
            "gene": ic.gene,
            "suspected_model": ic.suspected_model.value,
            "compatible": ic.compatible,
            "limitations": ic.limitations,
        }
        for ic in state.inheritance_results
    ]

    gwas_evidence = [
        {
            "query": e.query,
            "trait": e.trait,
            "p_value": e.p_value,
            "source_db": e.source_db,
            "warning": e.warning,
        }
        for e in state.gwas_evidence
    ]

    gene_disease_evidence = [
        {
            "gene": be.gene,
            "biological_support": be.biological_support,
            "disease_associations": be.disease_associations,
            "pathways": be.pathways[:5],
            "uniprot_id": be.uniprot_id,
        }
        for be in state.biological_evidence
    ]

    literature_evidence = [
        {
            "gene": le.gene,
            "pmid_count": len(le.pmids),
            "support_level": le.support_level,
            "pmids": le.pmids,
        }
        for le in state.literature_evidence
    ]

    mechanism_evidence = [
        {
            "gene": be.gene,
            "doctor_summary": be.mechanism_summary_doctor,
            "expert_summary": be.mechanism_summary_expert,
        }
        for be in state.biological_evidence
        if be.mechanism_summary_doctor or be.mechanism_summary_expert
    ]

    limitations = [
        "ACMG classifications are computationally-derived and require expert validation.",
        "GWAS associations are population-level and do not establish Mendelian causality.",
        "VEP consequence predictions may not account for all splice or regulatory effects.",
        "HPO mapping was performed using NLP and LLM-assisted selection — review for accuracy.",
        "This analysis is decision support only and must not replace clinical judgement.",
    ]
    if not state.has_vcf:
        limitations.append("No VCF provided — variant-level evidence is absent.")
    if not state.hpo_profile.present_ids:
        limitations.append("No HPO terms mapped — phenotype similarity scoring was not possible.")

    return GeneticistAppendix(
        case_id=state.case_id,
        hpo_terms_used=hpo_terms,
        vcf_context_summary=vcf_summary,
        candidate_variants=candidate_variants,
        acmg_evidence=acmg_evidence,
        inheritance_assessment=inheritance_assessment,
        gwas_evidence=gwas_evidence,
        gene_disease_evidence=gene_disease_evidence,
        literature_evidence=literature_evidence,
        mechanism_evidence=mechanism_evidence,
        tool_versions=list(state.provenance),
        limitations=limitations,
        warnings=list(state.warnings),
    )
