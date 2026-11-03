from __future__ import annotations

import functools
import importlib
import logging

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph

from generesolver.schemas.case import CaseState, TaskMode
from generesolver.schemas.classification import CaseClass

logger = logging.getLogger(__name__)

_RECURSION_LIMIT = 100


STEP_MODULES: dict[str, str] = {
    "demographics":      "generesolver.components.demographics",
    "biochem_extractor": "generesolver.components.biochem_extractor",
    "deidentifier":      "generesolver.components.deidentifier",
    "disease_extractor": "generesolver.components.disease_extractor",
    "hpo_mapper":        "generesolver.components.hpo_mapper",
    "clinical_synthesizer": "generesolver.agents.clinical_synthesizer",
    "vcf_input":         "generesolver.components.vcf_input",
    "variant_annotator": "generesolver.components.variant_annotator",
    "gwas_lookup":        "generesolver.components.gwas_lookup",
    "locus_lookup":       "generesolver.components.locus_lookup",
    "inheritance":        "generesolver.components.inheritance",
    "case_classifier":    "generesolver.agents.case_classifier",
    "class_retrieval":    "generesolver.components.class_retrieval",
    "case_reclassify":    "generesolver.agents.case_classifier",
    "branch_chromosomal":    "generesolver.agents.branch_chromosomal",
    "branch_multifactorial": "generesolver.agents.branch_multifactorial",
    "branch_phenocopy":      "generesolver.agents.branch_phenocopy",
    "branch_uncertain":      "generesolver.agents.branch_uncertain",
    "gene_aggregator":  "generesolver.components.gene_aggregator",
    "gene_ranker":       "generesolver.components.gene_ranker",
    "variant_disease_context": "generesolver.components.variant_disease_context_builder",
    "acmg_engine":       "generesolver.components.acmg_engine_step",
    "acmg_rerank":       "generesolver.components.acmg_rerank",
    "differential_synthesizer": "generesolver.agents.differential_synthesizer",
    "reasoner":          "generesolver.agents.reasoner",
    "case_integrator":   "generesolver.components.case_integrator",
    "bio_integrator":    "generesolver.components.bio_integrator",
    "literature":        "generesolver.components.literature",
    "mechanism":         "generesolver.agents.mechanism",
    "next_steps":        "generesolver.agents.next_steps",
    "safety_monitor":    "generesolver.components.safety_monitor",
    "review_manager":    "generesolver.components.review_manager",
    "report_generator":  "generesolver.agents.report_generator",
    "exporter":           "generesolver.components.exporter",
}


_CANONICAL_ORDER: list[str] = [
    "demographics", "deidentifier", "biochem_extractor", "disease_extractor",
    "hpo_mapper", "clinical_synthesizer", "vcf_input", "variant_annotator",
    "inheritance", "case_classifier", "gene_aggregator",
    "bio_integrator", "gwas_lookup", "locus_lookup", "literature", "gene_ranker",
    "variant_disease_context", "acmg_engine", "acmg_rerank", "mechanism",
    "differential_synthesizer", "reasoner", "case_integrator", "next_steps",
    "safety_monitor", "review_manager", "report_generator", "exporter",
]


_STEPS_PHENOTYPE_ONLY = [
    "demographics", "deidentifier", "biochem_extractor", "disease_extractor",
    "hpo_mapper", "clinical_synthesizer", "case_classifier", "gene_aggregator",
    "bio_integrator", "gwas_lookup", "literature", "gene_ranker", "mechanism",
    "differential_synthesizer", "reasoner", "case_integrator", "next_steps",
    "safety_monitor", "review_manager", "report_generator", "exporter",
]

_STEPS_PHENOTYPE_VCF = [step for step in _CANONICAL_ORDER if step != "locus_lookup"]

_STEPS_VARIANT_INTERPRETATION = [
    "vcf_input", "variant_annotator", "inheritance",
    "case_classifier", "gene_aggregator", "bio_integrator", "gwas_lookup",
    "literature", "gene_ranker", "variant_disease_context",
    "acmg_engine", "acmg_rerank", "mechanism",
    "differential_synthesizer",
    "reasoner", "case_integrator", "next_steps", "safety_monitor",
    "review_manager", "report_generator", "exporter",
]

_STEPS_GENE_DISEASE_QUERY = [
    "bio_integrator", "gwas_lookup", "literature", "mechanism", "next_steps",
    "safety_monitor", "review_manager", "report_generator", "exporter",
]

_STEPS_LOCUS_LOOKUP = ["locus_lookup", "report_generator", "exporter"]

_MODE_STEPS: dict[TaskMode, list[str]] = {
    TaskMode.PHENOTYPE_ONLY: _STEPS_PHENOTYPE_ONLY,
    TaskMode.PHENOTYPE_VCF: _STEPS_PHENOTYPE_VCF,
    TaskMode.VARIANT_INTERPRETATION: _STEPS_VARIANT_INTERPRETATION,
    TaskMode.GENE_DISEASE_QUERY: _STEPS_GENE_DISEASE_QUERY,
    TaskMode.LOCUS_LOOKUP: _STEPS_LOCUS_LOOKUP,
}


_GENE_RANKING_STEPS = [
    "gene_aggregator", "bio_integrator", "gwas_lookup", "literature",
    "gene_ranker", "variant_disease_context", "acmg_engine", "acmg_rerank",
    "mechanism", "differential_synthesizer", "reasoner", "case_integrator",
]


_BRANCH_NODE: dict[CaseClass | None, str] = {
    CaseClass.CHROMOSOMAL: "branch_chromosomal",
    CaseClass.MULTIFACTORIAL: "branch_multifactorial",
    CaseClass.PHENOCOPY: "branch_phenocopy",
    None: "branch_uncertain",
}


def _should_skip(step: str, state: CaseState) -> bool:

    if step in {"demographics", "biochem_extractor"} and not state.has_clinical_note:
        return True
    if step == "disease_extractor" \
            and not state.has_clinical_note \
            and not state.input.disease_label:
        return True
    if step == "vcf_input" and not state.has_vcf:
        return True
    if step == "variant_annotator" and not state.vcf_context:
        return True
    if step == "gene_aggregator" \
            and not state.vcf_context \
            and not state.hpo_profile.present_ids \
            and not state.input.candidate_gene:
        return True
    if step == "differential_synthesizer" and not state.ranked_genes:
        return True
    if step == "variant_disease_context" \
            and (not state.annotated_variants or not state.ranked_genes):
        return True
    if step == "acmg_engine" and not state.variant_disease_contexts:
        return True
    if step == "acmg_rerank" and not state.acmg_assessments:
        return True
    if step == "clinical_synthesizer" and not state.has_clinical_note:
        return True
    if step == "gwas_lookup" and not state.has_rsid \
            and not state.annotated_variants:
        return True
    if step == "locus_lookup" and not state.has_rsid:
        return True
    if step == "inheritance" and not state.has_family_history \
            and not state.annotated_variants:
        return True
    return False


def _detect_mode(state: CaseState) -> TaskMode:
    inp = state.input
    if inp.rsid and not inp.clinical_note and not inp.vcf_path:
        return TaskMode.LOCUS_LOOKUP
    if inp.candidate_gene and not inp.vcf_path:
        return TaskMode.GENE_DISEASE_QUERY
    _has_phenotype = bool(inp.clinical_note or inp.hpo_terms)
    if (inp.vcf_path or inp.dragen_dir or inp.candidate_variant) and not _has_phenotype:
        return TaskMode.VARIANT_INTERPRETATION
    if _has_phenotype and (inp.vcf_path or inp.dragen_dir):
        return TaskMode.PHENOTYPE_VCF
    return TaskMode.PHENOTYPE_ONLY


def _route(state: CaseState) -> CaseState:
    inp = state.input

    state.has_clinical_note = bool(inp.clinical_note)
    state.has_vcf = bool(inp.vcf_path or inp.dragen_dir)
    state.has_candidate_variant = bool(inp.candidate_variant)
    state.has_rsid = bool(inp.rsid)
    state.has_family_history = bool(inp.family_history)

    state.task_mode = _detect_mode(state)
    steps = list(_MODE_STEPS[state.task_mode])

    import generesolver.config as _cfg
    if not getattr(_cfg, "CASE_BRANCHING_ENABLED", True):
        steps = [s for s in steps if s != "case_classifier"]

    state.pending_steps = steps

    if not state.has_vcf and state.task_mode != TaskMode.LOCUS_LOOKUP:
        state.add_warning("No VCF provided — ranking will be phenotype-based only.")
    if not state.has_family_history and state.task_mode != TaskMode.LOCUS_LOOKUP:
        state.add_warning("No family history — inheritance compatibility cannot be assessed.")

    state.mark_done("router")
    return state


def _next_in_canonical(after: str | None, state: CaseState) -> str:
    if after == "case_reclassify":
        after = "case_classifier"
    start_idx = _CANONICAL_ORDER.index(after) + 1 if after in _CANONICAL_ORDER else 0
    for step in _CANONICAL_ORDER[start_idx:]:
        if step in state.pending_steps:
            return step
    return END


def _route_after_classifier(step_name: str, state: CaseState) -> str:
    cc = state.case_class
    decision = cc.decision if cc else None

    if step_name == "case_classifier" and decision is None:
        return "class_retrieval"

    if decision == CaseClass.SINGLE_GENE:
        return _next_in_canonical(step_name, state)

    state.pending_steps = [s for s in state.pending_steps if s not in _GENE_RANKING_STEPS]
    return _BRANCH_NODE.get(decision, _BRANCH_NODE[None])


def _route_after(step_name: str, state: CaseState) -> str:
    if state.stop_after_step and step_name == state.stop_after_step:
        return END
    if step_name in ("case_classifier", "case_reclassify"):
        return _route_after_classifier(step_name, state)
    if step_name == "class_retrieval":
        return "case_reclassify"
    if step_name.startswith("branch_"):
        return "next_steps"
    return _next_in_canonical(step_name, state)


def _route_after_router(state: CaseState) -> str:
    if state.stop_after_step == "router":
        return END
    return _next_in_canonical(None, state)


def _make_node(step_name: str, module_path: str):
    def _run(state: CaseState) -> CaseState:
        if _should_skip(step_name, state):
            logger.info("Skipping step '%s' — preconditions not met.", step_name)
            state.mark_done(step_name)
            return state
        try:
            module = importlib.import_module(module_path)
            logger.info("Running step: %s", step_name)
            state = module.run(state)
        except NotImplementedError:
            logger.info("Step '%s' not yet implemented — skipping.", step_name)
            state.mark_done(step_name)
        except Exception as exc:
            logger.error("Step '%s' failed: %s", step_name, exc)
            if step_name in {"deidentifier", "variant_annotator", "case_classifier", "locus_lookup"}:
                raise RuntimeError(f"Required step '{step_name}' failed; pipeline stopped: {exc}") from exc
            state.add_warning(f"Step '{step_name}' encountered an error: {exc}")
            state.mark_done(step_name)

        if step_name not in state.completed_steps:
            logger.warning("Step '%s' did not mark itself done — forcing.", step_name)
            state.mark_done(step_name)
        return state

    return _run


def _build_graph():
    graph = StateGraph(CaseState)
    graph.add_node("router", _route)
    for name, module_path in STEP_MODULES.items():
        graph.add_node(name, _make_node(name, module_path))
    graph.set_entry_point("router")

    path_map = {name: name for name in (["router"] + list(STEP_MODULES))}
    path_map[END] = END

    graph.add_conditional_edges("router", _route_after_router, path_map)
    for name in STEP_MODULES:
        graph.add_conditional_edges(name, functools.partial(_route_after, name), path_map)

    return graph.compile()


_GRAPH = _build_graph()


def run(state: CaseState) -> CaseState:
    try:
        result = _GRAPH.invoke(state, config={"recursion_limit": _RECURSION_LIMIT})
    except GraphRecursionError:
        logger.error("Step budget (%d) exhausted for case %s — aborting run.",
                     _RECURSION_LIMIT, state.case_id)
        state.add_warning(
            f"Pipeline step budget ({_RECURSION_LIMIT}) exhausted; run was cut short."
        )
        return state
    return CaseState.model_validate(result)


def run_steps(state: CaseState, steps: list[str]) -> CaseState:
    state = _route(state)
    for name in steps:
        module_path = STEP_MODULES.get(name)
        if module_path is None:
            logger.warning("No module registered for step '%s' — skipping.", name)
            continue
        state = _make_node(name, module_path)(state)
    return state
