from __future__ import annotations

import logging
import time

from generesolver import config
from generesolver.components import biochemical
from generesolver.branches.descriptors import BranchDescriptor
from generesolver.integrations import europepmc, gwas_catalog, hpo_a, llm_client, open_targets_api
from generesolver.schemas.biochem import Direction
from generesolver.schemas.case import CaseState
from generesolver.schemas.report import ProvenanceItem

logger = logging.getLogger(__name__)

_MAX_LIT = 6
_MAX_TARGETS = 10
_MAX_PHENO_TERMS = 2

_ULTRA_RARE_DISEASE_THRESHOLD = 2


_MAX_DX = 10


def _phenotype_terms(state: CaseState) -> list[str]:
    return [t.label for t in state.hpo_profile.terms
            if getattr(t, "label", None) and t.status.value == "present"]


def _non_rare_phenotype_terms(state: CaseState) -> list[str]:
    rev = hpo_a.get_hpo_to_diseases_map()
    scored: list[tuple[int, str]] = []
    for t in state.hpo_profile.terms:
        if not getattr(t, "label", None) or t.status.value != "present":
            continue
        try:
            ti = int(str(t.hpo_id).split(":")[1])
        except (ValueError, IndexError):
            continue
        n_diseases = len(rev.get(ti) or ())
        if n_diseases > _ULTRA_RARE_DISEASE_THRESHOLD:
            scored.append((n_diseases, t.label))
    scored.sort(key=lambda x: x[0])
    return [label for _, label in scored]


def _disease_candidates(state: CaseState, min_overlap: int = 2) -> list[str]:
    present = list(state.hpo_profile.present_ids)
    id2label = {t.hpo_id: (getattr(t, "label", None) or t.hpo_id) for t in state.hpo_profile.terms}
    out: list[dict] = []
    seen: set[str] = set()
    try:
        for c in hpo_a.candidate_diseases_for_hpo_profile(present, min_overlap=min_overlap, top_n=_MAX_DX):
            if getattr(c, "label", None) and c.label not in seen:
                seen.add(c.label)
                matched = [id2label.get(h, h) for h in getattr(c, "matched_hpo_ids", ())]
                out.append({"name": c.label, "matched": matched[:6],
                            "overlap": getattr(c, "overlap", len(matched))})
    except Exception as exc:
        logger.warning("HPO-A candidate diseases failed: %s", exc)
    try:
        for d in biochemical.marker_diseases(present)[:_MAX_DX]:
            nm = d.get("name")
            if nm and nm not in seen:
                seen.add(nm)
                markers = [m.get("label") for m in (d.get("markers") or [])[:4] if m.get("label")]
                out.append({"name": nm, "matched": markers, "overlap": len(markers)})
    except Exception as exc:
        logger.warning("biochemical marker diseases failed: %s", exc)
    out = out[:_MAX_DX]
    state.trace.add_retrieval(source="hpo_a", query="phenotype->disease candidates + matched findings",
                              n_results=len(out), results=[c["name"] for c in out][:8])
    return out


def _case_summary(state: CaseState, desc: BranchDescriptor) -> str:
    lines: list[str] = []

    pheno = _phenotype_terms(state)
    if pheno:
        lines.append("PHENOTYPE (HPO):\n" + "\n".join(f"- {p}" for p in pheno))

    abnormal = [m.summary() for m in state.biochem_panel.measurements
                if m.direction != Direction.UNKNOWN]
    if abnormal:
        lines.append("BIOCHEMISTRY:\n" + "\n".join(f"- {a}" for a in abnormal))

    demo = []
    if state.age_years is not None:
        demo.append(f"Age at presentation: {state.age_years} years")
    if state.sex:
        demo.append(f"Sex: {state.sex}")
    if state.consanguinity is not None:
        demo.append(f"Consanguinity: {state.consanguinity}")
    if state.family_history:
        demo.append(f"Family history: {state.family_history}")
    if demo:
        lines.append("DEMOGRAPHICS:\n" + "\n".join(f"- {d}" for d in demo))

    genomic = [f"{c.computed_acmg_assessment.gene} {c.computed_acmg_assessment.variant_id}: "
               f"ACMG {c.computed_acmg_assessment.classification}"
               for c in state.acmg_assessments[:10]]
    if genomic:
        lines.append("GENOMIC FINDINGS:\n" + "\n".join(f"- {g}" for g in genomic))

    if state.case_class:
        cc = state.case_class
        lines.append(
            f"ETIOLOGY VERDICT: {cc.decision.value if cc.decision else 'uncertain'} "
            f"(mode={cc.mode.value}, margin={cc.margin:.2f})"
        )

    return "\n\n".join(lines)


def _literature_query(state: CaseState, desc: BranchDescriptor, disease_seeds: list[str]) -> str:
    seeds = list(disease_seeds[:2])
    seeds += [g.gene for g in state.ranked_genes[:3] if g.gene]

    if not config.LOCAL_MODE:
        seeds += _non_rare_phenotype_terms(state)[:_MAX_PHENO_TERMS]

    if not seeds:
        return ""

    query = " OR ".join(f'"{s}"' for s in dict.fromkeys(seeds))
    if desc.literature_terms:
        query = f"({query}) AND (" + " OR ".join(desc.literature_terms) + ")"
    return query


def _literature(state: CaseState, desc: BranchDescriptor, disease_seeds: list[str]) -> list[dict]:
    query = _literature_query(state, desc, disease_seeds)
    if not query:
        state.trace.add_retrieval(
            source="europepmc", query="", n_results=0, failed=True,
            error=("skipped: no disease/gene hypothesis and no phenotype term common enough "
                   "to send, or LOCAL_MODE blocked the phenotype terms entirely"),
        )
        logger.info("Literature skipped — no non-identifying query available.")
        return []

    t0 = time.time()
    try:
        hits = europepmc.search(query, page_size=_MAX_LIT)
        state.trace.add_retrieval(
            source="europepmc", query=query, n_results=len(hits),
            results=[f"{europepmc.citation(h)} {h['title']}"[:160] for h in hits],
            took_ms=int((time.time() - t0) * 1000),
        )
        return hits
    except Exception as exc:
        state.trace.add_retrieval(
            source="europepmc", query=query, failed=True, error=str(exc),
            took_ms=int((time.time() - t0) * 1000),
        )
        logger.warning("Literature retrieval failed: %s", exc)
        return []


def _open_targets(state: CaseState, desc: BranchDescriptor, disease_seeds: list[str]) -> list[dict]:
    seeds = list(disease_seeds[:2])
    if not config.LOCAL_MODE:
        seeds += _non_rare_phenotype_terms(state)[:_MAX_PHENO_TERMS]
    out: list[dict] = []

    for seed in seeds:
        t0 = time.time()
        try:
            targets = open_targets_api.get_disease_associated_targets(seed, size=_MAX_TARGETS)
            out.extend(targets)
            state.trace.add_retrieval(
                source="open_targets", query=seed, n_results=len(targets),
                results=[f"{t.get('gene_symbol')} ({t.get('score', 0):.2f})" for t in targets[:8]],
                took_ms=int((time.time() - t0) * 1000),
            )
        except Exception as exc:
            state.trace.add_retrieval(
                source="open_targets", query=seed, failed=True, error=str(exc),
                took_ms=int((time.time() - t0) * 1000),
            )
            logger.warning("Open Targets retrieval failed for %r: %s", seed, exc)
    return out


def _gwas(state: CaseState, desc: BranchDescriptor) -> list[dict]:
    genes = [g.gene for g in state.ranked_genes[:5]] or \
            [t.get("gene_symbol") for t in state.diagnostics.get("_ot_targets", [])[:5]]
    genes = [g for g in genes if g]
    if not genes:
        return []

    out: list[dict] = []
    for gene in genes[:5]:
        t0 = time.time()
        try:
            hits = gwas_catalog.lookup(gene=gene)
            out.extend(hits)
            state.trace.add_retrieval(
                source="gwas_catalog", query=gene, n_results=len(hits),
                results=[str(h.get("trait") or h)[:80] for h in hits[:5]],
                took_ms=int((time.time() - t0) * 1000),
            )
        except Exception as exc:
            state.trace.add_retrieval(
                source="gwas_catalog", query=gene, failed=True, error=str(exc),
                took_ms=int((time.time() - t0) * 1000),
            )
    return out


_SYSTEM = (
    "You are a clinical geneticist writing the reasoned conclusion for a case that has ALREADY "
    "been classified by etiology. Do not re-litigate the classification.\n\n"
    "GROUND EVERY CLAIM IN THE EVIDENCE PROVIDED. You are given the patient's findings and the "
    "results of specific database retrievals. Do not introduce genes, diseases, studies or "
    "mechanisms that do not appear in that evidence. If the evidence is too thin to support a "
    "conclusion, say so — an honest 'insufficient evidence' is correct and useful; an invented "
    "mechanism is not.\n\n"
    "Do NOT propose confirmatory tests — that is done once, at the end, over the whole case "
    "record. Your job is the explanation, not the plan.\n\n"
    "Each condition NAME must be a SPECIFIC disease or syndrome — not a phenotype/finding "
    "(e.g. 'Overgrowth') and not a bare mechanism (e.g. 'chromosomal abnormality'). Prefer a "
    "candidate diagnosis when one fits; if the evidence truly cannot resolve a specific "
    "disease, say so in 'reasoning' and give your best specific hypotheses in 'conditions'.\n\n"
    "Reply as ONE JSON object with these exact keys:\n"
    '{"conditions": [{"name": "<specific disease or syndrome>", "likelihood": "<high|moderate|low>", '
    '"supporting": ["<evidence cited>"], "against": ["<evidence>"]}],\n'
    ' "mechanism": "<how the proposed etiology produces THESE findings, 2-4 sentences>",\n'
    ' "reasoning": "<how you got here from the evidence, 2-4 sentences>"}'
)


def _reason(state: CaseState, desc: BranchDescriptor, candidates: list[str],
            lit: list[dict], targets: list[dict], gwas: list[dict]) -> dict:
    evidence = [_case_summary(state, desc)]

    if candidates:
        evidence.append(
            "CANDIDATE DIAGNOSES (phenotype-matched, local HPO-A + biochemical index):\n"
            + "\n".join(f"- {c['name']}" for c in candidates)
            + "\nRank/select from these when one fits the picture; you may instead name a "
              "better-fitting SPECIFIC disease, but do not fall back to a bare mechanism.")

    if targets:
        evidence.append(
            "OPEN TARGETS (disease-associated genes):\n"
            + "\n".join(f"- {t.get('gene_symbol')} (score {t.get('score', 0):.2f})"
                        for t in targets[:_MAX_TARGETS])
        )
    if gwas:
        evidence.append(
            "GWAS CATALOG (trait associations):\n"
            + "\n".join(f"- {g.get('gene', '?')}: {g.get('trait', '?')}" for g in gwas[:10])
        )
    if lit:
        evidence.append(
            "LITERATURE:\n"
            + "\n".join(f"- [{europepmc.citation(h)}] {h['title']}" for h in lit)
        )
    if desc.limitations:
        evidence.append(
            "KNOWN GAPS IN THIS INVESTIGATION (state these honestly, do not paper over them):\n"
            + "\n".join(f"- {l}" for l in desc.limitations)
        )

    system = _SYSTEM + "\n\nTASK FOR THIS ETIOLOGY:\n" + desc.mechanism_focus
    user = "\n\n".join(evidence)

    try:
        res = llm_client.complete_json(system, user, max_tokens=1200)
        return res if isinstance(res, dict) else {}
    except Exception as exc:
        logger.warning("Branch reasoning failed (%s): %s", desc.label, exc)
        state.add_warning(f"Branch reasoning failed for {desc.label}: {exc}")
        return {}



def run_branch(state: CaseState, desc: BranchDescriptor, step: str) -> CaseState:
    candidates = _disease_candidates(state)
    if not candidates:
        candidates = _disease_candidates(state, min_overlap=1)
    seed_names = [c["name"] for c in candidates]
    targets = _open_targets(state, desc, seed_names) if desc.use_open_targets else []
    if targets:
        state.diagnostics["_ot_targets"] = targets
    gwas = _gwas(state, desc) if desc.use_gwas else []
    lit = _literature(state, desc, seed_names) if desc.use_literature else []

    result = _reason(state, desc, candidates, lit, targets, gwas)

    conditions = result.get("conditions") or []
    mechanism = str(result.get("mechanism") or "").strip()
    reasoning = str(result.get("reasoning") or "").strip()

    state.branch_result = {
        "branch": desc.label,
        "conditions": conditions,
        "mechanism": mechanism,
        "reasoning": reasoning,
        "limitations": list(desc.limitations),
        "sources_used": state.trace.sources_used(),
    }

    for limitation in desc.limitations:
        state.add_warning(f"[{desc.label}] {limitation}")

    state.trace.add_step(
        step=step,
        claim=mechanism or f"{desc.label} branch produced no mechanism (insufficient evidence)",
        grounded_in=state.trace.sources_used(),
        confidence=str(conditions[0].get("likelihood", "")) if conditions else "",
    )
    state.provenance.append(ProvenanceItem(
        component=step,
        tool_name=f"{desc.label} evidence branch",
        tool_version="sources: " + (", ".join(state.trace.sources_used()) or "none"),
    ))

    logger.info(
        "Branch %s: %d condition(s), %d next step(s), sources=%s",
        desc.label, len(conditions), len(state.next_steps.tests),
        state.trace.sources_used() or "none",
    )
    state.mark_done(step)
    return state
