from __future__ import annotations

import json
import logging
from typing import Any

from generesolver.integrations.llm_client import LLMResponseError, complete_json, active_backend
from generesolver.integrations.llm_facts import (
    RERANK_ADJUSTMENT_CAP,
    build_case_facts,
    clamp_adjustment,
    validate_grounded_in,
)
from generesolver.config import (
    JUDGE_AUTHORITATIVE_RANK,
    JUDGE_AUTHORITATIVE_DISEASE_ORDER,
    JUDGE_RANK_WEIGHT,
    JUDGE_TOP_K,
    JUDGE_MAX_CANDIDATES,
    JUDGE_SEED_RESERVE,
    JUDGE_TIEBREAK_CAP,
)
from generesolver.schemas.case import CaseState
from generesolver.schemas.gene import GeneRank
from generesolver.schemas.report import (
    AlternativeHypothesis,
    AppliedRerank,
    ConfidenceBand,
    DifferentialDecision,
    ProvenanceItem,
)

logger = logging.getLogger(__name__)

_TOP_K = 10

_MAX_TOKENS = 2048

_REASONING_CHAR_BUDGET = 60
_VALID_CONFIDENCE = {b.value for b in ConfidenceBand}

_CTX_LIMIT = 16384  # MedGemma 27B IT serving context
_SYS_RESERVE = 2000
_CHARS_PER_TOKEN = 2.8
_BUNDLE_MIN_KEEP = 5
_TRIM_FIELD_ORDER = ("literature", "reactome_pathways", "orphanet_diseases",
                     "variants", "acmg", "hpo_match", "lr_evidence")


def _estimate_tokens(text: str) -> int:
    return int(len(text) / _CHARS_PER_TOKEN) + 1


def _bundle_tokens(bundle: dict) -> int:
    return _estimate_tokens(json.dumps(bundle, separators=(",", ":"), ensure_ascii=False))


def _fit_bundle(bundle: dict, max_output_tokens: int) -> dict:
    ctx = _CTX_LIMIT
    budget = ctx - max_output_tokens - _SYS_RESERVE
    if _bundle_tokens(bundle) <= budget:
        return bundle

    b = dict(bundle)
    genes = [dict(g) for g in (b.get("candidate_genes") or [])]
    n_before = len(genes)

    for field in _TRIM_FIELD_ORDER:
        for g in genes:
            g.pop(field, None)
        b["candidate_genes"] = genes
        if _bundle_tokens(b) <= budget:
            logger.info(
                "differential_synthesizer: trimmed %r from all candidate cards to fit "
                "context budget (%d genes kept)", field, len(genes),
            )
            return b

    while len(genes) > _BUNDLE_MIN_KEEP and _bundle_tokens(b) > budget:
        genes.pop()
        b["candidate_genes"] = genes
    logger.warning(
        "differential_synthesizer: candidate pool trimmed %d -> %d gene(s) to fit "
        "context budget", n_before, len(genes),
    )
    return b


def run(state: CaseState) -> CaseState:
    if not state.ranked_genes:
        logger.info("differential_synthesizer: no ranked genes - skip")
        state.mark_done("differential_synthesizer")
        return state

    judge_view = _select_judge_candidates(state)
    candidate_symbols = {g.gene for g in judge_view}
    allowed_orpha: set[str] = {m.orpha_id for m in state.disease_mentions if m.orpha_id}
    for gr in judge_view:
        for oid in (gr.seeded_by_diseases or []):
            allowed_orpha.add(oid)

    bundle = build_case_facts(state, candidates=judge_view, clinical_note=_safe_note(state))
    bundle = _fit_bundle(bundle, max(_PRIORITY_MAX_TOKENS, _MAX_TOKENS))

    if JUDGE_AUTHORITATIVE_RANK:
        priority_raw, rerank_note = _call_priority_llm(bundle, sorted(candidate_symbols))
        rerank_applied = (
            _apply_authoritative_rerank(priority_raw, bundle, candidate_symbols, judge_view)
            if priority_raw is not None else []
        )
    else:
        rerank_raw, rerank_note = _call_rerank_llm_with_salvage(bundle, sorted(candidate_symbols))
        rerank_applied = (
            _apply_rerank(rerank_raw, bundle, candidate_symbols, judge_view)
            if rerank_raw is not None else []
        )

    if rerank_applied:
        state.ranked_genes.sort(key=lambda g: g.final_score, reverse=True)
        for i, gr in enumerate(state.ranked_genes, start=1):
            gr.rank = i
    top_k = state.ranked_genes[:_TOP_K]
    candidate_symbols = {g.gene for g in top_k}

    if JUDGE_AUTHORITATIVE_DISEASE_ORDER:
        _reorder_diseases_by_judge(state)

    decision_raw, decision_note = _call_decision_llm(
        bundle,
        sorted(candidate_symbols),
        sorted(allowed_orpha),
    )

    decision = _parse_and_validate_decision(
        decision_raw,
        bundle,
        candidate_symbols,
        allowed_orpha,
        state,
    )
    if decision is None:
        logger.warning("differential_synthesizer: decision failed validation - fallback stub")
        decision = _fallback_decision(
            state,
            reason=decision_note or "LLM decision failed grounding validation",
        )

    decision.rerank_applied = rerank_applied
    for note in [rerank_note, decision_note]:
        if note:
            decision.flag_human_review = True
            if note not in decision.review_reasons:
                decision.review_reasons.append(note)

    if decision.confidence_band in (ConfidenceBand.LOW, ConfidenceBand.UNCERTAIN):
        decision.flag_human_review = True
        reason = (
            f"Reasoner confidence is {decision.confidence_band.value} — the "
            "candidate evidence does not clearly favour one gene; widen the "
            "differential and review alternatives."
        )
        if reason not in decision.review_reasons:
            decision.review_reasons.append(reason)
    state.confidence_level = decision.confidence_band.value

    state.differential_decision = decision
    _apply_review_flag(state)

    n_adj = len(rerank_applied)
    if JUDGE_AUTHORITATIVE_RANK:
        mode_tag = (
            f"authoritative; judge_weight={JUDGE_RANK_WEIGHT}; "
            f"candidates={len(judge_view)}"
        )
    else:
        mode_tag = f"tiebreak; cap=±{JUDGE_TIEBREAK_CAP}; top_k={len(judge_view)}"
    state.provenance.append(ProvenanceItem(
        component="differential_synthesizer",
        tool_name="LLM judge - rerank + differential (grounded in retrieved facts)",
        tool_version=(
            f"{mode_tag}; rerank_entries={n_adj}; "
            f"fallback={decision.fallback_used}; "
            f"flagged_review={decision.flag_human_review}; "
            f"salvaged={'yes' if (rerank_note or decision_note) else 'no'}"
        ),
    ))
    state.mark_done("differential_synthesizer")
    return state

def _reorder_diseases_by_judge(state: CaseState) -> None:
    diseases = getattr(state, "ranked_diseases", None) or []
    if not diseases or not state.ranked_genes:
        return
    gene_score = {g.gene: float(g.final_score) for g in state.ranked_genes}

    def best_gene_score(d: dict) -> float | None:
        if not isinstance(d, dict):
            return None
        scores = [gene_score[g] for g in (d.get("genes") or []) if g in gene_score]
        return max(scores) if scores else None

    keyed: list[tuple[int, float, dict]] = []
    tail: list[tuple[int, dict]] = []
    for i, d in enumerate(diseases):
        js = best_gene_score(d)
        if js is None:
            tail.append((i, d))
        else:
            keyed.append((i, js, d))

    keyed.sort(key=lambda t: (-t[1], t[0]))
    reordered = [d for _, _, d in keyed] + [d for _, d in tail]
    state.ranked_diseases = reordered


def _safe_note(state: CaseState) -> str | None:
    if state.deidentification_applied and state.clinical_note_deidentified:
        return state.clinical_note_deidentified
    return None


def _select_judge_candidates(state: CaseState) -> list[GeneRank]:
    ranked = state.ranked_genes
    if not JUDGE_AUTHORITATIVE_RANK:
        return list(ranked[:_TOP_K])

    cap = JUDGE_MAX_CANDIDATES
    reserve = min(JUDGE_SEED_RESERVE, cap)
    det_floor = max(0, min(JUDGE_TOP_K, cap - reserve))

    view = list(ranked[:det_floor])
    seen = {g.gene for g in view}

    seeds = [
        g for g in ranked[det_floor:]
        if (g.score_disease_match > 0.0 or bool(g.seeded_by_diseases))
        and g.gene not in seen
    ]
    seeds.sort(key=lambda g: g.score_disease_match, reverse=True)
    for gr in seeds:
        if len(view) >= cap:
            break
        view.append(gr)
        seen.add(gr.gene)

    for gr in ranked[det_floor:]:
        if len(view) >= cap:
            break
        if gr.gene not in seen:
            view.append(gr)
            seen.add(gr.gene)
    return view[:cap]

_PRIORITY_MAX_TOKENS = 4096
_PRIORITY_REASONING_CHARS = 90


def _build_priority_prompt(bundle: dict, candidate_symbols: list[str]) -> str:
    facts_json = json.dumps(bundle, separators=(",", ":"), ensure_ascii=False)
    allowed_genes = ", ".join(candidate_symbols) or "(none)"
    return (
        "You are a clinical geneticist adjudicating candidate genes for ONE patient.\n"
        "Unlike the upstream tools — each of which saw only HPO terms or one "
        "database — you see the WHOLE picture: the clinical_note narrative, the "
        "structured HPO profile, the disease mentions, and every candidate's "
        "evidence. Reason over ALL of it together. In particular, use context in "
        "clinical_note that may NOT have become a structured HPO term (age/onset "
        "and temporal course, consanguinity, family history, a distinctive "
        "descriptive phrase, what is explicitly ABSENT).\n"
        "Use ONLY the FACTS below — do NOT use outside memory; do NOT introduce "
        "genes, diseases, or IDs not in FACTS.\n\n"
        f"FACTS: {facts_json}\n\n"
        f"Score EVERY allowed gene. Allowed genes: {allowed_genes}\n\n"
        "Ranking principle: a gene that matches the patient's SPECIFIC phenotype "
        "(rare/discriminating HPO terms) or fits the narrative and the patient's "
        "named/inferred disease should outrank a generically well-studied gene "
        "that only matches common, non-specific terms. Reward fit to THIS "
        "patient, not fame.\n\n"
        "HOW TO READ THE EVIDENCE — this is the crux:\n"
        "  - candidate_genes[i].lr_evidence.supporting_terms lists, PER candidate, "
        "which patient HPO terms drive its best disease and each term's log_lr "
        "(higher = more discriminating). A candidate whose high-log_lr term is "
        "SPECIFIC to the patient's problem (e.g. a named metabolite/biochemical "
        "finding) beats one driven only by generic/severe terms shared by many "
        "catastrophic diseases (e.g. 'Death in infancy', 'Neonatal death').\n"
        "  - ranked_diseases gives the LR's disease hypotheses with posttest_prob "
        "and causal genes. Cross-check: prefer a candidate that IS the causal gene "
        "of a disease whose top_supporting_terms are the patient's most specific "
        "findings, even if its ranker.final_score is not the highest.\n"
        "  - Do NOT simply echo ranker rank-1. The deterministic score can be "
        "inflated by a single non-specific term; use lr_evidence + ranked_diseases "
        "to check whether the rank-1 gene's match is specific or generic, and "
        "promote a lower-ranked gene when ITS match is the specific one.\n\n"
        "For each gene return:\n"
        "  - priority: number 0.00-1.00 (1.0 = best explains this patient)\n"
        f"  - reasoning: ONE clause, AT MOST {_PRIORITY_REASONING_CHARS} characters. Be terse.\n"
        "  - grounded_in: exactly 1 JSON path into FACTS (e.g. "
        "candidate_genes.0.lr_evidence, candidate_genes.2.hpo_match, "
        "ranked_diseases.0, disease_mentions.0). A gene with no valid path keeps "
        "its prior score, so cite a real path.\n\n"
        "Keep the WHOLE response compact — terse reasoning for every gene.\n"
        "Respond JSON only:\n"
        '{"priorities":[{"gene":"","priority":0.0,"reasoning":"","grounded_in":[]}]}'
    )


def _call_priority_llm(
    bundle: dict,
    candidate_symbols: list[str],
) -> tuple[dict | None, str]:
    prompt = _build_priority_prompt(bundle, candidate_symbols)
    try:
        return complete_json(system="", user=prompt, max_tokens=_PRIORITY_MAX_TOKENS), ""
    except LLMResponseError as exc:
        raw_payload = getattr(exc, "raw_payload", "") or ""
        salvaged = _try_salvage_truncated(raw_payload, array_key="priorities")
        if salvaged is not None:
            n = len(salvaged.get("priorities") or [])
            logger.warning(
                "differential_synthesizer: salvaged truncated priority response "
                "(%d entries recovered)", n,
            )
            return salvaged, f"LLM priority response was truncated; recovered {n} entry(ies)"
        logger.warning("differential_synthesizer: priority LLM failed (%s)", exc)
        return None, f"LLM priority error: {exc}"
    except Exception as exc:
        logger.warning("differential_synthesizer: unexpected priority LLM error (%s)", exc)
        return None, f"unexpected priority LLM error: {exc}"


def _apply_authoritative_rerank(
    raw: Any,
    bundle: dict,
    candidate_symbols: set[str],
    judge_view: list[GeneRank],
) -> list[AppliedRerank]:

    if not isinstance(raw, dict):
        return []
    entries = raw.get("priorities") or []
    if not isinstance(entries, list):
        return []

    gene_index = {g.gene: g for g in judge_view}
    judged: dict[str, tuple[float, str, list[str]]] = {}
    for d in entries:
        if not isinstance(d, dict):
            continue
        gene = (d.get("gene") or "").strip()
        if gene not in candidate_symbols or gene not in gene_index:
            continue
        valid, _invalid = validate_grounded_in(d.get("grounded_in") or [], bundle)
        if not valid:
            logger.debug("priority: %s ungrounded -> deterministic score kept", gene)
            continue
        try:
            priority = float(d.get("priority"))
        except (TypeError, ValueError):
            continue
        priority = min(1.0, max(0.0, priority))
        reasoning = (d.get("reasoning") or "").strip()
        judged[gene] = (priority, reasoning, valid)

    if not judged:
        return []

    w = JUDGE_RANK_WEIGHT
    applied: list[AppliedRerank] = []
    for gene, (priority, reasoning, valid) in judged.items():
        gr = gene_index[gene]
        det = float(gr.final_score)
        blended = round(min(1.0, max(0.0, w * priority + (1.0 - w) * det)), 4)
        delta = round(blended - det, 4)
        gr.rerank_adjustment = delta
        gr.rerank_reasoning = reasoning
        gr.rerank_grounded_in = valid
        gr.final_score = blended
        applied.append(AppliedRerank(
            gene=gene,
            adjustment=delta,
            reasoning=reasoning,
            grounded_in=valid,
        ))
    return applied


def _build_rerank_prompt(
    bundle: dict,
    candidate_symbols: list[str],
) -> str:
    facts_json = json.dumps(bundle, separators=(",", ":"), ensure_ascii=False)
    allowed_genes = ", ".join(candidate_symbols) or "(none)"
    cap = JUDGE_TIEBREAK_CAP

    return (
        "You break near-ties between candidate genes for a clinical-genomics case.\n"
        "The deterministic ranker order is final except where two genes have almost the "
        "same ranker score; your adjustments only matter there.\n"
        "Use ONLY the FACTS. Do NOT use outside memory.\n\n"
        f"FACTS: {facts_json}\n\n"
        f"Allowed genes: {allowed_genes}\n\n"
        f"Return at most 3 rerank_adjustments. Adjustment must be between -{cap:.2f} and {cap:.2f}.\n"
        "Only include genes that need a non-zero change.\n"
        "Each reasoning must be one short sentence.\n"
        "Each grounded_in must contain at most 2 valid JSON paths into FACTS.\n\n"
        "Respond JSON only:\n"
        '{"rerank_adjustments":[{"gene":"","adjustment":0.0,"reasoning":"","grounded_in":[]}]}'
    )

def _build_decision_prompt(
    bundle: dict,
    candidate_symbols: list[str],
    allowed_orpha: list[str],
) -> str:
    facts_json = json.dumps(bundle, separators=(",", ":"), ensure_ascii=False)
    allowed_genes = ", ".join(candidate_symbols) or "(none)"
    allowed_orpha_str = ", ".join(allowed_orpha) or "(none)"

    return (
        "You make the final differential decision for a clinical-genomics case.\n"
        "Use ONLY the FACTS. Do NOT introduce genes, diseases, ORPHA IDs, OMIM IDs, or pathways not in FACTS.\n"
        "Do NOT use outside memory.\n\n"
        f"FACTS: {facts_json}\n\n"
        f"Allowed genes: {allowed_genes}\n"
        f"Allowed ORPHA: {allowed_orpha_str}\n\n"
        "Choose most_likely_gene from allowed genes only.\n"
        "most_likely_orpha must be from allowed ORPHA or null.\n"
        "grounded_in must contain at most 3 simple top-level paths into FACTS. "
"Use ONLY paths like candidate_genes.0, candidate_genes.1, disease_mentions.0, hpo_profile.present.0. "
"Do NOT use deeper paths such as candidate_genes.0.ranker.final_score. "
"Do NOT include case_id. Do NOT use colon syntax.\n"
        "Use at most 2 alternatives and at most 3 decisive_tests.\n"
        "primary_reasoning must be <=300 characters.\n\n"
        "Respond JSON only:\n"
        '{"decision":{"most_likely_gene":"","most_likely_orpha":null,'
        '"most_likely_disease_label":"","confidence_band":"uncertain",'
        '"primary_reasoning":"","grounded_in":[],'
        '"alternatives":[{"gene":"","orpha_id":null,"disease_label":"",'
        '"brief_reasoning":"","grounded_in":[]}],'
        '"decisive_tests":[],"flag_human_review":true,"review_reasons":[]}}'
    )

def _call_llm_with_salvage(
    bundle: dict,
    candidate_symbols: list[str],
    allowed_orpha: list[str],
) -> tuple[dict | None, str]:
    prompt = _build_prompt(bundle, candidate_symbols, allowed_orpha)
    try:
        return complete_json(system="", user=prompt, max_tokens=_MAX_TOKENS), ""
    except LLMResponseError as exc:
        raw_payload = getattr(exc, "raw_payload", "") or ""
        salvaged = _try_salvage_truncated(raw_payload)
        if salvaged is not None:
            n_recovered = len(salvaged.get("rerank_adjustments") or [])
            logger.warning(
                "differential_synthesizer: salvaged truncated LLM response "
                "(%d rerank entries recovered, decision missing)", n_recovered,
            )
            return salvaged, (
                f"LLM response was truncated; recovered {n_recovered} rerank "
                f"entry(ies), differential decision fell back to stub"
            )
        logger.warning("differential_synthesizer: LLM failed (%s) - fallback stub", exc)
        return None, f"LLM error: {exc}"
    except Exception as exc:
        logger.warning("differential_synthesizer: unexpected error (%s) - fallback stub", exc)
        return None, f"unexpected error: {exc}"

def _call_rerank_llm_with_salvage(
    bundle: dict,
    candidate_symbols: list[str],
) -> tuple[dict | None, str]:
    prompt = _build_rerank_prompt(bundle, candidate_symbols)

    try:
        return complete_json(system="", user=prompt, max_tokens=1024), ""
    except LLMResponseError as exc:
        raw_payload = getattr(exc, "raw_payload", "") or ""
        salvaged = _try_salvage_truncated(raw_payload)

        if salvaged is not None:
            n_recovered = len(salvaged.get("rerank_adjustments") or [])
            logger.warning(
                "differential_synthesizer: salvaged truncated rerank response "
                "(%d rerank entries recovered)",
                n_recovered,
            )
            return salvaged, f"LLM rerank response was truncated; recovered {n_recovered} entry(ies)"

        logger.warning("differential_synthesizer: rerank LLM failed (%s)", exc)
        return None, f"LLM rerank error: {exc}"


def _call_decision_llm(
    bundle: dict,
    candidate_symbols: list[str],
    allowed_orpha: list[str],
) -> tuple[dict | None, str]:
    prompt = _build_decision_prompt(bundle, candidate_symbols, allowed_orpha)

    try:
        return complete_json(system="", user=prompt, max_tokens=2048), ""
    except LLMResponseError as exc:
        logger.warning("differential_synthesizer: decision LLM failed (%s)", exc)
        return None, f"LLM decision error: {exc}"
    except Exception as exc:
        logger.warning("differential_synthesizer: unexpected decision LLM error (%s)", exc)
        return None, f"unexpected decision LLM error: {exc}"

def _try_salvage_truncated(raw_payload: str, array_key: str = "rerank_adjustments") -> dict | None:
    if not raw_payload:
        return None
    needle = f'"{array_key}":['
    start = raw_payload.find(needle)
    if start < 0:
        return None
    array_start = start + len(needle)

    complete_elements: list[str] = []
    i = array_start
    n = len(raw_payload)
    while i < n:
        while i < n and raw_payload[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        if raw_payload[i] == "]":
            break
        if raw_payload[i] != "{":
            break
        depth = 0
        in_string = False
        escaped = False
        obj_start = i
        obj_end = -1
        while i < n:
            ch = raw_payload[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        obj_end = i + 1
                        break
            i += 1
        if obj_end < 0:
            break
        complete_elements.append(raw_payload[obj_start:obj_end])
        i = obj_end

    if not complete_elements:
        return None

    patched = (
        f'{{"{array_key}":['
        + ",".join(complete_elements)
        + "]}"
    )
    try:
        return json.loads(patched)
    except json.JSONDecodeError:
        return None
def _apply_rerank(
    raw: Any,
    bundle: dict,
    candidate_symbols: set[str],
    top_k_view: list,
) -> list[AppliedRerank]:
    if not isinstance(raw, dict):
        return []
    decisions = raw.get("rerank_adjustments") or []
    if not isinstance(decisions, list):
        return []

    applied: list[AppliedRerank] = []
    gene_index = {g.gene: g for g in top_k_view}

    for d in decisions:
        if not isinstance(d, dict):
            continue
        gene = (d.get("gene") or "").strip()
        if gene not in candidate_symbols:
            logger.debug("rerank: non-candidate gene %r - discarded", gene)
            continue
        grounded_raw = d.get("grounded_in") or []
        valid, _invalid = validate_grounded_in(grounded_raw, bundle)
        adjustment = clamp_adjustment(d.get("adjustment", 0.0), cap=JUDGE_TIEBREAK_CAP)
        if not valid and adjustment != 0.0:
            logger.debug("rerank: %s ungrounded -> zeroed", gene)
            adjustment = 0.0
        if adjustment == 0.0:
            continue
        reasoning = (d.get("reasoning") or "").strip()

        gr = gene_index.get(gene)
        if gr is None:
            continue
        gr.rerank_adjustment = adjustment
        gr.rerank_reasoning = reasoning
        gr.rerank_grounded_in = valid
        gr.final_score = round(min(1.0, max(0.0, gr.final_score + adjustment)), 4)

        applied.append(AppliedRerank(
            gene=gene,
            adjustment=adjustment,
            reasoning=reasoning,
            grounded_in=valid,
        ))

    return applied

def _parse_and_validate_decision(
    raw: Any,
    bundle: dict,
    candidate_symbols: set[str],
    allowed_orpha: set[str],
    state: CaseState,
) -> DifferentialDecision | None:
    if not isinstance(raw, dict):
        logger.warning("decision validation failed: raw is not dict: %r", type(raw))
        return None

    decision_obj = raw.get("decision")
    if not isinstance(decision_obj, dict):
        logger.warning(
            "decision validation failed: missing/non-dict decision. raw keys=%s",
            list(raw.keys()),
        )
        return None

    primary_gene = (decision_obj.get("most_likely_gene") or "").strip()
    if primary_gene not in candidate_symbols:
        logger.warning(
            "decision validation failed: primary gene %r not in candidates %s",
            primary_gene,
            sorted(candidate_symbols),
        )
        return None

    primary_orpha = decision_obj.get("most_likely_orpha")
    if isinstance(primary_orpha, str):
        primary_orpha = primary_orpha.strip() or None
        if primary_orpha and primary_orpha not in allowed_orpha:
            logger.warning(
                "decision validation note: ORPHA %r not allowed; setting to None. allowed=%s",
                primary_orpha,
                sorted(allowed_orpha),
            )
            primary_orpha = None
    else:
        primary_orpha = None

    primary_label = (decision_obj.get("most_likely_disease_label") or "").strip()

    confidence_raw = (decision_obj.get("confidence_band") or "uncertain").strip().lower()
    if confidence_raw not in _VALID_CONFIDENCE:
        logger.warning(
            "decision validation note: invalid confidence %r -> uncertain",
            confidence_raw,
        )
        confidence_raw = "uncertain"

    confidence = ConfidenceBand(confidence_raw)

    primary_reasoning = (decision_obj.get("primary_reasoning") or "").strip()

    grounded_raw = decision_obj.get("grounded_in") or []
    valid_paths, invalid_paths = validate_grounded_in(grounded_raw, bundle)

    if not valid_paths:
        logger.warning(
            "decision grounding failed; using deterministic candidate-level grounding. "
            "raw_grounded_in=%r invalid=%r",
            grounded_raw,
            invalid_paths,
        )

        candidate_key = "candidate_genes" if "candidate_genes" in bundle else "candidates"
        candidate_list = bundle.get(candidate_key) or []

        valid_paths = [
            f"{candidate_key}.{i}"
            for i, c in enumerate(candidate_list)
            if isinstance(c, dict) and c.get("gene") == primary_gene
        ]

        if not valid_paths:
            logger.warning(
                "decision validation failed: no candidate-level facts available for primary gene %r",
                primary_gene,
            )
            return None

    alternatives: list[AlternativeHypothesis] = []
    alt_list = decision_obj.get("alternatives") or []

    if isinstance(alt_list, list):
        for a in alt_list:
            if not isinstance(a, dict):
                continue

            gene = (a.get("gene") or "").strip()

            if gene not in candidate_symbols:
                logger.debug(
                    "decision alternative discarded: gene %r not in candidates %s",
                    gene,
                    sorted(candidate_symbols),
                )
                continue

            if gene == primary_gene:
                logger.debug(
                    "decision alternative discarded: gene %r is same as primary",
                    gene,
                )
                continue

            orpha = a.get("orpha_id")
            if isinstance(orpha, str):
                orpha = orpha.strip() or None
                if orpha and orpha not in allowed_orpha:
                    logger.debug(
                        "decision alternative ORPHA %r not allowed; setting to None",
                        orpha,
                    )
                    orpha = None
            else:
                orpha = None

            alt_grounded_raw = a.get("grounded_in") or []
            alt_valid, alt_invalid = validate_grounded_in(alt_grounded_raw, bundle)

            if not alt_valid:
                logger.debug(
                    "alternative grounding failed for %s; using deterministic candidate-level grounding. "
                    "raw_grounded_in=%r invalid=%r",
                    gene,
                    alt_grounded_raw,
                    alt_invalid,
                )

                candidate_key = "candidate_genes" if "candidate_genes" in bundle else "candidates"
                candidate_list = bundle.get(candidate_key) or []

                alt_valid = [
                    f"{candidate_key}.{i}"
                    for i, c in enumerate(candidate_list)
                    if isinstance(c, dict) and c.get("gene") == gene
                ]

                if not alt_valid:
                    logger.debug(
                        "alternative %s has no valid candidate-level grounding; skipping alternative",
                        gene,
                    )
                    continue

            gr = next((g for g in state.ranked_genes if g.gene == gene), None)

            alternatives.append(
                AlternativeHypothesis(
                    gene=gene,
                    orpha_id=orpha,
                    disease_label=(a.get("disease_label") or "").strip(),
                    score_in_ranker=float(gr.final_score) if gr else 0.0,
                    brief_reasoning=(a.get("brief_reasoning") or "").strip(),
                )
            )

    decisive_tests = [
        str(t).strip()
        for t in (decision_obj.get("decisive_tests") or [])
        if str(t).strip()
    ][:6]

    flag = bool(decision_obj.get("flag_human_review", True))

    review_reasons = [
        str(r).strip()
        for r in (decision_obj.get("review_reasons") or [])
        if str(r).strip()
    ]

    return DifferentialDecision(
        most_likely_gene=primary_gene,
        most_likely_orpha=primary_orpha,
        most_likely_disease_label=primary_label,
        confidence_band=confidence,
        primary_reasoning=primary_reasoning,
        alternatives=alternatives,
        decisive_tests=decisive_tests,
        flag_human_review=flag,
        review_reasons=review_reasons,
        fallback_used=False,
        grounded_in=valid_paths,
    )


def _apply_review_flag(state: CaseState) -> None:
    if state.differential_decision and state.differential_decision.flag_human_review:
        state.requires_human_review = True
        for reason in state.differential_decision.review_reasons:
            if reason and reason not in state.human_review_reasons:
                state.human_review_reasons.append(reason)


def _fallback_decision(state: CaseState, *, reason: str) -> DifferentialDecision:
    top = state.ranked_genes[0] if state.ranked_genes else None
    if top is None:
        return DifferentialDecision(
            most_likely_gene="",
            confidence_band=ConfidenceBand.UNCERTAIN,
            primary_reasoning=f"No candidate genes available. {reason}",
            flag_human_review=True,
            review_reasons=[f"No candidates produced by deterministic ranker. {reason}"],
            fallback_used=True,
        )
    alternatives: list[AlternativeHypothesis] = []
    for gr in state.ranked_genes[1:3]:
        alternatives.append(AlternativeHypothesis(
            gene=gr.gene,
            orpha_id=(gr.seeded_by_diseases[0] if gr.seeded_by_diseases else None),
            score_in_ranker=float(gr.final_score),
            brief_reasoning=gr.prioritization_reason or "see deterministic score breakdown",
        ))
    primary_orpha = top.seeded_by_diseases[0] if top.seeded_by_diseases else None
    confidence = (
        ConfidenceBand(top.confidence)
        if top.confidence in _VALID_CONFIDENCE
        else ConfidenceBand.UNCERTAIN
    )
    return DifferentialDecision(
        most_likely_gene=top.gene,
        most_likely_orpha=primary_orpha,
        confidence_band=confidence,
        primary_reasoning=(
            f"Top-ranked candidate by deterministic score: {top.gene} "
            f"(final={top.final_score:.3f}). {top.prioritization_reason}. "
            f"LLM judge unavailable - review the per-gene score breakdown directly."
        ),
        alternatives=alternatives,
        decisive_tests=[],
        flag_human_review=True,
        review_reasons=[f"Differential fell back to deterministic stub. {reason}"],
        fallback_used=True,
    )


def _build_prompt(
    bundle: dict,
    candidate_symbols: list[str],
    allowed_orpha: list[str],
) -> str:
    facts_json = json.dumps(bundle, separators=(",", ":"), ensure_ascii=False)
    allowed_genes = ", ".join(candidate_symbols) or "(none)"
    allowed_orpha_str = ", ".join(allowed_orpha) or "(none)"
    cap = RERANK_ADJUSTMENT_CAP

    char_budget = _REASONING_CHAR_BUDGET
    return (
        "You judge candidate genes for a clinical-genomics case. "
        "The FACTS below come from authoritative databases (Orphanet, "
        "Open Targets, UniProt, Reactome, gnomAD, HPO/Phenodigm, ClinVar). "
        "Do NOT introduce gene symbols, ORPHA IDs, OMIM IDs, pathway "
        "names, or disease labels that are not in FACTS. Do NOT use "
        "training-data memory.\n\n"
        "LENGTH BUDGET - BE BRIEF:\n"
        f"  - Each `reasoning` / `brief_reasoning`: ONE short sentence, <={char_budget} chars.\n"
        "  - `primary_reasoning`: ONE paragraph, <=400 chars.\n"
        "  - `grounded_in`: at most 3 paths per claim.\n"
        "  - rerank_adjustments: at most 5 entries (only the genes that "
        "actually need a non-zero nudge; omit the rest).\n"
        "  - alternatives: at most 2.\n"
        "  - decisive_tests: at most 3 short test names.\n\n"
        f"FACTS: {facts_json}\n\n"
        "Two jobs in one JSON response:\n\n"
        f"1) rerank_adjustments - per-gene nudge between -{cap:.2f} and {cap:.2f}. "
        "Use a positive adjustment such as 0.02 for: Assessed Disease-causing germline link to a "
        "clinician-named disease, matching HPO, and high constraint. "
        "Use a negative adjustment such as -0.02 for: only candidate-tested-in links, low constraint, "
        "no HPO match. Each entry needs reasoning AND grounded_in JSON "
        "paths into FACTS (paths that don't resolve -> adjustment zeroed).\n\n"
        "2) decision:\n"
        "  - most_likely_gene from allowed list\n"
        "  - most_likely_orpha from allowed list or null\n"
        "  - most_likely_disease_label as in FACTS\n"
        "  - confidence_band: high|moderate|low|uncertain\n"
        "  - primary_reasoning, alternatives, decisive_tests, "
        "flag_human_review, review_reasons, grounded_in\n"
        "  - flag_human_review = true if top-2 final_scores differ <0.05 "
        "OR confidence low/uncertain OR safety_flags present.\n\n"
        f"Allowed genes: {allowed_genes}\n"
        f"Allowed ORPHA: {allowed_orpha_str}\n\n"
        "Important JSON rule: never write a leading plus sign before positive numbers. "
        "Use 0.02, not +0.02.\n\n"
        "Respond with JSON only, no markdown fences, no commentary:\n"
        '{"rerank_adjustments":[{"gene":"","adjustment":0.0,'
        '"reasoning":"","grounded_in":[]}],'
        '"decision":{"most_likely_gene":"","most_likely_orpha":null,'
        '"most_likely_disease_label":"","confidence_band":"uncertain",'
        '"primary_reasoning":"","grounded_in":[],'
        '"alternatives":[{"gene":"","orpha_id":null,"disease_label":"",'
        '"brief_reasoning":"","grounded_in":[]}],'
        '"decisive_tests":[],"flag_human_review":true,"review_reasons":[]}}'
    )
