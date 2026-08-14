from __future__ import annotations

import logging
from typing import Any

from generesolver.schemas.case import CaseState
from generesolver.schemas.clinical import DiseaseStatus
from generesolver.schemas.gene import GeneRank
from generesolver.schemas.hpo import HPOStatus

logger = logging.getLogger(__name__)

_MAX_ORPHA_PER_GENE = 3
_MAX_OT_DISEASES = 3
_MAX_PATHWAYS = 2
_MAX_HPO_MATCHED = 5
_MAX_HPO_UNMATCHED = 3
_FUNCTION_EXCERPT_CHARS = 180

RERANK_ADJUSTMENT_CAP = 0.20


def build_gene_facts(
    gene_rank: GeneRank,
    state: CaseState,
    *,
    max_orpha: int = _MAX_ORPHA_PER_GENE,
    max_pathways: int = _MAX_PATHWAYS,
    max_hpo_matched: int = _MAX_HPO_MATCHED,
) -> dict[str, Any]:
    from generesolver.integrations import orphanet

    gene = gene_rank.gene

    seeded_set = set(gene_rank.seeded_by_diseases or [])
    orpha_links: list[dict[str, Any]] = []
    seen_orpha: set[str] = set()
    pairs = list(orphanet.get_diseases_for_gene(gene))
    pairs.sort(
        key=lambda p: (p[0] not in seeded_set, -p[1].strength),
    )
    for orpha_id, ga in pairs[:max_orpha]:
        if orpha_id in seen_orpha:
            continue
        seen_orpha.add(orpha_id)
        disease = orphanet.get_by_orpha_id(orpha_id)
        orpha_links.append({
            "orpha_id": orpha_id,
            "label": disease.label if disease else "",
            "association_type": ga.association_type,
            "validation_status": ga.validation_status,
            "strength": round(ga.strength, 3),
            "seeded_by_clinician_mention": orpha_id in seeded_set,
        })

    hpo_match = _build_hpo_match(gene, gene_rank, state, max_hpo_matched)

    acmg_facts = [
        {
            "variant_id": comp.computed_acmg_assessment.variant_id,
            "disease_id": comp.computed_acmg_assessment.disease_id,
            "classification": comp.computed_acmg_assessment.classification,
            "classification_status": comp.computed_acmg_assessment.classification_status,
            "pathogenic_criteria": list(comp.computed_acmg_assessment.pathogenic_criteria),
            "benign_criteria": list(comp.computed_acmg_assessment.benign_criteria),
            "clinvar_concordance": comp.concordance,
        }
        for comp in state.acmg_assessments if comp.computed_acmg_assessment.gene == gene
    ]
    variant_facts = [
        {
            "variant_id": v.variant_id,
            "consequence": v.consequence,
            "impact": v.impact,
            "gnomad_af": v.gnomad_af,
            "clinvar": v.clinvar_significance,
        }
        for v in state.annotated_variants if v.gene == gene
    ]

    bio = next((b for b in state.biological_evidence if b.gene == gene), None)
    open_targets: dict[str, Any] = {}
    uniprot: dict[str, Any] = {}
    pathways: list[str] = []
    if bio is not None:
        if bio.disease_associations or bio.support_score:
            open_targets = {
                "best_diseases": list(bio.disease_associations[:_MAX_OT_DISEASES]),
                "score": round(float(bio.support_score), 3),
            }
        if bio.uniprot_id or bio.known_function:
            uniprot = {
                "uniprot_id": bio.uniprot_id,
                "function": (bio.known_function or "")[:_FUNCTION_EXCERPT_CHARS],
            }
        pathways = list(bio.pathways[:max_pathways])

    from generesolver.integrations import gnomad_constraint
    constraint_score = gnomad_constraint.constraint_score(gene)

    lit = next((l for l in state.literature_evidence if l.gene == gene), None)
    literature: dict[str, Any] = {}
    if lit is not None and lit.support_level not in (None, "", "unknown", "none"):
        literature = {"support": lit.support_level, "n_pmids": len(lit.pmids)}

    inh = next((i for i in state.inheritance_results if i.gene == gene), None)
    inheritance: dict[str, Any] | None = None
    if inh is not None and inh.compatible is not None:
        inheritance = {
            "model": inh.suspected_model.value,
            "compatible": inh.compatible,
        }

    ranker: dict[str, Any] = {
        "rank": gene_rank.rank,
        "final_score": round(float(gene_rank.final_score), 4),
        "confidence": gene_rank.confidence,
    }
    for label, val in (
        ("phenotype", gene_rank.score_phenotype),
        ("variant", gene_rank.score_variant),
        ("gene_disease", gene_rank.score_gene_disease),
        ("literature", gene_rank.score_literature),
        ("gwas", gene_rank.score_gwas),
        ("disease_match", gene_rank.score_disease_match),
    ):
        if val and val != 0.0:
            ranker[f"score_{label}"] = round(float(val), 4)
    if gene_rank.prioritization_reason:
        ranker["prior_reason"] = gene_rank.prioritization_reason
    lr_evidence = _build_lr_evidence(gene, state)
    card: dict[str, Any] = {
        "gene": gene,
        "ranker": ranker,
    }
    if lr_evidence:
        card["lr_evidence"] = lr_evidence
    if gene_rank.hgnc_id:
        card["hgnc_id"] = gene_rank.hgnc_id
    if orpha_links:
        card["orphanet_diseases"] = orpha_links
    if hpo_match.get("patient_terms_matched_by_gene"):
        card["hpo_match"] = hpo_match
    if acmg_facts:
        card["acmg"] = acmg_facts
    if variant_facts:
        card["variants"] = variant_facts
    if open_targets:
        card["open_targets"] = open_targets
    if uniprot:
        card["uniprot"] = uniprot
    if pathways:
        card["reactome_pathways"] = pathways
    if constraint_score and constraint_score > 0.0:
        card["gnomad_constraint"] = round(float(constraint_score), 4)
    if literature:
        card["literature"] = literature
    if inheritance is not None:
        card["inheritance"] = inheritance
    return card


_MAX_LR_SUPPORTING_TERMS = 6


def _hpo_label(hpo_id: str, label_map: dict[str, str]) -> str:
    if hpo_id in label_map:
        return label_map[hpo_id]
    try:
        from generesolver.integrations import hpo_db
        return str(hpo_db._get_ontology().get_hpo_object(hpo_id).name)
    except Exception:
        return ""


def _build_lr_evidence(gene: str, state: CaseState) -> dict[str, Any]:
    try:
        from generesolver.integrations import lr_score
        if not lr_score.is_available():
            return {}
        ev = lr_score.evidence_for_gene(
            gene, state.hpo_profile.present_ids, state.hpo_profile.absent_ids
        )
    except Exception:
        return {}
    if not ev:
        return {}

    label_map = {t.hpo_id: t.label for t in state.hpo_profile.terms}
    terms = sorted(
        ev.get("supporting_terms", []),
        key=lambda s: -float(s.get("log_lr", 0.0)),
    )[:_MAX_LR_SUPPORTING_TERMS]
    supporting = [
        {
            "hpo_id": s["hpo_id"],
            "label": _hpo_label(s["hpo_id"], label_map),
            "log_lr": round(float(s.get("log_lr", 0.0)), 3),
        }
        for s in terms
    ]
    out: dict[str, Any] = {
        "best_disease_id": ev.get("disease_id"),
        "best_disease_name": ev.get("disease_name"),
        "log_lr": round(float(ev.get("log_lr", 0.0)), 3),
        "posttest_prob": round(float(ev.get("posttest_prob", 0.0)), 4),
    }
    if supporting:
        out["supporting_terms"] = supporting
    return out


def _build_hpo_match(
    gene: str,
    gene_rank: GeneRank,
    state: CaseState,
    max_matched: int,
) -> dict[str, Any]:
    from generesolver.integrations import hpo_db

    present_ids = state.hpo_profile.present_ids
    present_int_ids: set[int] = set()
    for hpo_id in present_ids:
        try:
            present_int_ids.add(int(hpo_id.split(":")[1]))
        except (ValueError, IndexError):
            continue

    try:
        gene_hpo_int_ids = hpo_db.get_merged_gene_hpo_ids(gene)
    except Exception:
        gene_hpo_int_ids = set()

    matched_ints = present_int_ids & gene_hpo_int_ids
    matched_ids = {f"HP:{i:07d}" for i in matched_ints}

    matched: list[dict[str, str]] = []
    unmatched: list[dict[str, str]] = []
    for term in state.hpo_profile.terms:
        if term.status != HPOStatus.PRESENT:
            continue
        item = {"hpo_id": term.hpo_id, "label": term.label}
        if term.hpo_id in matched_ids:
            matched.append(item)
        else:
            unmatched.append(item)

    out: dict[str, Any] = {
        "patient_terms_matched_by_gene": matched[:max_matched],
        "phenodigm_similarity": round(float(gene_rank.score_phenotype), 4),
    }
    if unmatched:
        out["patient_terms_unmatched"] = unmatched[:_MAX_HPO_UNMATCHED]
    return out


def build_case_facts(
    state: CaseState,
    *,
    top_k: int = 5,
    include_gene_facts: bool = True,
    candidates: list[GeneRank] | None = None,
    clinical_note: str | None = None,
) -> dict[str, Any]:
    hpo_terms = state.hpo_profile.terms
    hpo: dict[str, Any] = {}
    present_list = [
        {"hpo_id": t.hpo_id, "label": t.label}
        for t in hpo_terms if t.status == HPOStatus.PRESENT
    ]
    absent_list = [
        {"hpo_id": t.hpo_id, "label": t.label}
        for t in hpo_terms if t.status == HPOStatus.ABSENT
    ]
    uncertain_list = [
        {"hpo_id": t.hpo_id, "label": t.label}
        for t in hpo_terms if t.status == HPOStatus.UNCERTAIN
    ]
    if present_list:
        hpo["present"] = present_list
    if absent_list:
        hpo["absent"] = absent_list
    if uncertain_list:
        hpo["uncertain"] = uncertain_list

    disease_mentions: list[dict[str, Any]] = []
    for m in state.disease_mentions:
        d: dict[str, Any] = {
            "label": m.label,
            "status": m.status.value,
        }
        if m.orpha_id:
            d["orpha_id"] = m.orpha_id
        if m.omim_ids:
            d["omim_ids"] = list(m.omim_ids)[:2]
        disease_mentions.append(d)

    top_genes = candidates if candidates is not None else state.ranked_genes[:top_k]
    candidate_genes = (
        [build_gene_facts(gr, state) for gr in top_genes]
        if include_gene_facts else
        [{"gene": gr.gene, "rank": gr.rank,
          "final_score": round(float(gr.final_score), 4)}
         for gr in top_genes]
    )

    bundle: dict[str, Any] = {
        "case_id": state.case_id,
        "candidate_genes": candidate_genes,
    }
    ranked_dz = _build_ranked_diseases(state)
    if ranked_dz:
        bundle["ranked_diseases"] = ranked_dz
    if clinical_note and clinical_note.strip():
        bundle["clinical_note"] = clinical_note.strip()
    if hpo:
        bundle["hpo_profile"] = hpo
    if disease_mentions:
        bundle["disease_mentions"] = disease_mentions
    if state.safety_flags:
        bundle["safety_flags"] = list(state.safety_flags)
    ctx: dict[str, Any] = {}
    if state.input.sex:
        ctx["sex"] = state.input.sex
    if state.input.consanguinity is not None:
        ctx["consanguinity"] = state.input.consanguinity
    if state.has_family_history:
        ctx["family_history_present"] = True
    if ctx:
        bundle["patient_context"] = ctx
    return bundle


_MAX_RANKED_DISEASES = 8


def _build_ranked_diseases(state: CaseState) -> list[dict[str, Any]]:
    ranked = getattr(state, "ranked_diseases", None) or []
    label_map = {t.hpo_id: t.label for t in state.hpo_profile.terms}
    out: list[dict[str, Any]] = []
    for d in ranked[:_MAX_RANKED_DISEASES]:
        if not isinstance(d, dict):
            continue
        terms = sorted(
            d.get("supporting_terms", []) or [],
            key=lambda s: -float(s.get("log_lr", 0.0)),
        )[:4]
        entry: dict[str, Any] = {
            "disease_id": d.get("disease_id"),
            "name": d.get("name"),
            "posttest_prob": round(float(d.get("posttest_prob", 0.0)), 4),
            "genes": list(d.get("genes", []) or [])[:5],
        }
        if terms:
            entry["top_supporting_terms"] = [
                {"hpo_id": s["hpo_id"],
                 "label": _hpo_label(s["hpo_id"], label_map),
                 "log_lr": round(float(s.get("log_lr", 0.0)), 3)}
                for s in terms
            ]
        out.append(entry)
    return out


def _walk_path(obj: Any, path: str) -> bool:
    if not path:
        return False

    def _step(cur: Any, token: str) -> tuple[Any, bool]:
        if isinstance(cur, list) and token.isdigit():
            idx = int(token)
            if 0 <= idx < len(cur):
                return cur[idx], True
            return None, False
        if isinstance(cur, dict) and token in cur:
            return cur[token], True
        return None, False

    cur: Any = obj
    token = ""
    i = 0
    while i < len(path):
        ch = path[i]
        if ch == ".":
            if token:
                cur, ok = _step(cur, token)
                if not ok:
                    return False
                token = ""
            i += 1
        elif ch == "[":
            if token:
                cur, ok = _step(cur, token)
                if not ok:
                    return False
                token = ""
            end = path.find("]", i)
            if end < 0:
                return False
            try:
                idx = int(path[i + 1:end])
            except ValueError:
                return False
            if not isinstance(cur, list) or not (0 <= idx < len(cur)):
                return False
            cur = cur[idx]
            i = end + 1
        else:
            token += ch
            i += 1
    if token:
        cur, ok = _step(cur, token)
        if not ok:
            return False
    return True


def validate_grounded_in(
    paths: list[str] | None,
    bundle: dict[str, Any],
) -> tuple[list[str], list[str]]:
    if not paths:
        return [], []
    valid: list[str] = []
    invalid: list[str] = []
    for p in paths:
        if not isinstance(p, str):
            invalid.append(repr(p))
            continue
        if _walk_path(bundle, p.strip()):
            valid.append(p.strip())
        else:
            invalid.append(p.strip())
    return valid, invalid


def clamp_adjustment(value: Any, *, cap: float = RERANK_ADJUSTMENT_CAP) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v > cap:
        return cap
    if v < -cap:
        return -cap
    return v
