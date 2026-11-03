from __future__ import annotations

import logging

from generesolver.schemas.case import CaseState
from generesolver.schemas.clinical import DiseaseStatus
from generesolver.schemas.gene import GeneRank
from generesolver.integrations import hpo_a, hpo_db, orphanet, open_targets_api, lr_score
from generesolver.config import (
    LR_RANKER_ENABLED,
    GENE_CANDIDATE_CAP,
    GENE_SCORING_POOL_K,
    GENE_OUTPUT_TOP_K,
    DISEASE_OUTPUT_TOP_K,
    OT_RECALL_SEED_MAX_GENES,
    OT_RECALL_SEED_MIN_SCORE,
    OT_RECALL_SEED_FETCH_SIZE,
    OT_RECALL_SEED_BREADTH_SCORE,
    OT_RECALL_SEED_MAX_DISEASE_BREADTH,
    SHORTLIST_RECALL_SEED_TOP_K,
    SHORTLIST_RECALL_SEED_MAX_GENES,
    PHENOTYPE_DISEASE_SEED_TOP_K,
    PHENOTYPE_DISEASE_SEED_MIN_OVERLAP,
    PHENOTYPE_DISEASE_SEED_MAX_GENES,
    PHENOTYPE_DISEASE_SEED_MAX_TOTAL_GENES,
    CLINVAR_TRIAGE_ENABLED,
)

logger = logging.getLogger(__name__)

_TRIAGE_SCORE_CEILING = 0.75

_CLINVAR_TRIAGE_SCORE = 0.95
_CLINVAR_PATHOGENIC_TERMS = ("pathogenic",)


def _is_clinvar_pathogenic(clinvar_significance: str | None) -> bool:
    if not clinvar_significance:
        return False
    sig = clinvar_significance.lower()
    if "conflicting" in sig or "uncertain" in sig or "benign" in sig:
        return False
    return any(term in sig for term in _CLINVAR_PATHOGENIC_TERMS)


def _variant_triage_score(impact, gnomad_af, clinvar_significance=None) -> float:
    if CLINVAR_TRIAGE_ENABLED and _is_clinvar_pathogenic(clinvar_significance):
        return _CLINVAR_TRIAGE_SCORE
    impact_w = {"HIGH": 0.85, "MODERATE": 0.65, "LOW": 0.45, "MODIFIER": 0.30}.get(impact or "", 0.50)
    af = gnomad_af if gnomad_af is not None else 0.0
    rarity_w = 1.0 if af < 1e-4 else 0.9 if af < 1e-3 else 0.75 if af < 1e-2 else 0.55
    return round(min(_TRIAGE_SCORE_CEILING, impact_w * rarity_w), 4)


_CANDIDATE_RARITY_CEILING = 0.01


def _triage_af(v):
    # Original profile: overall gnomAD AF (reported Eq. 2). Revised: the filter frequency.
    from generesolver import config as _cfg
    return v.filter_af if _cfg.VARIANT_FILTER_PROFILE == "revised" else v.gnomad_af


def _has_plausible_variant_evidence(gene_variants: list) -> bool:
    from generesolver import config as _cfg
    if _cfg.VARIANT_FILTER_PROFILE == "revised":
        from generesolver.components.variant_annotator import is_relevant_variant, passes_rarity
        return any(
            is_relevant_variant(v.consequence, v.impact, is_splice_relevant=v.is_splice_relevant,
                                spliceai=v.spliceai_score, profile="revised")
            and passes_rarity(v.filter_af, v.filter_af_status or v.gnomad_frequency_status)
            for v in gene_variants
        )
    # Original (reported) gate, unchanged.
    return any(
        v.impact in ("HIGH", "MODERATE")
        and (v.gnomad_af is None or v.gnomad_af < _CANDIDATE_RARITY_CEILING)
        for v in gene_variants
    )


def run(state: CaseState) -> CaseState:
    patient_hpo_ids = state.hpo_profile.present_ids
    disease_seed_strengths, disease_seed_origins = _build_disease_seed(state)
    _merge_seed(
        disease_seed_strengths, disease_seed_origins, *_build_ot_recall_seed(state)
    )

    _merge_seed(
        disease_seed_strengths, disease_seed_origins,
        *_build_shortlist_recall_seed(state),
    )

    pheno_disease_cands = _phenotype_disease_candidates(state)
    _merge_seed(
        disease_seed_strengths, disease_seed_origins,
        *_seed_from_pheno_diseases(pheno_disease_cands, len(patient_hpo_ids)),
    )

    has_variant_evidence = bool(
        state.annotated_variants or state.input.candidate_gene
    )

    if has_variant_evidence:
        variants_by_gene_for_filter: dict[str, list] = {}
        for v in state.annotated_variants:
            if v.gene:
                variants_by_gene_for_filter.setdefault(v.gene, []).append(v)
        genes: set[str] = {
            gene for gene, gvs in variants_by_gene_for_filter.items()
            if _has_plausible_variant_evidence(gvs)
        }
        if state.input.candidate_gene:
            genes.add(state.input.candidate_gene)
        if disease_seed_strengths:
            genes.update(disease_seed_strengths.keys())
        state.ranked_genes = _score_variant_genes(
            genes, state, patient_hpo_ids,
            disease_seed_strengths, disease_seed_origins,
        )
    elif patient_hpo_ids or disease_seed_strengths:
        if LR_RANKER_ENABLED and patient_hpo_ids and lr_score.is_available():
            state.ranked_genes = _score_lr_candidates(
                patient_hpo_ids, state.hpo_profile.absent_ids,
                disease_seed_strengths, disease_seed_origins,
            )
            try:
                ranked = lr_score.rank_diseases(
                    patient_hpo_ids, state.hpo_profile.absent_ids,
                    top_n=DISEASE_OUTPUT_TOP_K,
                )
                state.ranked_diseases = _augment_diseases_with_overlap(
                    ranked, pheno_disease_cands,
                    patient_hpo_ids, state.hpo_profile.absent_ids,
                )
            except Exception as exc:
                logger.warning("disease ranking failed: %s", exc)
        else:
            state.ranked_genes = _score_hpo_candidates(
                patient_hpo_ids, disease_seed_strengths, disease_seed_origins,
            )
    else:
        logger.info("gene_aggregator: no genes and no HPO terms — nothing to aggregate")
    cap = GENE_CANDIDATE_CAP
    if cap and len(state.ranked_genes) > cap:
        by_pheno = sorted(state.ranked_genes, key=lambda g: g.score_phenotype, reverse=True)
        by_var = sorted(state.ranked_genes, key=lambda g: g.score_variant, reverse=True)
        keep, seen = [], set()
        for g in by_pheno[:cap] + by_var[: max(cap // 3, 1)]:
            if g.gene not in seen:
                seen.add(g.gene)
                keep.append(g)
        if state.input.candidate_gene:
            for g in state.ranked_genes:
                if g.gene == state.input.candidate_gene and g.gene not in seen:
                    seen.add(g.gene)
                    keep.append(g)
        logger.info("gene_aggregator: capped candidates %d -> %d (top phenotype + variant)",
                    len(state.ranked_genes), len(keep))
        state.ranked_genes = keep

    state.mark_done("gene_aggregator")
    return state


_UNCERTAIN_STRENGTH_MULTIPLIER = 0.5


def _build_disease_seed(
    state: CaseState,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    seed_strength: dict[str, float] = {}
    seed_origins: dict[str, list[str]] = {}

    if not state.disease_mentions:
        return seed_strength, seed_origins

    for mention in state.disease_mentions:
        if mention.status not in (DiseaseStatus.PRESENT, DiseaseStatus.UNCERTAIN):
            continue
        if not mention.orpha_id:
            continue
        weight = (
            1.0
            if mention.status == DiseaseStatus.PRESENT
            else _UNCERTAIN_STRENGTH_MULTIPLIER
        )
        for ga in orphanet.get_genes_for_disease(mention.orpha_id):
            sym = ga.gene_symbol
            strength = ga.strength * weight
            if strength <= 0.0:
                continue
            if strength > seed_strength.get(sym, 0.0):
                seed_strength[sym] = strength
            origins = seed_origins.setdefault(sym, [])
            if mention.orpha_id not in origins:
                origins.append(mention.orpha_id)

    if seed_strength:
        logger.info(
            "gene_aggregator: disease-mention seeding contributed %d gene(s) "
            "from %d ORPHA-normalized mention(s)",
            len(seed_strength),
            sum(1 for m in state.disease_mentions
                if m.status == DiseaseStatus.PRESENT and m.orpha_id),
        )

    return seed_strength, seed_origins


def _merge_seed(
    base_strength: dict[str, float],
    base_origins: dict[str, list[str]],
    add_strength: dict[str, float],
    add_origins: dict[str, list[str]],
) -> None:
    for sym, strength in add_strength.items():
        if strength > base_strength.get(sym, 0.0):
            base_strength[sym] = strength
        origins = base_origins.setdefault(sym, [])
        for origin in add_origins.get(sym, []):
            if origin not in origins:
                origins.append(origin)


def _build_ot_recall_seed(
    state: CaseState,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    seed_strength: dict[str, float] = {}
    seed_origins: dict[str, list[str]] = {}

    if OT_RECALL_SEED_MAX_GENES <= 0 or not state.disease_mentions:
        return seed_strength, seed_origins

    scored: list[tuple[str, float, str]] = []
    for mention in state.disease_mentions:
        if mention.status not in (DiseaseStatus.PRESENT, DiseaseStatus.UNCERTAIN):
            continue
        query = mention.mondo_id or mention.orpha_id or mention.label or mention.text
        if not query:
            continue
        weight = (
            1.0
            if mention.status == DiseaseStatus.PRESENT
            else _UNCERTAIN_STRENGTH_MULTIPLIER
        )
        origin = mention.label or mention.text or query
        try:
            targets = open_targets_api.get_disease_associated_targets(
                query, size=OT_RECALL_SEED_FETCH_SIZE
            )
        except Exception as exc:
            logger.warning("OT disease→targets failed for %s: %s", query, exc)
            continue
        breadth = sum(
            1 for t in targets
            if float(t.get("score", 0.0)) >= OT_RECALL_SEED_BREADTH_SCORE
        )
        if breadth > OT_RECALL_SEED_MAX_DISEASE_BREADTH:
            logger.info(
                "gene_aggregator: skipping broad disease '%s' for OT recall "
                "seeding (%d high-confidence targets > %d)",
                origin, breadth, OT_RECALL_SEED_MAX_DISEASE_BREADTH,
            )
            continue

        for t in targets:
            sym = t.get("gene_symbol")
            strength = float(t.get("score", 0.0)) * weight
            if not sym or strength < OT_RECALL_SEED_MIN_SCORE:
                continue
            scored.append((sym, strength, origin))

    scored.sort(key=lambda x: x[1], reverse=True)
    for sym, strength, origin in scored[:OT_RECALL_SEED_MAX_GENES]:
        if strength > seed_strength.get(sym, 0.0):
            seed_strength[sym] = strength
        origins = seed_origins.setdefault(sym, [])
        if origin not in origins:
            origins.append(origin)

    if seed_strength:
        logger.info(
            "gene_aggregator: Open Targets recall seeding contributed %d gene(s) "
            "from %d disease mention(s)",
            len(seed_strength),
            sum(1 for m in state.disease_mentions
                if m.status in (DiseaseStatus.PRESENT, DiseaseStatus.UNCERTAIN)),
        )

    return seed_strength, seed_origins


def _build_shortlist_recall_seed(
    state: CaseState,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    seed_strength: dict[str, float] = {}
    seed_origins: dict[str, list[str]] = {}

    if SHORTLIST_RECALL_SEED_TOP_K <= 0:
        return seed_strength, seed_origins

    diag = (getattr(state, "diagnostics", {}) or {}).get("synthesizer", {})
    shortlist = diag.get("shortlist", []) or []
    if not shortlist:
        return seed_strength, seed_origins
    n_present = diag.get("n_present_hpo") or 0

    seeded_diseases = 0
    for r in shortlist:
        if seeded_diseases >= SHORTLIST_RECALL_SEED_TOP_K:
            break
        genes = r.get("genes", []) or []
        if not genes or len(genes) > SHORTLIST_RECALL_SEED_MAX_GENES:
            continue
        overlap = r.get("overlap", 0) or 0
        strength = min(1.0, overlap / n_present) if n_present > 0 else 0.5
        if strength <= 0.0:
            continue
        origin = r.get("orpha_id") or r.get("label") or "shortlist"
        for sym in genes:
            if strength > seed_strength.get(sym, 0.0):
                seed_strength[sym] = strength
            origins = seed_origins.setdefault(sym, [])
            if origin not in origins:
                origins.append(origin)
        seeded_diseases += 1

    if seed_strength:
        logger.info(
            "gene_aggregator: HPO-shortlist recall seeding contributed %d gene(s) "
            "from %d specific disease(s)",
            len(seed_strength), seeded_diseases,
        )
    return seed_strength, seed_origins


def _phenotype_disease_candidates(state: CaseState):

    if PHENOTYPE_DISEASE_SEED_TOP_K <= 0:
        return []
    if state.disease_mentions:
        return []
    synth = (getattr(state, "diagnostics", {}) or {}).get("synthesizer", {}) or {}
    if synth.get("shortlist"):
        return []
    present = state.hpo_profile.present_ids
    if not present:
        return []
    try:
        return hpo_a.candidate_diseases_for_hpo_profile(
            present,
            min_overlap=PHENOTYPE_DISEASE_SEED_MIN_OVERLAP,
            top_n=PHENOTYPE_DISEASE_SEED_TOP_K,
        )
    except Exception as exc:
        logger.warning("phenotype disease candidates failed: %s", exc)
        return []


def _seed_from_pheno_diseases(
    cands, n_present: int,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    seed_strength: dict[str, float] = {}
    seed_origins: dict[str, list[str]] = {}
    scored: list[tuple[float, str, str]] = []
    for c in cands or []:
        try:
            genes = {ga.gene_symbol for ga in orphanet.get_genes_for_disease(c.orpha_id)}
        except Exception:
            genes = set()
        if not genes or len(genes) > PHENOTYPE_DISEASE_SEED_MAX_GENES:
            continue
        strength = min(1.0, c.overlap / n_present) if n_present > 0 else 0.5
        if strength <= 0.0:
            continue
        for sym in genes:
            scored.append((strength, sym, c.orpha_id))

    scored.sort(key=lambda t: t[0], reverse=True)
    for strength, sym, origin in scored:
        if (sym not in seed_strength
                and len(seed_strength) >= PHENOTYPE_DISEASE_SEED_MAX_TOTAL_GENES):
            continue
        if strength > seed_strength.get(sym, 0.0):
            seed_strength[sym] = strength
        origins = seed_origins.setdefault(sym, [])
        if origin not in origins:
            origins.append(origin)
    if seed_strength:
        logger.info(
            "gene_aggregator: phenotype-overlap disease seeding contributed %d "
            "gene(s) from %d candidate disease(s)",
            len(seed_strength), len(cands or []),
        )
    return seed_strength, seed_origins


def _augment_diseases_with_overlap(
    ranked: list[dict], cands, present_ids: list[str], absent_ids: list[str],
) -> list[dict]:
    if not cands:
        return ranked
    existing = {d.get("disease_id") for d in ranked if isinstance(d, dict)}
    try:
        score_map = dict(lr_score.score_diseases(present_ids, absent_ids))
    except Exception:
        score_map = {}
    appended = 0
    for c in cands:
        if c.orpha_id in existing:
            continue
        try:
            genes = sorted({ga.gene_symbol for ga in orphanet.get_genes_for_disease(c.orpha_id)})
        except Exception:
            genes = []
        ranked.append({
            "disease_id": c.orpha_id,
            "name": c.label,
            "log_lr": round(score_map.get(c.orpha_id, 0.0), 3),
            "posttest_prob": 0.0,
            "genes": genes,
            "supporting_terms": [],
            "hpo_overlap": c.overlap,
        })
        existing.add(c.orpha_id)
        appended += 1
    if appended:
        logger.info(
            "gene_aggregator: union-ed %d phenotype-overlap disease(s) into "
            "ranked_diseases (LR pool=%d)", appended, len(ranked) - appended,
        )
    return ranked


def _score_variant_genes(
    genes: set[str],
    state: CaseState,
    patient_hpo_ids: list[str],
    disease_seed_strengths: dict[str, float],
    disease_seed_origins: dict[str, list[str]],
) -> list[GeneRank]:
    patient_set = (
        hpo_db.build_patient_hpo_set(patient_hpo_ids) if patient_hpo_ids else None
    )
    lr_pheno: dict[str, float] = {}
    if patient_hpo_ids and lr_score.is_available():
        try:
            for g, loglr, _best in lr_score.score_genes(
                patient_hpo_ids, state.hpo_profile.absent_ids, top_n=20000
            ):
                lr_pheno[g] = loglr
        except Exception as exc:
            logger.warning("LR phenotype scoring failed in variant mode: %s", exc)
    _use_lr = bool(lr_pheno)

    variants_by_gene: dict[str, list] = {}
    for v in state.annotated_variants:
        if v.gene:
            variants_by_gene.setdefault(v.gene, []).append(v)

    gene_ranks: list[GeneRank] = []
    for gene in sorted(genes):
        gene_variants = variants_by_gene.get(gene, [])
        if gene_variants:
            best = max(
                gene_variants,
                key=lambda v: _variant_triage_score(v.impact, _triage_af(v), v.clinvar_significance),
            )
            score_variant = _variant_triage_score(best.impact, _triage_af(best), best.clinvar_significance)
            best_variant_id = best.variant_id
        else:
            score_variant = 0.0
            best_variant_id = None

        if _use_lr:
            score_phenotype = lr_pheno.get(gene, 0.0)
        else:
            try:
                score_phenotype = (
                    hpo_db.compute_similarity_with_set(patient_set, gene)
                    if patient_set is not None
                    else 0.0
                )
            except Exception as exc:
                logger.warning("HPO similarity failed for %s: %s", gene, exc)
                score_phenotype = 0.0

        gene_ranks.append(
            GeneRank(
                rank=0,
                gene=gene,
                final_score=0.0,
                confidence="insufficient",
                score_phenotype=score_phenotype,
                score_variant=score_variant,
                score_disease_match=disease_seed_strengths.get(gene, 0.0),
                seeded_by_diseases=list(disease_seed_origins.get(gene, [])),
                best_variant_id=best_variant_id,
            )
        )
    return gene_ranks


def _score_lr_candidates(
    present_ids: list[str],
    absent_ids: list[str],
    disease_seed_strengths: dict[str, float],
    disease_seed_origins: dict[str, list[str]],
) -> list[GeneRank]:
    scored = lr_score.score_genes(present_ids, absent_ids, top_n=GENE_OUTPUT_TOP_K * 4)

    gene_ranks: list[GeneRank] = []
    seen: set[str] = set()
    for gene, loglr, _best_disease in scored:
        seen.add(gene)
        gene_ranks.append(
            GeneRank(
                rank=0,
                gene=gene,
                final_score=0.0,
                confidence="insufficient",
                score_phenotype=loglr,
                score_variant=0.0,
                score_disease_match=disease_seed_strengths.get(gene, 0.0),
                seeded_by_diseases=list(disease_seed_origins.get(gene, [])),
            )
        )

    for gene, strength in disease_seed_strengths.items():
        if gene not in seen:
            gene_ranks.append(
                GeneRank(
                    rank=0,
                    gene=gene,
                    final_score=0.0,
                    confidence="insufficient",
                    score_phenotype=0.0,
                    score_variant=0.0,
                    score_disease_match=strength,
                    seeded_by_diseases=list(disease_seed_origins.get(gene, [])),
                )
            )

    gene_ranks.sort(
        key=lambda g: (g.score_phenotype, g.score_disease_match), reverse=True
    )
    truncated = gene_ranks[:GENE_OUTPUT_TOP_K]
    kept = {g.gene for g in truncated}
    for g in gene_ranks[GENE_OUTPUT_TOP_K:]:
        if g.score_disease_match > 0.0 and g.gene not in kept:
            truncated.append(g)
            kept.add(g.gene)
    return truncated


def _score_hpo_candidates(
    patient_hpo_ids: list[str],
    disease_seed_strengths: dict[str, float],
    disease_seed_origins: dict[str, list[str]],
) -> list[GeneRank]:
    patient_int_ids: set[int] = set()
    for hpo_id in patient_hpo_ids:
        try:
            patient_int_ids.add(int(hpo_id.split(":")[1]))
        except (ValueError, IndexError):
            pass

    if not patient_int_ids and not disease_seed_strengths:
        return []

    if patient_int_ids:
        expanded_patient_ids = hpo_db.expand_with_ancestors(patient_int_ids)
        gene_index = hpo_db.get_gene_index()
        hpoa_map = hpo_a.get_gene_hpo_map()
        ic_weight: dict[int, float] = {
            tid: hpo_db.get_term_omim_ic(str(tid)) for tid in expanded_patient_ids
        }

        candidate_symbols = set(gene_index.keys()) | set(hpoa_map.keys())
        overlapping: list[tuple[str, float, int]] = []
        for symbol in candidate_symbols:
            pyhpo_set: set[int] = set(getattr(gene_index.get(symbol), "hpo", set()))
            merged_set = pyhpo_set | hpoa_map.get(symbol, set())
            shared = merged_set & expanded_patient_ids
            if not shared:
                continue
            ic_score = sum(ic_weight.get(tid, 0.0) for tid in shared)
            overlapping.append((symbol, ic_score, len(shared)))
        overlapping.sort(key=lambda x: (x[1], x[2]), reverse=True)
        top_symbols = [sym for sym, _, _ in overlapping[:GENE_SCORING_POOL_K]]
    else:
        top_symbols = []

    candidate_set: set[str] = set(top_symbols) | set(disease_seed_strengths.keys())
    patient_set = (
        hpo_db.build_patient_hpo_set(patient_hpo_ids) if patient_hpo_ids else None
    )

    gene_ranks: list[GeneRank] = []
    for gene in sorted(candidate_set):
        try:
            score = (
                hpo_db.compute_similarity_with_set(patient_set, gene)
                if patient_set is not None else 0.0
            )
        except Exception as exc:
            logger.warning("HPO similarity failed for %s: %s", gene, exc)
            score = 0.0
        disease_strength = disease_seed_strengths.get(gene, 0.0)
        if score > 0.0 or disease_strength > 0.0:
            gene_ranks.append(
                GeneRank(
                    rank=0,
                    gene=gene,
                    final_score=0.0,
                    confidence="insufficient",
                    score_phenotype=score,
                    score_variant=0.0,
                    score_disease_match=disease_strength,
                    seeded_by_diseases=list(disease_seed_origins.get(gene, [])),
                )
            )
    gene_ranks.sort(
        key=lambda g: (g.score_phenotype, g.score_disease_match),
        reverse=True,
    )
    truncated = gene_ranks[:GENE_OUTPUT_TOP_K]
    truncated_symbols = {g.gene for g in truncated}
    for g in gene_ranks[GENE_OUTPUT_TOP_K:]:
        if g.score_disease_match > 0.0 and g.gene not in truncated_symbols:
            truncated.append(g)
            truncated_symbols.add(g.gene)

    return truncated
