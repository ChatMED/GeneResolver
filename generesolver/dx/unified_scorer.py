from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from generesolver.integrations import hpo_lr, hpo_a, hpo_db

logger = logging.getLogger(__name__)

_MISS_FLOOR = 0.01
_MISS_SHRINK = 0.5
_ABSENT_FLOOR = 0.02
_BACKGROUND_FLOOR = 1e-4

_GENE_HOP_BONUS = 1.5


@dataclass
class ScoredDisease:
    disease_id: str
    name: str
    genes: list[str]
    log_lr: float
    matched_terms: list[str]
    contradicted_terms: list[str]
    gene_hop_genes: list[str]
    n_direct_matches: int
    source: list[str] = field(default_factory=list)


def _to_int(hpo_id: str) -> int | None:
    try:
        return int(str(hpo_id).split(":")[1])
    except (ValueError, IndexError):
        return None


def _label(hpo_id: str) -> str:
    ti = _to_int(hpo_id)
    if ti is None:
        return hpo_id
    try:
        return str(hpo_db._get_ontology().get_hpo_object(ti).name)
    except Exception:
        return hpo_id


def _gene_hop_map() -> dict[int, set[str]]:
    fwd = hpo_a.get_gene_hpo_map()
    rev: dict[int, set[str]] = {}
    for gene, hpo_ints in fwd.items():
        for hi in hpo_ints:
            rev.setdefault(hi, set()).add(gene)
    return rev


def score_diseases(
    present_ids: list[str],
    absent_ids: list[str] | None = None,
    *,
    top_n: int = 40,
) -> list[ScoredDisease]:
    present_ids = [h for h in (present_ids or []) if h]
    absent_ids = [h for h in (absent_ids or []) if h]
    if not present_ids:
        return []

    present_ints = {i for i in (_to_int(h) for h in present_ids) if i is not None}
    absent_ints = {i for i in (_to_int(h) for h in absent_ids) if i is not None}
    int_to_present_label = {i: _label(f"HP:{i:07d}") for i in present_ints}
    int_to_absent_label = {i: _label(f"HP:{i:07d}") for i in absent_ints}

    freqs_by_disease = hpo_lr.disease_hpo_frequencies()
    excluded_by_disease = hpo_lr.disease_excluded_terms()
    background = hpo_lr.background_frequency()
    names = hpo_lr.disease_names()
    d2g, _ = hpo_lr._load_disease_gene_maps()
    gene_hop = _gene_hop_map()

    implicated_genes: set[str] = set()
    for pi in present_ints:
        implicated_genes |= gene_hop.get(pi, set())

    def bg(term: int) -> float:
        return max(_BACKGROUND_FLOOR, background.get(term, _BACKGROUND_FLOOR))

    scored: list[ScoredDisease] = []
    for disease_id, freq_map in freqs_by_disease.items():
        log_lr = 0.0
        matched_terms: list[str] = []
        n_direct = 0

        for pi in present_ints:
            p_disease = freq_map.get(pi)
            if p_disease is not None:
                lr = p_disease / bg(pi)
                log_lr += math.log(lr)
                matched_terms.append(int_to_present_label[pi])
                n_direct += 1
            else:
                miss_p = min(_MISS_FLOOR, _MISS_SHRINK * bg(pi))
                log_lr += math.log(miss_p / bg(pi))

        contradicted: list[str] = []
        excluded = excluded_by_disease.get(disease_id, set())
        for ai in absent_ints:
            if ai in excluded:
                continue
            p_disease = freq_map.get(ai)
            if p_disease is None:
                continue
            p_absent_given_d = max(_ABSENT_FLOOR, 1.0 - p_disease)
            p_absent_given_not_d = max(_ABSENT_FLOOR, 1.0 - bg(ai))
            log_lr += math.log(p_absent_given_d / p_absent_given_not_d)
            if p_disease >= 0.5:
                contradicted.append(int_to_absent_label[ai])

        genes = sorted(d2g.get(disease_id, set()))
        gene_hop_only_genes = sorted(implicated_genes & set(genes)) if not matched_terms else []
        if gene_hop_only_genes and n_direct == 0:
            log_lr += _GENE_HOP_BONUS

        if n_direct == 0 and not gene_hop_only_genes:
            continue

        source = []
        if n_direct > 0:
            source.append("phenotype_lr")
        if gene_hop_only_genes:
            source.append("gene_hop")

        scored.append(ScoredDisease(
            disease_id=disease_id,
            name=names.get(disease_id, disease_id),
            genes=genes,
            log_lr=round(log_lr, 4),
            matched_terms=matched_terms,
            contradicted_terms=contradicted,
            gene_hop_genes=gene_hop_only_genes,
            n_direct_matches=n_direct,
            source=source,
        ))

    scored.sort(key=lambda s: (-s.log_lr, s.disease_id))
    logger.info("unified_scorer: scored %d disease(s) with any evidence, returning top %d",
                len(scored), min(top_n, len(scored)))
    return scored[:top_n]


def as_candidate_dicts(scored: list[ScoredDisease]) -> list[dict]:
    out = []
    for s in scored:
        biochem_findings = []
        if s.gene_hop_genes:
            biochem_findings.append({
                "finding": f"INDIRECT gene-linkage only (no direct phenotype match): "
                           f"causative gene(s) {', '.join(s.gene_hop_genes)} implicated by "
                           f"patient's findings via HPO-A gene annotation. Weaker, exploratory "
                           f"support — verify plausibility, don't rank on this alone.",
                "characteristic_of_n_diseases": None,
            })
        if s.contradicted_terms:
            biochem_findings.append({
                "finding": f"CONTRADICTED: patient lacks {', '.join(s.contradicted_terms)}, "
                           f"which this disease commonly causes — already penalised in the "
                           f"score, but flagged for the reasoner's judgement too.",
                "characteristic_of_n_diseases": None,
            })
        out.append({
            "disease_id": s.disease_id,
            "name": s.name,
            "genes": s.genes,
            "matched_terms": s.matched_terms,
            "biochem_findings": biochem_findings,
            "best_specificity": None,
            "n_matched": s.n_direct_matches,
            "log_lr": s.log_lr,
            "source": s.source,
        })
    return out