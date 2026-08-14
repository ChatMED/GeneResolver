from __future__ import annotations

import logging

from generesolver import config
from generesolver.components import biochemical, literature
from generesolver.integrations import europepmc, hpo_a, hpo_lr, reactome_db
from generesolver.schemas.case import CaseState
from generesolver.schemas.report import ProvenanceItem

logger = logging.getLogger(__name__)

_N_SEEDS = 3
_MAX_CANDIDATES = 10
_MAX_PATHWAY_GENES = 8
_MAX_PATHWAYS = 10

_ULTRA_RARE_DISEASE_THRESHOLD = 2


def _specificity_ranked(state: CaseState) -> list[tuple[int, str, str]]:
    rev = hpo_a.get_hpo_to_diseases_map()
    scored: list[tuple[int, str, str]] = []
    for t in state.hpo_profile.terms:
        status = getattr(t, "status", None)
        if status is not None and status.value != "present":
            continue
        try:
            ti = int(str(t.hpo_id).split(":")[1])
        except (ValueError, IndexError):
            continue
        n = len(rev.get(ti) or ())
        scored.append((n if n > 0 else 10 ** 9, t.hpo_id, getattr(t, "label", "") or t.hpo_id))
    scored.sort(key=lambda x: x[0])
    return scored


def _literature(seed_terms: list[tuple[str, int]], state: CaseState) -> list[str]:
    if not seed_terms or config.LOCAL_MODE:
        labels = [t for t, _ in seed_terms]
        state.trace.add_retrieval(
            source="europepmc", query=" + ".join(labels), n_results=0, failed=bool(seed_terms),
            error="skipped under LOCAL_MODE" if config.LOCAL_MODE else "no specific term to search on",
        )
        return []

    titles: list[str] = []
    seen: set[str] = set()

    for term, disease_count in seed_terms:
        if disease_count <= _ULTRA_RARE_DISEASE_THRESHOLD:
            logger.info("class_retrieval: blocking online search for ultra-rare term %r "
                       "(~%d disease(s) <= threshold %d) -- quasi-identifier risk",
                       term, disease_count, _ULTRA_RARE_DISEASE_THRESHOLD)
            state.trace.add_retrieval(
                source="europepmc", query=f'"{term}"', n_results=0, failed=True,
                error=f"blocked: ultra-rare term (~{disease_count} disease(s)) -- quasi-identifier risk",
            )
            continue

        query = f'"{term}"'
        try:
            hits = europepmc.search(query, page_size=6)
            ep_titles = [h.get("title", "") for h in hits if h.get("title")]
            state.trace.add_retrieval(source="europepmc", query=query,
                                      n_results=len(ep_titles), results=ep_titles[:6])
            for title in ep_titles:
                if title not in seen:
                    seen.add(title)
                    titles.append(title)
        except Exception as exc:
            state.trace.add_retrieval(source="europepmc", query=query, failed=True, error=str(exc))

        try:
            pmids = literature.pubmed_search(query)
            pm_titles = literature.pubmed_titles(tuple(pmids[:6]), max_n=6) if pmids else []
            state.trace.add_retrieval(source="pubmed", query=query,
                                      n_results=len(pm_titles), results=pm_titles[:6])
            for title in pm_titles:
                if title not in seen:
                    seen.add(title)
                    titles.append(title)
        except Exception as exc:
            state.trace.add_retrieval(source="pubmed", query=query, failed=True, error=str(exc))

    return titles[:12]


def _search_mechanism_facts(query: str, max_results: int = 5) -> list[str]:
    if not query or config.LOCAL_MODE:
        return []

    titles: list[str] = []
    seen: set[str] = set()

    try:
        hits = europepmc.search(query, page_size=max_results)
        for h in hits:
            title = h.get("title", "")
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
    except Exception as exc:
        logger.warning("class_retrieval: europepmc mechanism search failed for %r: %s", query, exc)

    if len(titles) < max_results:
        try:
            pmids = literature.pubmed_search(query)
            pm_titles = literature.pubmed_titles(tuple(pmids[:max_results]), max_n=max_results) if pmids else []
            for title in pm_titles:
                if title and title not in seen:
                    seen.add(title)
                    titles.append(title)
        except Exception as exc:
            logger.warning("class_retrieval: pubmed mechanism search failed for %r: %s", query, exc)

    return titles[:max_results]


def _pathways_for_diseases(orpha_ids: list[str], state: CaseState) -> list[str]:
    genes: list[str] = []
    seen_genes: set[str] = set()
    for orpha_id in orpha_ids:
        if not orpha_id or len(genes) >= _MAX_PATHWAY_GENES:
            break
        try:
            for g in sorted(hpo_lr.genes_for_disease(orpha_id) or ()):
                if g not in seen_genes:
                    seen_genes.add(g)
                    genes.append(g)
        except Exception as exc:
            logger.warning("gene lookup failed for %s: %s", orpha_id, exc)

    pathways: list[str] = []
    seen_pathways: set[str] = set()
    for gene in genes[:_MAX_PATHWAY_GENES]:
        try:
            for p in sorted(reactome_db.get_pathways_for_gene(gene) or ()):
                if p not in seen_pathways:
                    seen_pathways.add(p)
                    pathways.append(f"{p} ({gene})")
        except Exception as exc:
            logger.warning("Reactome pathway lookup failed for %s: %s", gene, exc)
        if len(pathways) >= _MAX_PATHWAYS:
            break

    pathways = pathways[:_MAX_PATHWAYS]
    state.trace.add_retrieval(source="reactome", query="+".join(genes[:_MAX_PATHWAY_GENES]),
                              n_results=len(pathways), results=pathways[:8])
    return pathways


def run(state: CaseState) -> CaseState:
    present = state.hpo_profile.present_ids
    ranked = _specificity_ranked(state)

    specific = [(n, hid, lab) for n, hid, lab in ranked if 0 < n < 10 ** 9][:_N_SEEDS]
    seed_terms = [lab for _, _, lab in specific] or [lab for _, _, lab in ranked[:_N_SEEDS]]
    seed_pairs = [(lab, n) for n, _, lab in specific] or [(lab, 0) for _, _, lab in ranked[:_N_SEEDS]]
    seed_note = ", ".join(f"{lab} (~{n} conditions)" for n, hid, lab in ranked[:_N_SEEDS])

    candidate_diseases: list[str] = []
    candidate_orpha_ids: list[str] = []

    def _try_candidates(min_overlap: int) -> None:
        for c in hpo_a.candidate_diseases_for_hpo_profile(present, min_overlap=min_overlap,
                                                          top_n=_MAX_CANDIDATES):
            if c.label and c.label not in candidate_diseases:
                candidate_diseases.append(c.label)
                if getattr(c, "orpha_id", None):
                    candidate_orpha_ids.append(c.orpha_id)

    try:
        _try_candidates(min_overlap=2)
        if not candidate_diseases:
            _try_candidates(min_overlap=1)
        state.trace.add_retrieval(source="hpo_a", query="+".join(seed_terms),
                                  n_results=len(candidate_diseases),
                                  results=candidate_diseases[:8])
    except Exception as exc:
        logger.warning("HPO-A candidate lookup failed: %s", exc)

    biochem_lines: list[str] = []
    try:
        for d in biochemical.marker_diseases(present)[:_MAX_CANDIDATES]:
            biochem_lines.append(
                f"{d['name']}: characterised by {', '.join(m['label'] for m in d['markers'][:3])}")
            if d["name"] not in candidate_diseases:
                candidate_diseases.append(d["name"])
    except Exception as exc:
        logger.warning("biochemical marker retrieval failed: %s", exc)

    lit_titles = _literature(seed_pairs, state)
    pathways = _pathways_for_diseases(candidate_orpha_ids, state)

    state.diagnostics["class_evidence"] = {
        "seed_terms": seed_terms,
        "candidate_diseases": candidate_diseases[:_MAX_CANDIDATES],
        "biochemical": biochem_lines,
        "literature": lit_titles,
        "pathways": pathways,
        "sources": state.trace.sources_used(),
    }

    logger.info("class_retrieval: seeded on [%s]; %d candidate disease(s)",
                seed_note, len(candidate_diseases))
    state.trace.add_step(
        step="class_retrieval",
        claim=(f"Searched on the most specific findings [{seed_note}]; "
               f"candidate diagnoses: {', '.join(candidate_diseases[:5]) or 'none found'}"),
        grounded_in=state.trace.sources_used() or ["hpo_a (local index)"],
    )
    state.provenance.append(ProvenanceItem(
        component="class_retrieval",
        tool_name=("specificity-seeded phenotype->disease retrieval "
                  "(local HPO-A + biochemical + Reactome index; EuropePMC + PubMed literature)"),
        tool_version=f"seeds={len(seed_terms)} least-connected terms; candidate diseases, not genes",
    ))
    state.mark_done("class_retrieval")
    return state
