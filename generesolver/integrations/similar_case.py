from __future__ import annotations

import logging
import os
from heapq import nlargest

import generesolver.config as config
from generesolver.integrations import case_corpus, hpo_db
from generesolver.integrations._disk_cache import disk_cached

logger = logging.getLogger(__name__)

_IC: dict[int, float] = {}


def _ic(int_id: int) -> float:
    v = _IC.get(int_id)
    if v is None:
        v = hpo_db.get_term_omim_ic(str(int_id))
        _IC[int_id] = v
    return v


def _profile(hpo_ints: frozenset[int] | set[int]) -> tuple[frozenset[int], float]:
    anc = hpo_db.expand_with_ancestors(set(hpo_ints))
    return frozenset(anc), sum(_ic(t) for t in anc)

_index: list[tuple[case_corpus.CorpusCase, frozenset[int], float]] | None = None


def _build_index() -> list[tuple[case_corpus.CorpusCase, frozenset[int], float]]:
    global _index
    if _index is None:
        idx = []
        for c in case_corpus.get_corpus():
            anc, icsum = _profile(c.hpo_ids)
            if icsum > 0:
                idx.append((c, anc, icsum))
        _index = idx
        logger.info("similar_case: indexed %d corpus cases", len(idx))
    return _index


@disk_cached("similar_case")
def _search(hpo_csv: str, exclude: str | None, top_cases: int,
            top_diseases: int, min_sim: float, dedup_sim: float) -> list[dict]:
    q_ints = {int(x) for x in hpo_csv.split(",") if x}
    if not q_ints:
        return []
    a_q, icsum_q = _profile(q_ints)
    if icsum_q <= 0:
        return []
    ex = exclude.strip().upper() if exclude else None

    scored: list[tuple[float, case_corpus.CorpusCase]] = []
    for case, a_c, icsum_c in _build_index():
        if ex and case.dataset.upper() == ex:
            continue
        if case.hpo_ids == q_ints:
            continue
        inter = a_q & a_c
        if not inter:
            continue
        ic_inter = sum(_ic(t) for t in inter)
        ic_union = icsum_q + icsum_c - ic_inter
        sim = ic_inter / ic_union if ic_union > 0 else 0.0
        if sim >= dedup_sim:
            continue
        if sim >= min_sim:
            scored.append((sim, case))

    if not scored:
        return []
    neighbours = nlargest(top_cases, scored, key=lambda t: t[0])

    votes: dict[str, float] = {}
    for sim, case in neighbours:
        for did in case.gold_disease_ids:
            votes[did] = votes.get(did, 0.0) + sim

    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[:top_diseases]
    return [
        {"disease_id": did, "name": did, "genes": [], "score": round(score, 4),
         "rank": i, "source": "similar_case"}
        for i, (did, score) in enumerate(ranked, 1)
    ]


def ranked_diseases(hpo_ids: list[str], top_n: int | None = None) -> list[dict]:
    if not (getattr(config, "SIMILAR_CASE_ENABLED", False)
            or getattr(config, "SIMILAR_CASE_RERANK", False)):
        return []
    ints = sorted({int(h.split(":")[1]) for h in (hpo_ids or []) if ":" in h and h.split(":")[1].isdigit()})
    if not ints:
        return []
    top_diseases = int(top_n or getattr(config, "SIMILAR_CASE_TOP_DISEASES", 15))
    mode = (os.environ.get("GENERESOLVER_SIMILAR_CASE_MODE")
            or getattr(config, "SIMILAR_CASE_MODE", "cohort")).strip().lower()
    if mode == "loo":
        exclude = None
    else:
        exclude = os.environ.get("GENERESOLVER_SIMILAR_CASE_EXCLUDE") or \
            getattr(config, "SIMILAR_CASE_EXCLUDE_DATASET", None)
    try:
        return _search(
            ",".join(str(i) for i in ints),
            exclude,
            int(getattr(config, "SIMILAR_CASE_TOP_CASES", 25)),
            top_diseases,
            float(getattr(config, "SIMILAR_CASE_MIN_SIM", 0.05)),
            float(getattr(config, "SIMILAR_CASE_DEDUP_SIM", 0.98)),
        )
    except Exception as exc:
        logger.warning("similar_case retrieval failed: %s", exc)
        return []
