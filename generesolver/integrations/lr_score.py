from __future__ import annotations

import logging
import math
from functools import lru_cache

from generesolver.integrations import hpo_lr, hpo_db, orphanet

logger = logging.getLogger(__name__)

_NO_MATCH_LR = 0.2
_EXCLUDED_OBSERVED_LR = 0.001
_NEGATED_EXCLUDED_LR = 1000.0
_CHILD_PENALTY = 0.5
_ABSENT_FLOOR = 0.05


def _ont():
    return hpo_db._get_ontology()


@lru_cache(maxsize=20000)
def _ancestors(term_int: int) -> frozenset[int]:
    try:
        term = _ont().get_hpo_object(int(term_int))
    except Exception:
        return frozenset()
    out: set[int] = set()
    for p in (getattr(term, "all_parents", None) or []):
        pid = getattr(p, "index", None)
        if isinstance(pid, int):
            out.add(pid)
    return frozenset(out)


@lru_cache(maxsize=1)
def _disease_closure() -> dict[str, dict[int, float]]:
    raw = hpo_lr.disease_hpo_frequencies()
    closure: dict[str, dict[int, float]] = {}
    for disease_id, term_map in raw.items():
        acc: dict[int, float] = {}
        for term_int, freq in term_map.items():
            if freq > acc.get(term_int, 0.0):
                acc[term_int] = freq
            for anc in _ancestors(term_int):
                if freq > acc.get(anc, 0.0):
                    acc[anc] = freq
        closure[disease_id] = acc
    logger.info("lr_score: built annotation closure for %d diseases", len(closure))
    return closure


@lru_cache(maxsize=1)
def _bg_closure() -> dict[int, float]:
    closure = _disease_closure()
    n = max(1, len(closure))
    totals: dict[int, float] = {}
    for acc in closure.values():
        for term_int, freq in acc.items():
            totals[term_int] = totals.get(term_int, 0.0) + freq
    floor = 1.0 / (2.0 * n)
    return {t: max(floor, s / n) for t, s in totals.items()}


@lru_cache(maxsize=1)
def _inverted_closure() -> dict[int, set[str]]:
    inv: dict[int, set[str]] = {}
    for disease_id, acc in _disease_closure().items():
        for term_int in acc:
            inv.setdefault(term_int, set()).add(disease_id)
    return inv


@lru_cache(maxsize=1)
def _inverted_raw() -> dict[int, set[str]]:
    inv: dict[int, set[str]] = {}
    for disease_id, term_map in hpo_lr.disease_hpo_frequencies().items():
        for term_int in term_map:
            inv.setdefault(term_int, set()).add(disease_id)
    return inv


def _candidate_diseases(present_ints: set[int]) -> set[str]:
    inv_closure = _inverted_closure()
    inv_raw = _inverted_raw()

    cands: set[str] = set()

    for q in present_ints:
        cands |= inv_closure.get(q, set())

        for anc in _ancestors(q):
            cands |= inv_raw.get(anc, set())

    return cands


def _term_lr(
    q: int,
    closure_D: dict[int, float],
    raw_D: dict[int, float],
    excluded_D: set[int],
    bg: dict[int, float],
) -> float:
    if q in excluded_D:
        return _EXCLUDED_OBSERVED_LR
    b = bg.get(q)
    if b is None or b <= 0.0:
        b = 1.0 / (2.0 * max(1, hpo_lr.n_diseases()))
    f = closure_D.get(q)
    if f is not None:
        return max(f, 1e-6) / b
    best: tuple[float, float] | None = None
    for a in _ancestors(q):
        fa = raw_D.get(a)
        if fa is None:
            continue
        ba = bg.get(a, 1.0)
        if best is None or ba < best[1]:
            best = (fa, ba)
    if best is not None:
        fa, ba = best
        return (fa * _CHILD_PENALTY) / max(ba, 1e-6)
    return _NO_MATCH_LR


def _absent_term_lr(
    q: int,
    closure_D: dict[int, float],
    raw_D: dict[int, float],
    excluded_D: set[int],
    bg: dict[int, float],
) -> float:
    if q in excluded_D:
        return _NEGATED_EXCLUDED_LR
    f = closure_D.get(q)

    if f is None:
        return 1.0

    b = bg.get(q, 0.0)
    denom = max(1e-3, 1.0 - b)
    return max(_ABSENT_FLOOR, (1.0 - f) / denom)


def score_diseases(
    present_ids: list[str],
    absent_ids: list[str] | None = None,
) -> list[tuple[str, float]]:
    present_ints = {i for i in (_to_int(h) for h in present_ids) if i is not None}
    absent_ints = {i for i in (_to_int(h) for h in (absent_ids or [])) if i is not None}
    if not present_ints:
        return []

    closure = _disease_closure()
    raw = hpo_lr.disease_hpo_frequencies()
    excluded = hpo_lr.disease_excluded_terms()
    bg = _bg_closure()

    scored: list[tuple[str, float]] = []
    for D in _candidate_diseases(present_ints):
        closure_D = closure.get(D, {})
        raw_D = raw.get(D, {})
        excluded_D = excluded.get(D, set())
        loglr = 0.0
        for q in present_ints:
            loglr += math.log(_term_lr(q, closure_D, raw_D, excluded_D, bg))
        for q in absent_ints:
            loglr += math.log(_absent_term_lr(q, closure_D, raw_D, excluded_D, bg))
        scored.append((D, loglr))

    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored


def score_genes(
    present_ids: list[str],
    absent_ids: list[str] | None = None,
    top_n: int = 200,
) -> list[tuple[str, float, str]]:
    disease_scores = score_diseases(present_ids, absent_ids)
    if not disease_scores:
        return []

    best: dict[str, tuple[float, str]] = {}
    for disease_id, loglr in disease_scores:
        genes = hpo_lr.genes_for_disease(disease_id)
        if not genes and disease_id.upper().startswith("ORPHA"):
            try:
                genes = {ga.gene_symbol for ga in orphanet.get_genes_for_disease(disease_id)}
            except Exception:
                genes = set()
        for g in genes:
            cur = best.get(g)
            if cur is None or loglr > cur[0]:
                best[g] = (loglr, disease_id)

    out = [(g, s, d) for g, (s, d) in best.items()]
    out.sort(key=lambda x: (-x[1], x[0]))
    return out[:top_n]


def _posttest_probabilities(disease_scores: list[tuple[str, float]]) -> dict[str, float]:
    if not disease_scores:
        return {}
    mx = max(s for _, s in disease_scores)
    exps = [(d, math.exp(s - mx)) for d, s in disease_scores]
    z = sum(e for _, e in exps) or 1.0
    return {d: e / z for d, e in exps}


def evidence_terms_for_disease(
    disease_id: str,
    present_ids: list[str],
    absent_ids: list[str] | None = None,
) -> dict:
    present_ints = [i for i in (_to_int(h) for h in present_ids) if i is not None]
    absent_ints = [i for i in (_to_int(h) for h in (absent_ids or [])) if i is not None]

    closure_D = _disease_closure().get(disease_id, {})
    raw_D = hpo_lr.disease_hpo_frequencies().get(disease_id, {})
    excluded_D = hpo_lr.disease_excluded_terms().get(disease_id, set())
    bg = _bg_closure()
    refs_map = hpo_lr.annotation_references().get(disease_id, {})

    terms = []
    full_log_lr = 0.0

    for q in present_ints:
        lr = _term_lr(q, closure_D, raw_D, excluded_D, bg)
        log_lr = math.log(max(lr, 1e-12))
        full_log_lr += log_lr

        terms.append({
            "hpo_id": f"HP:{q:07d}",
            "status": "present",
            "log_lr": round(log_lr, 3),
            "supports": lr > 1.0,
            "references": list(refs_map.get(q, [])),
        })

    for q in absent_ints:
        lr = _absent_term_lr(q, raw_D, excluded_D, bg)
        log_lr = math.log(max(lr, 1e-12))
        full_log_lr += log_lr

        terms.append({
            "hpo_id": f"HP:{q:07d}",
            "status": "absent",
            "log_lr": round(log_lr, 3),
            "supports": lr > 1.0,
            "references": list(refs_map.get(q, [])),
        })

    supporting = [t for t in terms if t["log_lr"] > 0]
    contradicting = [t for t in terms if t["log_lr"] < 0]

    supporting.sort(key=lambda x: x["log_lr"], reverse=True)
    contradicting.sort(key=lambda x: x["log_lr"])

    return {
        "disease_id": disease_id,
        "full_log_lr": round(full_log_lr, 3),
        "supporting_terms": supporting,
        "contradicting_terms": contradicting,
        "all_terms": terms,
    }


def supporting_terms_for_disease(
    disease_id: str, present_ids: list[str], top_k: int = 5,
) -> list[dict]:
    present_ints = [i for i in (_to_int(h) for h in present_ids) if i is not None]
    closure_D = _disease_closure().get(disease_id, {})
    raw_D = hpo_lr.disease_hpo_frequencies().get(disease_id, {})
    excluded_D = hpo_lr.disease_excluded_terms().get(disease_id, set())
    bg = _bg_closure()
    refs_map = hpo_lr.annotation_references().get(disease_id, {})
    contribs: list[dict] = []
    for q in present_ints:
        lr = _term_lr(q, closure_D, raw_D, excluded_D, bg)
        if lr <= 1.0:
            continue
        contribs.append({
            "hpo_id": f"HP:{q:07d}",
            "log_lr": round(math.log(lr), 3),
            "references": list(refs_map.get(q, [])),
        })
    contribs.sort(key=lambda c: c["log_lr"], reverse=True)
    return contribs[:top_k]


def rank_diseases(
    present_ids: list[str],
    absent_ids: list[str] | None = None,
    top_n: int = 20,
) -> list[dict]:
    scores = score_diseases(present_ids, absent_ids)
    if not scores:
        return []
    post = _posttest_probabilities(scores)
    names = hpo_lr.disease_names()
    out: list[dict] = []
    for disease_id, loglr in scores[:top_n]:
        genes = sorted(hpo_lr.genes_for_disease(disease_id))
        if not genes and disease_id.upper().startswith("ORPHA"):
            try:
                genes = sorted({ga.gene_symbol for ga in orphanet.get_genes_for_disease(disease_id)})
            except Exception:
                genes = []
        out.append({
            "disease_id": disease_id,
            "name": names.get(disease_id, disease_id),
            "log_lr": round(loglr, 3),
            "posttest_prob": round(post.get(disease_id, 0.0), 4),
            "genes": genes,
            "supporting_terms": supporting_terms_for_disease(disease_id, present_ids),
        })
    return out


def evidence_for_gene(
    gene: str, present_ids: list[str], absent_ids: list[str] | None = None,
) -> dict | None:
    for d in rank_diseases(present_ids, absent_ids, top_n=300):
        if gene in d["genes"]:
            refs = sorted({r for t in d["supporting_terms"] for r in t["references"]})
            return {
                "disease_id": d["disease_id"],
                "disease_name": d["name"],
                "log_lr": d["log_lr"],
                "posttest_prob": d["posttest_prob"],
                "supporting_terms": d["supporting_terms"],
                "references": refs,
            }
    return None


def _to_int(hpo_id: str) -> int | None:
    try:
        return int(hpo_id.split(":")[1]) if ":" in hpo_id else int(hpo_id)
    except (ValueError, IndexError, AttributeError):
        return None


def is_available() -> bool:
    return hpo_lr.is_available()
