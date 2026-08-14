from __future__ import annotations

import logging
import re
from typing import Any

from generesolver.dx import unified_scorer, pubmed_agent, reasoner, ranker
from generesolver.integrations import hpo_db

logger = logging.getLogger(__name__)

_PROMPT_CANDIDATES = 80
_PUBMED_TOPN = 12


def _norm(name: str) -> str:
    s = re.sub(r"\(.*?\)", " ", (name or "").lower())
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _label(hpo_id: str) -> str:
    try:
        return str(hpo_db._get_ontology().get_hpo_object(int(hpo_id.split(":")[1])).name)
    except Exception:
        return hpo_id


def _findings_block(cands: list[dict], lit: dict[str, dict]) -> str:
    lines: list[str] = []
    for c in cands:
        genes = ", ".join(c.get("genes") or []) or "unknown"
        log_lr = c.get("log_lr")
        lr_note = f" | log-LR: {log_lr:+.2f}" if log_lr is not None else ""
        lines.append(f"- {c['name']} | gene(s): {genes}{lr_note}")
        if c.get("matched_terms"):
            lines.append(f"    matched HPO terms: {', '.join(c['matched_terms'][:8])}")
        for bf in c.get("biochem_findings") or []:
            lines.append(f"    {bf.get('finding')}")
        ev = lit.get(c["name"]) or {}
        for t in ev.get("titles") or []:
            lines.append(f"    literature: {t}")
    return "\n".join(lines)


def diagnose(present_ids: list[str], absent_ids: list[str] | None = None,
             *, top_n: int = 10, debug_gold_ids: list[str] | None = None) -> list[dict[str, Any]]:
    present_ids = [h for h in (present_ids or []) if h]
    absent_ids = [h for h in (absent_ids or []) if h]
    if not present_ids:
        return []
    scored = unified_scorer.score_diseases(present_ids, absent_ids, top_n=_PROMPT_CANDIDATES)
    shown = unified_scorer.as_candidate_dicts(scored)

    if debug_gold_ids:
        full_scored = unified_scorer.score_diseases(present_ids, absent_ids, top_n=10_000)
        for gid in debug_gold_ids:
            hit = next((i for i, s in enumerate(full_scored) if s.disease_id == gid), None)
            if hit is None:
                logger.warning("debug_gold: %s NOT in scored universe at all (0 present-term "
                                "overlap and 0 gene-hop match) out of %d scored diseases",
                                gid, len(full_scored))
            else:
                in_prompt = "WITHIN" if hit < _PROMPT_CANDIDATES else "BEYOND"
                logger.warning("debug_gold: %s at rank %d/%d (log_lr=%.3f, source=%s) — "
                                "%s the top-%d prompt cutoff",
                                gid, hit + 1, len(full_scored), full_scored[hit].log_lr,
                                full_scored[hit].source, in_prompt, _PROMPT_CANDIDATES)

    patient_words = {w for h in present_ids for w in re.findall(r"[a-z]{5,}", _label(h).lower())}
    lit = pubmed_agent.evidence_for([c["name"] for c in shown[:_PUBMED_TOPN]], patient_words)

    case_block = "\n".join(f"- {h} {_label(h)}".rstrip() for h in present_ids)
    if absent_ids:
        case_block += "\n\nABSENT / excluded findings (already penalised in each candidate's " \
            "log-LR score above; still weigh them in your own judgement):\n" + \
            "\n".join(f"- {h} {_label(h)}".rstrip() for h in absent_ids)
    findings_block = _findings_block(shown, lit) or "(no candidates scored)"

    reasoning, mechanism, proposed = reasoner.reason(case_block, findings_block)

    cand_names: list[str] = [c["name"] for c in shown]
    seen_norm = {_norm(n) for n in cand_names}
    for p in proposed:
        if _norm(p) not in seen_norm:
            cand_names.append(p)
            seen_norm.add(_norm(p))

    ranked_names = ranker.rank(case_block, reasoning, mechanism, findings_block, cand_names)
    if not ranked_names:
        ranked_names = [c["name"] for c in shown]
    else:
        seen = {_norm(n) for n in ranked_names}
        missing = [n for n in cand_names if _norm(n) not in seen]
        if missing:
            logger.warning("dx.ranker returned a partial ranking: %d/%d candidates; "
                            "appending %d dropped candidate(s) at the tail: %s",
                            len(ranked_names), len(cand_names), len(missing), missing)
            ranked_names = ranked_names + missing

    by_norm: dict[str, dict] = {}
    for c in sorted(shown, key=lambda c: c.get("log_lr") or float("-inf")):
        by_norm[_norm(c["name"])] = c
    out: list[dict[str, Any]] = []
    used: set[str] = set()
    for name in ranked_names:
        key = _norm(name)
        if key in used:
            continue
        used.add(key)
        c = by_norm.get(key)
        if c is None:
            c = next((cc for k, cc in by_norm.items() if k and (k in key or key in k)), None)
        if c is not None:
            out.append({"disease_id": c["disease_id"], "name": c["name"],
                        "genes": list(c.get("genes") or []), "supporting_terms": []})
        else:
            out.append({"disease_id": "", "name": name, "genes": [], "supporting_terms": []})
        if len(out) >= top_n:
            break

    n = len(out) or 1
    for i, d in enumerate(out):
        d["posttest_prob"] = round((n - i) / n, 4)
    if out:
        out[0]["reasoning"] = reasoning
        out[0]["mechanism"] = mechanism
    logger.info("dx.pipeline: ranked %d diseases (%d scored, %d proposed)",
                len(out), len(scored), len(proposed))
    return out