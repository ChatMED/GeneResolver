from __future__ import annotations

import functools
import json
import logging
import re
import time
from typing import Any

from generesolver.integrations import (
    llm_client,
    orphanet,
    lr_score,
    open_targets_api,
    hpo_db,
    hpo_lr,
    mondo_names,
    similar_case,
    pubcasefinder,
)
from generesolver.components import literature, biochemical
from generesolver.config import LOCAL_MODE
from pydantic import BaseModel, ConfigDict, Field
import math

logger = logging.getLogger(__name__)

_MAX_CANDIDATES = 25
_MAX_GROUNDED = 30
_LR_SEED_TOP_N = 100
_LR_SEED_PARENT_SCAN = 300
_STAGE_A_MAX_TOKENS = 3000
_STAGE_C_MAX_TOKENS = 2048
_STAGE_C_CTX_LIMIT = {"medgemma": 16384, "claude": 200000, "anthropic": 200000,
                      "deepseek": 64000}
_STAGE_C_SYS_RESERVE = 800
_STAGE_C_MIN_KEEP = 12


def _stage_c_input_budget() -> int:
    ctx = _STAGE_C_CTX_LIMIT.get(llm_client.active_backend(), 16384)
    return ctx - _STAGE_C_MAX_TOKENS - _STAGE_C_SYS_RESERVE
_STAGE_C_SLATE = 20
_STAGE_C_SEED_SLOTS = 15
class _RankedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    priority: float


class _StageCResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ranked: list[_RankedItem] = Field(default_factory=list)


class _QueriesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    queries: list[str] = Field(default_factory=list)

_STAGE_C_LIT_ENABLED = True
_STAGE_C_LIT_TOPN = 3
_STAGE_C_LIT_TITLES = 3
_STAGE_C_ABS_ENABLED = True
_STAGE_C_ABS_TOPN = 3
_STAGE_C_ABS_PER = 3
_STAGE_C_ABS_CHARS = 1500

def _label(hpo_id: str) -> str:
    try:
        return str(hpo_db._get_ontology().get_hpo_object(int(hpo_id.split(":")[1])).name)
    except Exception:
        return ""


def _terms_block(ids: list[str]) -> str:
    lines = []
    for hid in ids:
        lab = _label(hid)
        lines.append(f"- {hid} {lab}".rstrip())
    return "\n".join(lines)


def _hpo_terms_str(terms: list[dict]) -> str:
    parts = []
    for t in terms:
        lab = t.get("label") or t.get("hpo_id") or ""
        parts.append(f"{lab} ({t.get('log_lr')})" if "log_lr" in t else str(lab))
    return ", ".join(parts) if parts else "none"


def _render_evidence(evidence: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for i, e in enumerate(evidence, 1):
        genes = ", ".join(e.get("genes") or []) or "unknown"
        lines = [
            f"[{i}] {e['disease_id']} - {e['disease']} | gene(s): {genes}",
            f"    matched HPO terms: {_hpo_terms_str(e.get('matched_hpo_terms') or [])}",
            f"    contradicting HPO terms: {_hpo_terms_str(e.get('contradicting_hpo_terms') or [])}",
            f"    deterministic phenotype score (summed log-LR): {e.get('full_summed_log_lr')}",
            f"    pubmed hits: {e.get('pubmed_hits')} | open-targets gene-disease assoc: {e.get('open_targets_assoc')}",
        ]
        for m in e.get("pathognomonic_biochemical_marker") or []:
            lines.append(
                f"    pathognomonic biochemical marker: {m.get('finding')} "
                f"(specific to {m.get('specific_to_n_diseases')} diseases)"
            )
        for t in e.get("literature_titles") or []:
            lines.append(f"    literature title: {t}")
        for ab in e.get("literature_abstracts") or []:
            text = ab if isinstance(ab, str) else (ab.get("abstract") or ab.get("text") or str(ab))
            lines.append(f"    literature abstract: {text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _estimate_tokens(text: str) -> int:
    return int(len(text) / 3.6) + 1


def _fit_stage_c_user(assemble, evidence: list[dict[str, Any]], budget_tokens: int) -> str:
    user = assemble(evidence)
    if _estimate_tokens(user) <= budget_tokens:
        return user
    ev = [dict(e) for e in evidence]
    for field in ("literature_abstracts", "literature_titles"):
        for e in reversed(ev):
            if e.get(field):
                e.pop(field, None)
                user = assemble(ev)
                if _estimate_tokens(user) <= budget_tokens:
                    return user
    while len(ev) > _STAGE_C_MIN_KEEP and _estimate_tokens(user) > budget_tokens:
        ev.pop()
        user = assemble(ev)
    return user


_STAGE_A_SYSTEM = "You are a consultant clinical geneticist specialised in rare disease."

def _biochemical_search_queries(
    bio_cands: list[dict],
    present_ids: list[str] | None = None,
) -> list[str]:
    marker_lines: list[str] = []
    seen_markers: set[str] = set()
    for c in bio_cands:
        for m in c.get("markers") or []:
            hid = m.get("hpo_id")
            if not hid or hid in seen_markers:
                continue
            seen_markers.add(hid)
            marker_lines.append(
                f"- {hid} {m.get('label', '')} "
                f"(specific_to_n_diseases={m.get('n_diseases')})"
            )

    if not marker_lines and present_ids:
        for hid in present_ids[:12]:
            marker_lines.append(f"- {hid} {_label(hid)}".rstrip())

    if not marker_lines:
        return []

    existing_lines = [
        f"- {c.get('disease_id')}: {c.get('name')}"
        for c in bio_cands[:15]
    ]

    user = (
        "Generate biomedical/PubMed search queries to find the SPECIFIC rare disease "
        "(usually an inborn error of metabolism) that explains this patient's findings. "
        "Patient findings:\n"
        + "\n".join(marker_lines)
        + ("\n\nDiseases already found by HPO-A reverse lookup:\n" + "\n".join(existing_lines)
           if existing_lines else "")
        + "\n\nRules:\n"
        "- Do NOT output disease names as candidates. Do NOT diagnose. Output only queries.\n"
        "- CRITICAL: keep each query findable. AND together AT MOST TWO concepts; never AND "
        "three or more exact quoted phrases (that returns zero PubMed results). For a "
        "multi-word finding, prefer unquoted words or an OR of close synonyms over an exact "
        "quoted phrase.\n"
        "- Provide a GRADED set: 2-3 narrow queries (the most discriminating finding pair) "
        "and 2-3 broader ones (a single key finding + 'deficiency'/'inborn error'), so at "
        "least some return hits.\n"
        '- Example shape: ["alopecia AND lactic acidosis", "(alopecia OR dermatitis) AND '
        'organic aciduria", "infantile lactic acidosis inborn error"].\n\n'
        'Return JSON of the form {"queries":["...", "..."]}.'
    )

    try:
        resp = llm_client.complete_structured(
            _STAGE_A_SYSTEM, user, schema_model=_QueriesResponse,
            max_tokens=_STAGE_A_MAX_TOKENS,
        )
        raw_queries = resp.queries if resp else []
    except Exception as exc:
        logger.warning("biochemical query generation failed: %s", exc)
        return []

    out: list[str] = []
    seen: set[str] = set()
    for q in raw_queries:
        q = str(q).strip()
        if len(q) >= 6 and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)

    return out[:8]

def _stage_a_hypothesize(present_ids: list[str], absent_ids: list[str]) -> list[str]:
    user = (
        "A patient presents with the following standardised (HPO) phenotype.\n\n"
        f"PRESENT findings:\n{_terms_block(present_ids)}\n"
    )
    if absent_ids:
        user += f"\nEXPLICITLY ABSENT findings:\n{_terms_block(absent_ids)}\n"
    user += (
        "\nList a BROAD differential that could explain this phenotype. Cast wide for "
        f"recall: up to {_MAX_CANDIDATES} entries, most likely first.\n\n"
        "CRITICAL — name SPECIFIC diseases exactly as catalogued in OMIM / Orphanet, "
        "NOT umbrella categories, groups, or pathways (a category resolves to no single "
        "disease and is dropped). If you think of a category, expand it into its "
        "specific member diseases instead.\n"
        "  WRONG (category): 'Organic acidemias', 'Mitochondrial disorders', "
        "'Congenital disorders of glycosylation', 'Fatty acid oxidation defects', "
        "'Urea cycle disorders', 'Aminoacidopathies'.\n"
        "  RIGHT (specific): 'Isovaleric acidemia', '3-methylcrotonyl-CoA carboxylase "
        "deficiency', 'Mitochondrial complex IV deficiency', 'PMM2-CDG (CDG type Ia)', "
        "'Succinic semialdehyde dehydrogenase deficiency', 'L-2-hydroxyglutaric "
        "aciduria', 'Maple syrup urine disease', 'Systemic primary carnitine "
        "deficiency'.\n"
        "For any inborn error of metabolism, name the SPECIFIC enzyme deficiency / "
        "disease and its subtype, using the full disease name (not just an acronym), "
        "not the gene name. Give each a one-line reason grounded in the findings.\n"
        'Return ONLY JSON: {"candidates": [{"disease": "<specific disease name>", '
        '"why": "<findings that suggest it>"}, ...]}'
    )
    names = _stage_a_extract(_stage_a_text(user))
    if not names:
        names = _stage_a_extract(_stage_a_text(user))

    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        name = (name or "").strip()
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            out.append(name)
    return out[:_MAX_CANDIDATES]


def _stage_a_text(user: str) -> str:
    try:
        return llm_client.complete_text(_STAGE_A_SYSTEM, user, max_tokens=_STAGE_A_MAX_TOKENS)
    except Exception as exc:
        logger.warning("reasoner Stage A call failed: %s", exc)
        return ""


def _stage_a_extract(raw: str) -> list[str]:
    if not raw:
        return []
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and isinstance(obj.get("candidates"), list):
            got = [(c.get("disease") if isinstance(c, dict) else str(c)) for c in obj["candidates"]]
        elif isinstance(obj, dict) and isinstance(obj.get("diseases"), list):
            got = [str(x) for x in obj["diseases"]]
        elif isinstance(obj, list):
            got = [str(x) for x in obj]
        else:
            got = []
        if got:
            return [g for g in got if g]
    except (json.JSONDecodeError, ValueError):
        pass
    return re.findall(r'"disease"\s*:\s*"([^"]+)"', cleaned)


def _ot_disease_gene_score(name: str, genes: set[str]) -> float:
    if not genes:
        return 0.0
    try:
        targets = open_targets_api.get_disease_associated_targets(name, size=30)
    except Exception:
        return 0.0
    best = 0.0
    for t in targets:
        if t.get("gene_symbol") in genes:
            best = max(best, float(t.get("score", 0.0) or 0.0))
    return best


def _evidence_bundle(
    disease_id: str,
    name: str,
    genes: list[str],
    present_ids: list[str],
    absent_ids: list[str] | None = None,
    *,
    biochemical_marker: list[dict] | None = None,
    sources: list[str] | None = None,
    engine_score: float | None = None,
) -> dict[str, Any] | None:
    try:
        ev = lr_score.evidence_terms_for_disease(
            disease_id=disease_id,
            present_ids=present_ids,
            absent_ids=absent_ids or [],
        )
        matched = ev.get("supporting_terms", [])[:6]
        contradicting = ev.get("contradicting_terms", [])[:6]
        lr_sum = float(ev.get("full_log_lr", 0.0))
    except Exception:
        matched = []
        contradicting = []
        lr_sum = 0.0
    top_labels = [m.get("label") or _label(m["hpo_id"]) for m in matched[:3]]
    top_labels = [l for l in top_labels if l]
    finding_clause = " AND (" + " OR ".join(f'"{l}"' for l in top_labels) + ")" if top_labels else ""
    query = f'"{name}"[ti]' + finding_clause
    try:
        pmids = literature.pubmed_search(query)
        if not pmids:
            pmids = literature.pubmed_search(f'"{name}"' + finding_clause)
    except Exception:
        pmids = []

    ot_score = _ot_disease_gene_score(name, set(genes))

    grounded = (bool(matched) or bool(pmids) or ot_score > 0.0
                or bool(biochemical_marker) or bool(sources))
    if not grounded:
        return None

    bundle = {
        "disease": name,
        "disease_id": disease_id,
        "genes": genes,
        "matched_terms": [
            {
                "hpo_id": m["hpo_id"],
                "label": m.get("label") or _label(m["hpo_id"]),
                "log_lr": round(float(m.get("log_lr", 0.0)), 2),
            }
            for m in matched
        ],
        "contradicting_terms": [
            {
                "hpo_id": m["hpo_id"],
                "label": m.get("label") or _label(m["hpo_id"]),
                "log_lr": round(float(m.get("log_lr", 0.0)), 2),
            }
            for m in contradicting
        ],
        "lr_sum": round(lr_sum, 3),
        "n_pmids": len(pmids),
        "pmids": list(pmids[:5]),
        "ot_assoc": round(ot_score, 3),
    }
    if biochemical_marker:
        bundle["biochemical_marker"] = biochemical_marker
    if sources:
        bundle["sources"] = sources
    if engine_score:
        bundle["engine_score"] = float(engine_score)
    return bundle


def _genes_for_disease(disease_id: str) -> list[str]:
    if disease_id.upper().startswith("ORPHA"):
        try:
            return sorted({ga.gene_symbol for ga in orphanet.get_genes_for_disease(disease_id)})
        except Exception:
            pass
    try:
        return sorted(hpo_lr.genes_for_disease(disease_id) or [])
    except Exception:
        return []


def _name_variants(name: str) -> list[str]:
    out = [name]
    out += [m.strip() for m in re.findall(r"\(([^)]+)\)", name)]
    no_paren = re.sub(r"\s*\([^)]*\)", "", name).strip()
    if no_paren and no_paren != name:
        out.append(no_paren)
    if "," in no_paren:
        out.append(no_paren.split(",")[0].strip())
    seen: set[str] = set()
    return [v for v in out if v and not (v.lower() in seen or seen.add(v.lower()))]


def _expand_names(names: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def _push(s: str) -> None:
        s = s.strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)

    for name in names:
        _push(name)
        for paren in re.findall(r"\(([^)]+)\)", name):
            body = re.sub(r"^\s*(e\.?g\.?|i\.?e\.?|such as|including)[,:]?\s*", "",
                          paren.strip(), flags=re.IGNORECASE)
            if "," in body:
                for frag in body.split(","):
                    _push(frag)
    return out


def _name_tokens(s: str) -> frozenset[str]:
    return frozenset(t for t in re.findall(r"[a-z0-9]+", (s or "").lower())
                     if len(t) > 1 and not t.isdigit())


@functools.lru_cache(maxsize=1)
def _token_index() -> dict[frozenset[str], tuple[str, str]]:
    idx: dict[frozenset[str], tuple[str, str]] = {}
    ambiguous: set[frozenset[str]] = set()

    def _put(ts: frozenset[str], did: str, label: str) -> None:
        if len(ts) < 2:
            return
        if ts in idx and idx[ts][0] != did:
            ambiguous.add(ts)
        idx.setdefault(ts, (did, label))

    try:
        for d in orphanet.get_all():
            for lab in (d.label, *d.synonyms):
                _put(_name_tokens(lab), d.orpha_id, d.label or lab)
    except Exception as exc:
        logger.debug("token index: orphanet pass failed: %s", exc)
    try:
        for lab, ids in mondo_names._index().items():
            for pref in ("ORPHA", "OMIM", "MONDO"):
                hit = next((i for i in sorted(ids) if i.startswith(pref)), None)
                if hit:
                    _put(_name_tokens(lab), hit, lab)
                    break
    except Exception as exc:
        logger.debug("token index: mondo pass failed: %s", exc)

    for ts in ambiguous:
        idx.pop(ts, None)
    return idx


def _resolve_name(name: str) -> tuple[str, str] | None:
    variants = _name_variants(name)
    for variant in variants:
        d = orphanet.normalize(variant)
        if d is not None:
            return d.orpha_id, (d.label or name)
        hit = mondo_names.resolve(variant)
        if hit is not None:
            return hit[0], name
    tindex = _token_index()
    for variant in variants:
        hit = tindex.get(_name_tokens(variant))
        if hit is not None:
            return hit
    return None

def _disease_labels_in_text(text: str) -> list[tuple[str, str]]:
    text_l = (text or "").lower()
    hits: list[tuple[str, str]] = []

    try:
        for d in orphanet.get_all():
            labels = [d.label, *d.synonyms]
            for lab in labels:
                if not lab:
                    continue
                lab_l = lab.lower()
                if len(lab_l) >= 8 and lab_l in text_l:
                    hits.append((d.orpha_id, d.label or lab))
                    break
    except Exception as exc:
        logger.debug("biochemical query disease extraction: Orphanet scan failed: %s", exc)

    try:
        for lab, ids in mondo_names._index().items():
            lab_l = str(lab).lower()
            if len(lab_l) < 8 or lab_l not in text_l:
                continue
            for pref in ("OMIM", "ORPHA", "MONDO"):
                did = next((i for i in sorted(ids) if i.startswith(pref)), None)
                if did:
                    hits.append((did, lab))
                    break
    except Exception as exc:
        logger.debug("biochemical query disease extraction: MONDO scan failed: %s", exc)

    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for did, lab in hits:
        if did not in seen:
            seen.add(did)
            out.append((did, lab))

    return out[:20]

def _biochemical_query_search_candidates(
    bio_cands: list[dict],
    present_ids: list[str] | None = None,
) -> list[dict]:
    queries = _biochemical_search_queries(bio_cands, present_ids)
    if not queries:
        return []

    _LAST_ENGINES["biochemical_queries"] = list(queries)

    markers: list[dict] = []
    seen_marker: set[str] = set()
    for c in bio_cands:
        for m in c.get("markers") or []:
            hid = m.get("hpo_id")
            if hid and hid not in seen_marker:
                seen_marker.add(hid)
                markers.append(m)

    out: list[dict] = []
    seen_ids: set[str] = set()

    for rank, q in enumerate(queries, start=1):
        try:
            pmids = literature.pubmed_search(q)
            if not pmids:
                relaxed = q.replace('"', " ").replace(" AND ", " ").replace(" OR ", " ")
                relaxed = " ".join(relaxed.split())
                if relaxed and relaxed != q:
                    pmids = literature.pubmed_search(relaxed)
            titles = literature.pubmed_titles(tuple(pmids[:5]), max_n=5) if pmids else []
            abstracts = literature.pubmed_abstracts(
                tuple(pmids[:3]),
                max_n=3,
                max_chars=1200,
            ) if pmids else []
        except Exception as exc:
            logger.debug("biochemical query search failed for %r: %s", q, exc)
            continue

        text = " ".join(titles + abstracts)
        for did, label in _disease_labels_in_text(text):
            if did in seen_ids:
                continue
            seen_ids.add(did)
            out.append({
                "disease_id": did,
                "name": label,
                "genes": _genes_for_disease(did),
                "markers": markers,
                "sources": ["biochemical_query_search"],
                "engine_score": 1.0 / float(rank),
                "search_query": q,
            })

    return out


def _ground_candidate(
    name: str,
    present_ids: list[str],
    absent_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    hit = _resolve_name(name)
    if hit is None:
        return None
    disease_id, label = hit
    return _evidence_bundle(
        disease_id,
        label,
        _genes_for_disease(disease_id),
        present_ids,
        absent_ids,
    )


def _stage_b_ground(
    names: list[str],
    present_ids: list[str],
    absent_ids: list[str] | None = None,
    engine_candidates: list[dict] | None = None,
) -> list[dict[str, Any]]:
    grounded: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    prop_rank: dict[str, int] = {}
    for rank, name in enumerate(_expand_names(names), 1):
        hit = _resolve_name(name)
        if hit and hit[0] not in prop_rank:
            prop_rank[hit[0]] = rank

    for c in (engine_candidates or []):
        did = c["disease_id"]
        if did in seen_ids:
            continue
        try:
            bundle = _evidence_bundle(
                did,
                c.get("name") or did,
                c.get("genes") or [],
                present_ids,
                absent_ids,
                biochemical_marker=c.get("markers"),
                sources=c.get("sources"),
                engine_score=c.get("engine_score"),
            )
        except Exception as exc:
            logger.debug("engine grounding failed for %s: %s", did, exc)
            bundle = None
        if bundle:
            seen_ids.add(did)
            if did in prop_rank:
                bundle["stage_a_rank"] = prop_rank[did]
            grounded.append(bundle)
    n_bio = len(grounded)

    import generesolver.config as _gcfg
    fallback_grounding = getattr(_gcfg, "STAGE_B_FALLBACK_GROUNDING", False)
    if fallback_grounding:
        from generesolver.dx import tools as _dx_tools
    llm_grounded: list[dict[str, Any]] = []
    for rank, name in enumerate(_expand_names(names), 1):
        try:
            bundle = _ground_candidate(name, present_ids, absent_ids)
        except Exception as exc:
            logger.debug("grounding failed for %r: %s", name, exc)
            bundle = None
        if bundle and bundle["disease_id"] not in seen_ids:
            seen_ids.add(bundle["disease_id"])
            bundle["stage_a_rank"] = rank
            llm_grounded.append(bundle)

        if not fallback_grounding:
            continue
        try:
            extra = _dx_tools.search_disease(name, limit=3)
        except Exception as exc:
            logger.debug("fallback search_disease failed for %r: %s", name, exc)
            extra = []
        for cand in extra:
            did = cand.get("disease_id")
            if not did or did in seen_ids:
                continue
            try:
                bundle2 = _evidence_bundle(
                    did, cand.get("name") or did, cand.get("genes") or [],
                    present_ids, absent_ids,
                )
            except Exception as exc:
                logger.debug("fallback grounding failed for %s: %s", did, exc)
                bundle2 = None
            if bundle2:
                seen_ids.add(did)
                bundle2["stage_a_rank"] = rank
                llm_grounded.append(bundle2)
    def _cap_key(b: dict) -> tuple:
        return (_det_score(b), b["n_pmids"], b["ot_assoc"])
    protected = [b for b in llm_grounded
                 if (b.get("stage_a_rank") or 10**9) <= _STAGE_A_PROTECT_TOPN]
    unprotected = [b for b in llm_grounded
                   if (b.get("stage_a_rank") or 10**9) > _STAGE_A_PROTECT_TOPN]
    unprotected.sort(key=_cap_key, reverse=True)
    budget = max(0, _MAX_GROUNDED - n_bio)
    return grounded + protected + unprotected[:budget]


_STAGE_C_SYSTEM = (
    "You are a consultant clinical geneticist producing a final differential. "
    "Rank ONLY by how well the RETRIEVED EVIDENCE explains the patient — never by "
    "fame or prior belief. Use only the candidates provided."
)

def _merge_subtypes_into_parent(prelim: list[dict]) -> list[dict]:
    if len(prelim) < 2:
        return prelim
    norms = [_norm_disease_name(p["rep"].get("disease", "")) for p in prelim]

    parent_of: dict[int, int] = {}
    for ci in range(len(prelim)):
        best = None
        for pi in range(len(prelim)):
            if pi != ci and norms[pi] and _is_subtype_name(norms[pi], norms[ci]):
                if best is None or len(norms[pi]) > len(norms[best]):
                    best = pi
        if best is not None:
            parent_of[ci] = best
    if not parent_of:
        return prelim

    def _root(i: int) -> int:
        seen: set[int] = set()
        while i in parent_of and i not in seen:
            seen.add(i)
            i = parent_of[i]
        return i

    merged: dict[int, dict] = {}
    for i, p in enumerate(prelim):
        r = _root(i)
        if r not in merged:
            base = dict(prelim[r])
            base["tie_ids"] = list(prelim[r]["tie_ids"])
            base["member_pairs"] = list(prelim[r]["member_pairs"])
            merged[r] = base
        if i == r:
            continue
        tgt = merged[r]
        tgt["det_raw"] = max(tgt["det_raw"], p["det_raw"])
        tgt["priority"] = max(tgt["priority"], p["priority"])
        tgt["any_llm"] = tgt["any_llm"] or p["any_llm"]
        tgt["tie_ids"] = sorted(set(tgt["tie_ids"]) | set(p["tie_ids"]))
        tgt["member_pairs"] = tgt["member_pairs"] + p["member_pairs"]
    return list(merged.values())


def _similar_case_rerank(scored: list[tuple], present_ids: list[str]) -> list[tuple]:
    import generesolver.config as _gcfg
    if not getattr(_gcfg, "SIMILAR_CASE_RERANK", False):
        return scored
    K = int(getattr(_gcfg, "SIMILAR_CASE_RERANK_K", 12))
    lam = float(getattr(_gcfg, "SIMILAR_CASE_RERANK_LAMBDA", 1.0))
    try:
        votes = {v["disease_id"]: float(v.get("score", 0.0) or 0.0)
                 for v in similar_case.ranked_diseases(present_ids, top_n=40)}
    except Exception as exc:
        logger.warning("similar-case rerank failed: %s", exc)
        return scored
    if not votes:
        return scored
    cluster = sorted(range(len(scored)), key=lambda i: -scored[i][1])[:K]
    vmax = max([votes.get(scored[i][2].get("disease_id"), 0.0) for i in cluster] + [0.0])
    if vmax <= 0:
        return scored
    cl = set(cluster)
    out = []
    for i, t in enumerate(scored):
        if i in cl:
            tier, final, rep, tie, gi, mp = t
            final = final + lam * (votes.get(rep.get("disease_id"), 0.0) / vmax)
            out.append((tier, final, rep, tie, gi, mp))
        else:
            out.append(t)
    _LAST_ENGINES["simcase_rerank"] = {"cluster": len(cl), "voted_diseases": len(votes)}
    return out


def _stage_c_rank(
    grounded: list[dict[str, Any]], present_ids: list[str], top_n: int,
) -> list[dict[str, Any]]:
    if not grounded:
        return []
    evidence = []
    _ENGINE_SRC = {"lr_seed", "biochemical", "biochemical_query_search"}
    det_top = sorted(grounded, key=_det_score, reverse=True)[:_STAGE_C_SLATE]
    chosen = {b["disease_id"] for b in det_top}
    seed_extra = sorted(
        (b for b in grounded
         if (set(b.get("sources") or []) & _ENGINE_SRC) and b["disease_id"] not in chosen),
        key=lambda b: float(b.get("engine_score", 0.0) or 0.0), reverse=True,
    )[:_STAGE_C_SEED_SLOTS]
    llm_grounded = det_top + seed_extra
    _LAST_ENGINES["stage_c_slate"] = [b["disease_id"] for b in llm_grounded]
    logger.info("reasoner Stage C: ranking slate = %d candidates (%d det_top + %d seed_extra)",
                len(llm_grounded), len(det_top), len(seed_extra))
    _t_lit0 = time.time()
    n_titles_fetched = n_abstracts_fetched = 0
    for i, b in enumerate(llm_grounded):
        e = {
            "disease": b["disease"],
            "disease_id": b["disease_id"],
            "genes": b["genes"],
            "matched_hpo_terms": b["matched_terms"],
            "contradicting_hpo_terms": b.get("contradicting_terms", []),
            "full_summed_log_lr": b["lr_sum"],
            "pubmed_hits": b["n_pmids"],
            "open_targets_assoc": b["ot_assoc"],
            "pmids": list(b.get("pmids") or []),
        }
        if b.get("biochemical_marker"):
            e["pathognomonic_biochemical_marker"] = [
                {"finding": m.get("label"), "specific_to_n_diseases": m.get("n_diseases")}
                for m in b["biochemical_marker"]
            ]
        if _STAGE_C_LIT_ENABLED and i < _STAGE_C_LIT_TOPN and b.get("pmids"):
            _t0 = time.time()
            try:
                titles = literature.pubmed_titles(tuple(b["pmids"]), max_n=_STAGE_C_LIT_TITLES)
            except Exception as exc:
                logger.info("reasoner Stage C: title fetch failed for %s after %.1fs (%s)",
                            b["disease_id"], time.time() - _t0, exc)
                titles = []
            if titles:
                e["literature_titles"] = [t[:140] for t in titles]
                n_titles_fetched += 1
        if _STAGE_C_ABS_ENABLED and i < _STAGE_C_ABS_TOPN and b.get("pmids"):
            _t0 = time.time()
            try:
                abstracts = literature.pubmed_abstracts(
                    tuple(b["pmids"]), max_n=_STAGE_C_ABS_PER, max_chars=_STAGE_C_ABS_CHARS)
            except Exception as exc:
                logger.info("reasoner Stage C: abstract fetch failed for %s after %.1fs (%s)",
                            b["disease_id"], time.time() - _t0, exc)
                abstracts = []
            if abstracts:
                e["literature_abstracts"] = abstracts
                n_abstracts_fetched += 1
        evidence.append(e)
    logger.info(
        "reasoner Stage C: literature fetch done in %.1fs (%d/%d titles, %d/%d abstracts)",
        time.time() - _t_lit0, n_titles_fetched, min(_STAGE_C_LIT_TOPN, len(llm_grounded)),
        n_abstracts_fetched, min(_STAGE_C_ABS_TOPN, len(llm_grounded)),
    )
    _LAST_ENGINES["stage_c_evidence_by_id"] = {e["disease_id"]: e for e in evidence}
    _present_block = _terms_block(present_ids)
    _instructions = (
        "Score each candidate from 0 to 1 using only the evidence above. "
        "Use the matched HPO terms as support, contradicting HPO terms as negative "
        "evidence, and the deterministic phenotype score (summed log-LR) as the "
        "phenotype prior. Prefer a disease whose retrieved literature actually describes "
        "this patient's specific findings over a generically severe disease that merely "
        "shares non-specific features. "
        "Refer to each candidate by the block NUMBER shown in brackets — e.g. the block "
        "marked [3] is id 3. Include every candidate exactly once. "
        'Respond with JSON of the form {"ranked":[{"id":3,"priority":0.7}]} where "id" '
        'is the candidate\'s block number and "priority" is a number from 0 to 1.'
    )

    def _assemble(ev: list[dict[str, Any]]) -> str:
        return (
            f"Patient PRESENT findings:\n{_present_block}\n\n"
            "Candidate diseases with retrieved evidence (one per numbered block):\n"
            f"{_render_evidence(ev)}\n\n"
            f"{_instructions}"
        )

    _t_fit0 = time.time()
    user = _fit_stage_c_user(_assemble, evidence, _stage_c_input_budget())
    logger.info(
        "reasoner Stage C: prompt fitted in %.2fs (~%d input tokens, budget %d, %d candidate(s) in final prompt)",
        time.time() - _t_fit0, _estimate_tokens(user), _stage_c_input_budget(),
        user.count("full_summed_log_lr"),
    )
    import generesolver.config as _gcfg
    if not getattr(_gcfg, "STAGE_C_LLM_RANKING", True):
        ranked_items = []
    else:
        _t_llm0 = time.time()
        logger.info("reasoner Stage C: calling ranking LLM (max_tokens=%d)...", _STAGE_C_MAX_TOKENS)
        try:
            parsed = llm_client.complete_structured(
                _STAGE_C_SYSTEM, user, schema_model=_StageCResponse,
                max_tokens=_STAGE_C_MAX_TOKENS,
            )
            ranked_items = parsed.ranked if parsed else []
            logger.info("reasoner Stage C: ranking LLM responded in %.1fs (%d ranked)",
                        time.time() - _t_llm0, len(ranked_items))
        except Exception as exc:
            logger.warning("reasoner Stage C failed after %.1fs: %s — deterministic-only order",
                           time.time() - _t_llm0, exc)
            ranked_items = []

    llm: dict[str, dict] = {}
    for it in ranked_items:
        idx = it.id - 1
        if not (0 <= idx < len(llm_grounded)):
            continue
        did = llm_grounded[idx]["disease_id"]
        if did in llm:
            continue
        llm[did] = {"priority": max(0.0, min(1.0, float(it.priority)))}

    det_raw = {b["disease_id"]: _det_score(b) for b in grounded}

    groups: dict[tuple, list[dict]] = {}
    for b in grounded:
        groups.setdefault(_evidence_signature(b), []).append(b)

    def _rep_score(m: dict) -> tuple:
        sources = set(m.get("sources") or [])
        name = (m.get("disease") or "").lower()
        did = m.get("disease_id", "")

        sa_rank = m.get("stage_a_rank")
        return (
            -(sa_rank if sa_rank is not None else 10 ** 9),
            int("biochemical" in sources),
            int("biochemical_query_search" in sources),
            float(m.get("engine_score", 0.0) or 0.0),
            int(bool(m.get("genes"))),
            int(did.startswith("OMIM")),
            int(name != "disorder"),
            int("deficiency" in name or "acidemia" in name or "aciduria" in name),
            int(m.get("n_pmids", 0) or 0),
            did,
        )

    prelim: list[dict] = []
    for gi, members in enumerate(groups.values()):
        rep = max(members, key=_rep_score)
        tie_ids = sorted({m["disease_id"] for m in members})
        member_pairs = [{"disease_id": m["disease_id"], "name": m["disease"],
                         "genes": list(m.get("genes") or [])} for m in members]
        ps = [_clip01((llm.get(m["disease_id"]) or {}).get("priority")) for m in members]
        ps = [p for p in ps if p is not None]
        any_llm = any(m["disease_id"] in llm for m in members)
        priority = max(ps) if ps else (0.5 if any_llm else 0.0)
        prelim.append({
            "gi": gi, "rep": rep, "tie_ids": tie_ids, "member_pairs": member_pairs,
            "priority": priority, "any_llm": any_llm,
            "det_raw": det_raw[rep["disease_id"]],
        })

    prelim = _merge_subtypes_into_parent(prelim)
    dets = [p["det_raw"] for p in prelim]
    dmin, dmax = (min(dets), max(dets)) if dets else (0.0, 1.0)
    dspan = dmax - dmin

    scored: list[tuple] = []
    _LAST_SCORING.clear()
    for p in prelim:
        det_norm = (p["det_raw"] - dmin) / dspan if dspan > 1e-9 else 1.0
        promotion = p["priority"] if p["any_llm"] else 0.0
        final = det_norm + _REASONER_LLM_PROMOTE * promotion
        rep = p["rep"]
        tier = _specificity_tier(rep)
        scored.append((tier, final, rep, p["tie_ids"], p["gi"], p["member_pairs"]))
        _LAST_SCORING.append({
            "disease_id": rep["disease_id"], "name": rep["disease"], "tier": tier,
            "final": round(final, 4), "priority": round(p["priority"], 3),
            "llm_ranked": p["any_llm"], "det_norm": round(det_norm, 3),
            "lr_sum": rep["lr_sum"], "biochemical": bool(rep.get("biochemical_marker")),
            "n_tied": len(p["tie_ids"]),
        })

    scored = _similar_case_rerank(scored, present_ids)

    scored.sort(key=lambda t: (-t[1], t[0],
                               -float(t[2].get("engine_score", 0) or 0),
                               -int(t[2].get("n_pmids", 0) or 0),
                               t[4]))

    max_tier = max((t[0] for t in scored), default=0)
    eff = [final + (max_tier - tier) * _TIER_POSTTEST_BONUS for tier, final, *_ in scored]
    post = _softmax(eff, _POSTTEST_TEMP)

    ranked: list[dict[str, Any]] = []
    for (tier, final, b, tie_ids, _gi, member_pairs), pp in zip(scored[:top_n], post[:top_n]):
        r = llm.get(b["disease_id"]) or {}
        own_genes = list(b.get("genes") or [])
        llm_gene = r.get("gene")
        genes = ([llm_gene] if llm_gene in own_genes else own_genes[:3]) or own_genes[:1]
        ranked.append({
            "disease_id": b["disease_id"],
            "name": b["disease"],
            "phenotype_tie_ids": [t for t in tie_ids if t != b["disease_id"]],
            "candidate_disease_gene_pairs": member_pairs,
            "genes": genes,
            "supporting_terms": [{"hpo_id": t["hpo_id"], "log_lr": t["log_lr"]}
                                 for t in b["matched_terms"]],
            "posttest_prob": round(float(pp), 4),
            "rationale": (r.get("rationale") or r.get("why_short") or "")[:300],
            "evidence": {"lr_sum": b["lr_sum"], "n_pmids": b["n_pmids"],
                         "ot_assoc": b["ot_assoc"],
                         "biochemical": bool(b.get("biochemical_marker"))},
        })
    return ranked


_REASONER_LLM_PROMOTE = 0.4
_POSTTEST_TEMP = 0.15
_TIER_POSTTEST_BONUS = 2.0


def _softmax(scores: list[float], temp: float) -> list[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp((s - m) / temp) for s in scores]
    z = sum(exps) or 1.0
    return [e / z for e in exps]


def _evidence_signature(b: dict[str, Any]) -> tuple:
    terms = tuple(sorted((t.get("hpo_id"), round(float(t.get("log_lr", 0.0)), 2))
                         for t in (b.get("matched_terms") or [])))
    return (terms, round(float(b.get("lr_sum", 0.0)), 3), bool(b.get("biochemical_marker")))

_SUBTYPE_DELIMS = (",", ";", ":", "-", "(")
_SUBTYPE_CUES = {
    "type", "types", "due", "subtype", "form", "variant", "deficiency", "deficient",
    "responsive", "related", "associated", "cbl", "cobalamin", "vitamin", "classic",
    "classical", "infantile", "juvenile", "neonatal", "late", "early", "mild", "severe",
    "group", "complementation",
}


def _norm_disease_name(name: str) -> str:
    return " ".join((name or "").lower().split())


def _is_subtype_name(parent: str, child: str) -> bool:
    if not parent or len(child) <= len(parent) or not child.startswith(parent):
        return False
    rest = child[len(parent):]
    if rest[0] in _SUBTYPE_DELIMS:
        return True
    if rest[0] == " ":
        nxt = rest.strip().split(" ", 1)[0].strip(",;:-()")
        return nxt in _SUBTYPE_CUES or nxt.startswith("cbl")
    return False

_BREADTH_DAMP = 0.7
_BREADTH_DAMP_NEG = 0.0
_MARKER_SPEC_DOMINANT = 7
_STAGE_A_PRIOR_WEIGHT = 6.0
_STAGE_A_PRIOR_TOPN = 15
_STAGE_A_PROTECT_TOPN = 15
_COMBO_WEIGHT = 1.0
_COMBO_MIN_CLUSTER = 5

@functools.lru_cache(maxsize=1)
def _inv_index() -> dict[int, frozenset]:
    return {t: frozenset(s) for t, s in lr_score._inverted_closure().items()}


@functools.lru_cache(maxsize=1)
def _hpoa_disease_count() -> int:
    return len({d for s in _inv_index().values() for d in s}) or 1


def _combo_specificity_boost(matched_terms: list[dict]) -> float:
    inv = _inv_index()
    sets = []
    for t in matched_terms:
        try:
            ti = int(str(t.get("hpo_id")).split(":")[1])
        except (ValueError, IndexError, AttributeError):
            continue
        s = inv.get(ti)
        if s:
            sets.append(s)
    if len(sets) < 2:
        return 0.0
    combo = sets[0].intersection(*sets[1:])
    cc = len(combo)
    if cc < _COMBO_MIN_CLUSTER:
        return 0.0
    return math.log(_hpoa_disease_count() / cc)


def _specificity_tier(b: dict[str, Any]) -> int:
    import generesolver.config as _gcfg
    if not getattr(_gcfg, "SPECIFICITY_TIER_ENABLED", True):
        return 0
    marker = b.get("biochemical_marker")
    if marker:
        best_spec = min(int(m.get("n_diseases", 99) or 99) for m in marker)
        if best_spec <= _MARKER_SPEC_DOMINANT:
            return 0
    return 1

_LAST_SCORING: list[dict] = []
_LAST_ENGINES: dict[str, Any] = {}


def _clip01(v) -> float | None:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return None


def _det_score(b: dict[str, Any]) -> float:
    import generesolver.config as _gcfg
    lr_sum = float(b.get("lr_sum", 0.0))
    if getattr(_gcfg, "SPECIFICITY_TIER_ENABLED", True):
        matched = b.get("matched_terms") or []
        peak = max((float(t.get("log_lr", 0.0)) for t in matched), default=0.0)
        resid = lr_sum - peak
        score = peak + (_BREADTH_DAMP if resid >= 0 else _BREADTH_DAMP_NEG) * resid
    else:
        score = lr_sum

    marker = b.get("biochemical_marker")
    if marker:
        best_spec = min(int(m.get("n_diseases", 99) or 99) for m in marker)
        score += max(
            0.0,
            float(biochemical.MAX_DISEASES_PER_MARKER + 1 - best_spec),
        )
    score += _COMBO_WEIGHT * _combo_specificity_boost(b.get("matched_terms") or [])

    sa_rank = b.get("stage_a_rank")
    if sa_rank is not None and sa_rank <= _STAGE_A_PRIOR_TOPN:
        score += _STAGE_A_PRIOR_WEIGHT * (_STAGE_A_PRIOR_TOPN - sa_rank + 1) / _STAGE_A_PRIOR_TOPN

    sources = set(b.get("sources") or [])
    engine_score = float(b.get("engine_score", 0.0) or 0.0)

    if "biochemical" in sources:
        score += 4.0

    if "biochemical_query_search" in sources:
        score += 2.0 * min(1.0, engine_score)

    if "hpo_overlap" in sources:
        score += 2.0 * min(1.0, engine_score)

    if "similar_case" in sources:
        score += 2.0 * min(1.0, engine_score)

    if not sources:
        score -= 2.0

    return score


_PCF_TOP_N = 30


def _gather_engine_candidates(present_ids: list[str]) -> list[dict]:
    by_id: dict[str, dict] = {}

    def _add(c: dict, source: str, markers=None) -> None:
        did = c.get("disease_id")
        if not did:
            return
        e = by_id.setdefault(did, {"disease_id": did, "name": c.get("name") or did,
                                   "genes": [], "markers": None, "sources": [],
                                   "engine_score": 0.0})
        if c.get("name") and (not e["name"] or e["name"] == did):
            e["name"] = c["name"]
        for g in (c.get("genes") or []):
            if g not in e["genes"]:
                e["genes"].append(g)
        if markers:
            e["markers"] = markers
        if source not in e["sources"]:
            e["sources"].append(source)
        rk = c.get("rank")
        if rk:
            e["engine_score"] = max(float(e.get("engine_score", 0.0)), 1.0 / float(rk))
        raw_es = c.get("engine_score")
        if raw_es is not None:
            try:
                e["engine_score"] = max(float(e.get("engine_score", 0.0)), float(raw_es))
            except (TypeError, ValueError):
                pass

    fails = _LAST_ENGINES.setdefault("failed", [])
    bio_cands: list[dict] = []
    try:
        bio_cands = biochemical.marker_diseases(present_ids)
        for c in bio_cands:
            _add(c, "biochemical", markers=c.get("markers"))
    except Exception as exc:
        logger.warning("biochemical agent failed: %s", exc)
        fails.append("biochemical")
    try:
        for c in _biochemical_query_search_candidates(bio_cands, present_ids):
            _add(c, "biochemical_query_search", markers=c.get("markers"))
    except Exception as exc:
        logger.warning("biochemical query-search failed: %s", exc)
        fails.append("biochemical_query_search")
    if _LR_SEED_TOP_N:
        try:
            ranked = lr_score.rank_diseases(present_ids, top_n=_LR_SEED_PARENT_SCAN)
            norms = [_norm_disease_name(c.get("name", "")) for c in ranked]
            seeded_names: list[str] = []
            for i, c in enumerate(ranked[:_LR_SEED_TOP_N], 1):
                c = dict(c)
                c["rank"] = i
                _add(c, "lr_seed")
                seeded_names.append(norms[i - 1])
            for j in range(_LR_SEED_TOP_N, len(ranked)):
                pj = norms[j]
                if pj and any(_is_subtype_name(pj, sn) for sn in seeded_names):
                    c = dict(ranked[j])
                    c["rank"] = j + 1
                    _add(c, "lr_seed")
        except Exception as exc:
            logger.warning("LR-seed failed: %s", exc)
            fails.append("lr_seed")

    if getattr(__import__("generesolver.config", fromlist=["SIMILAR_CASE_ENABLED"]),
               "SIMILAR_CASE_ENABLED", False):
        try:
            for c in similar_case.ranked_diseases(present_ids):
                _add(c, "similar_case")
        except Exception as exc:
            logger.warning("similar-case recall failed: %s", exc)
            fails.append("similar_case")

    return list(by_id.values())


def _lr_pubcasefinder_fusion(
    present_ids: list[str], absent_ids: list[str] | None, top_n: int, *, k: int = 60,
) -> list[dict[str, Any]]:
    lr_ranked = lr_score.rank_diseases(present_ids, absent_ids, top_n=200)
    try:
        pcf = pubcasefinder.ranked_diseases(present_ids, top_n=_PCF_TOP_N)
    except Exception as exc:
        logger.warning("pubcasefinder fusion: recall failed (%s); LR only", exc)
        pcf = []
    fused: dict[str, float] = {}
    meta: dict[str, dict] = {}
    for r, d in enumerate(lr_ranked, 1):
        did = d.get("disease_id")
        if not did:
            continue
        fused[did] = fused.get(did, 0.0) + 1.0 / (k + r)
        meta.setdefault(did, d)
    for r, d in enumerate(pcf, 1):
        did = d.get("disease_id")
        if not did:
            continue
        fused[did] = fused.get(did, 0.0) + 1.0 / (k + r)
        meta.setdefault(did, {"disease_id": did, "name": d.get("name") or did,
                              "genes": d.get("genes") or [], "log_lr": None,
                              "posttest_prob": 0.0, "sources": ["pubcasefinder"],
                              "supporting_terms": []})
    order = sorted(fused, key=lambda x: -fused[x])
    total = sum(fused.values()) or 1.0
    out: list[dict[str, Any]] = []
    for did in order[:top_n]:
        d = dict(meta[did])
        d["posttest_prob"] = round(fused[did] / total, 4)
        d["fusion"] = "lr+pubcasefinder"
        out.append(d)
    _LAST_ENGINES["ranked_ids"] = [d["disease_id"] for d in out]
    _LAST_ENGINES["fusion"] = "lr+pubcasefinder"
    return out

_CLUSTER_SCAN_TOPN = 8
_CLUSTER_MIN_JACCARD = 0.4
_CLUSTER_MAX_SIZE = 5

_PLAYOFF_REASONING_CHARS = 150

_PLAYOFF_SYSTEM = (
    "You are a clinical geneticist doing a focused differential between a SMALL number of "
    "very similar rare diseases that share most of their presentation. Read each candidate's "
    "evidence carefully and identify the SPECIFIC detail (a distinguishing finding, an "
    "enzyme/gene mechanism, a contradicted finding) that discriminates between them. Commit "
    "to the ONE candidate whose evidence fits best. Never use fame or outside memory — only "
    "the evidence shown. Be terse."
)

_PLAYOFF_MAGNITUDE_CAVEAT_TEXT = (
    "\n\nIMPORTANT — a candidate whose rarer mechanism can explain EVERY listed finding is NOT "
    "automatically the best fit. Weigh the MAGNITUDE and SPECIFICITY of each candidate's "
    "strongest matched finding (its log-likelihood-ratio, and the candidate's own summed "
    "deterministic phenotype score) over how many findings its mechanism can account for. A "
    "common disease whose SINGLE marker is a dominant, high-specificity match can outweigh a "
    "rarer condition that merely explains a broader set of secondary or lower-specificity "
    "findings — breadth of explanation is not evidence of correctness."
)


class _PlayoffResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    winner_id: int
    reasoning: str = ""


def _find_confusable_cluster(ranked: list[dict[str, Any]]) -> list[int] | None:
    if not ranked:
        return None
    scan = ranked[:_CLUSTER_SCAN_TOPN]
    top_terms = {t.get("hpo_id") for t in (scan[0].get("supporting_terms") or [])}
    top_genes = set(scan[0].get("genes") or [])
    cluster = [0]
    for i, cand in enumerate(scan[1:], 1):
        terms = {t.get("hpo_id") for t in (cand.get("supporting_terms") or [])}
        genes = set(cand.get("genes") or [])
        union = top_terms | terms
        jaccard = len(top_terms & terms) / len(union) if union else 0.0
        if jaccard >= _CLUSTER_MIN_JACCARD or (top_genes and genes and top_genes & genes):
            cluster.append(i)
        if len(cluster) >= _CLUSTER_MAX_SIZE:
            break
    return cluster if len(cluster) >= 2 else None


def _run_cluster_playoff(
    ranked: list[dict[str, Any]],
    cluster_idx: list[int],
    present_ids: list[str],
    absent_ids: list[str],
) -> str | None:
    members = [ranked[i] for i in cluster_idx]
    ev_by_id: dict[str, dict] = _LAST_ENGINES.get("stage_c_evidence_by_id") or {}
    member_pmids = [list((ev_by_id.get(d["disease_id"]) or {}).get("pmids") or []) for d in members]
    pmid_counts: dict[str, int] = {}
    for pmids in member_pmids:
        for pmid in set(pmids):
            pmid_counts[pmid] = pmid_counts.get(pmid, 0) + 1
    shared_pmids = {pmid for pmid, n in pmid_counts.items() if n >= 2}

    evidence: list[dict[str, Any]] = []
    for d, pmids in zip(members, member_pmids):
        did = d["disease_id"]
        e = dict(ev_by_id.get(did) or {})
        e.setdefault("disease_id", did)
        e.setdefault("disease", d.get("name"))
        e.setdefault("genes", d.get("genes") or [])
        e.setdefault("matched_hpo_terms", d.get("supporting_terms") or [])
        if "full_summed_log_lr" not in e:
            e["full_summed_log_lr"] = (d.get("evidence") or {}).get("lr_sum")

        unique_pmids = [p for p in pmids if p not in shared_pmids]
        if shared_pmids & set(pmids):
            try:
                if unique_pmids:
                    e["literature_titles"] = [t[:140] for t in literature.pubmed_titles(
                        tuple(unique_pmids), max_n=_STAGE_C_LIT_TITLES)]
                    e["literature_abstracts"] = literature.pubmed_abstracts(
                        tuple(unique_pmids), max_n=_STAGE_C_ABS_PER, max_chars=_STAGE_C_ABS_CHARS)
                else:
                    e.pop("literature_titles", None)
                    e.pop("literature_abstracts", None)
            except Exception:
                pass
        evidence.append(e)

    present_block = _terms_block(present_ids)
    absent_block = _terms_block(absent_ids) if absent_ids else ""
    instructions = (
        "These candidates share most of their evidence — that is exactly why they are shown "
        "together. Identify what SPECIFICALLY discriminates them (a finding one has and the "
        "others lack, a contradicted finding, a mechanism only one explains) and commit to "
        "the single best fit. Refer to it by its block NUMBER.\n"
        f"reasoning: ONE short clause, AT MOST {_PLAYOFF_REASONING_CHARS} characters. Be terse.\n"
        'Respond with JSON of the form {"winner_id":1,"reasoning":""}'
    )

    def _assemble(ev: list[dict[str, Any]]) -> str:
        return (
            f"Patient PRESENT findings:\n{present_block}\n\n"
            + (f"Patient ABSENT findings:\n{absent_block}\n\n" if absent_block else "")
            + "Candidate diagnoses (one per numbered block):\n"
            + f"{_render_evidence(ev)}\n\n"
            + instructions
        )

    user = _fit_stage_c_user(_assemble, evidence, _stage_c_input_budget())

    import generesolver.config as _gcfg
    system = _PLAYOFF_SYSTEM
    if getattr(_gcfg, "CLUSTER_PLAYOFF_MAGNITUDE_CAVEAT", False):
        system = system + _PLAYOFF_MAGNITUDE_CAVEAT_TEXT

    try:
        parsed = llm_client.complete_structured(
            system, user, schema_model=_PlayoffResponse, max_tokens=2048,
        )
    except Exception as exc:
        logger.warning("cluster playoff failed: %s — no override", exc)
        return None
    if parsed is None:
        return None
    idx = int(parsed.winner_id) - 1
    if not (0 <= idx < len(members)):
        logger.warning("cluster playoff: winner_id %s out of range — no override",
                       parsed.winner_id)
        return None
    _LAST_ENGINES["cluster_playoff"] = {
        "cluster": [m["disease_id"] for m in members],
        "winner": members[idx]["disease_id"],
        "reasoning": parsed.reasoning,
    }
    return members[idx]["disease_id"]


def _apply_cluster_playoff(
    ranked: list[dict[str, Any]],
    present_ids: list[str],
    absent_ids: list[str],
) -> list[dict[str, Any]]:
    cluster_idx = _find_confusable_cluster(ranked)
    if not cluster_idx:
        return ranked
    winner_id = _run_cluster_playoff(ranked, cluster_idx, present_ids, absent_ids)
    if not winner_id:
        return ranked
    pos = next((i for i, d in enumerate(ranked) if d["disease_id"] == winner_id), None)
    if pos is None or pos == 0:
        return ranked
    winner = ranked[pos]
    top_pp = ranked[0].get("posttest_prob")
    winner_pp = winner.get("posttest_prob")
    if isinstance(top_pp, (int, float)) and isinstance(winner_pp, (int, float)):
        ranked[0]["posttest_prob"], winner["posttest_prob"] = winner_pp, top_pp
    return [winner] + ranked[:pos] + ranked[pos + 1:]

_CONCLUDE_SYSTEM = (
    "You are the lead clinical geneticist signing out a diagnosis. Several tools have "
    "already retrieved, per candidate disease, the patient's matched/contradicting HPO "
    "terms, a deterministic phenotype score, pathognomonic biochemical markers, and the "
    "actual PubMed literature. Weigh ALL of it together and COMMIT to the single most "
    "likely diagnosis. Choose ONLY from the numbered candidates. Prefer the candidate "
    "whose SPECIFIC evidence — a near-pathognomonic biochemical marker, or literature that "
    "describes THIS patient's exact findings — best explains the case, over one that merely "
    "shares generic/severe features. Never use fame or outside memory."
)

_CONCLUDE_ANNOTATION_CAVEAT_TEXT = (
    "\n\nIMPORTANT — the deterministic phenotype score (summed log-LR) reflects DATABASE "
    "ANNOTATION COMPLETENESS, not true likelihood: a disease can score LOW simply because the "
    "curated database doesn't annotate it with this patient's specific findings (especially "
    "biochemical/enzyme-activity findings), while a WRONG but well-annotated relative (e.g. a "
    "cofactor-defect sibling of the true enzyme defect) scores artificially HIGH by matching "
    "only generic/shared features. Do NOT default to the highest-scoring candidate — when a "
    "lower-scoring candidate's SPECIFIC evidence (a rare finding, a named enzyme/gene match) "
    "fits better, choose it."
)

_CONCLUDE_CONFIDENCE = {"high", "moderate", "low", "uncertain"}


class _ConclusionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chosen_id: int
    confidence: str
    discriminating_evidence: str = ""
    confirmatory_test: str = ""
    alternative_ids: list[int] = Field(default_factory=list)
    flag_review: bool = True


def _conclude_diagnosis(
    ranked: list[dict[str, Any]],
    present_ids: list[str],
    absent_ids: list[str],
) -> dict[str, Any] | None:
    from generesolver.config import REASONING_LEAD_TOP_K

    top = ranked[: max(1, int(REASONING_LEAD_TOP_K))]
    if not top:
        return None

    ev_by_id: dict[str, dict] = _LAST_ENGINES.get("stage_c_evidence_by_id") or {}
    evidence: list[dict[str, Any]] = []
    for d in top:
        did = d["disease_id"]
        e = dict(ev_by_id.get(did) or {})
        e.setdefault("disease_id", did)
        e.setdefault("disease", d.get("name"))
        e.setdefault("genes", d.get("genes") or [])
        e.setdefault("matched_hpo_terms", d.get("supporting_terms") or [])
        if "full_summed_log_lr" not in e:
            e["full_summed_log_lr"] = (d.get("evidence") or {}).get("lr_sum")
        evidence.append(e)

    present_block = _terms_block(present_ids)
    absent_block = _terms_block(absent_ids) if absent_ids else ""
    instructions = (
        "Commit to ONE diagnosis. Refer to it by its block NUMBER. Give a confidence of "
        "high, moderate, low, or uncertain; a one-sentence discriminating_evidence citing "
        "the block's findings; the single most decisive confirmatory_test; and up to 2 "
        "alternative block numbers. Set flag_review true unless the evidence clearly "
        "favours one candidate.\n"
        'Respond with JSON of the form {"chosen_id":1,"confidence":"moderate",'
        '"discriminating_evidence":"","confirmatory_test":"","alternative_ids":[2],'
        '"flag_review":true}'
    )

    def _assemble(ev: list[dict[str, Any]]) -> str:
        return (
            f"Patient PRESENT findings:\n{present_block}\n\n"
            + (f"Patient ABSENT findings:\n{absent_block}\n\n" if absent_block else "")
            + "Candidate diagnoses with retrieved evidence (one per numbered block):\n"
            + f"{_render_evidence(ev)}\n\n"
            + instructions
        )

    user = _fit_stage_c_user(_assemble, evidence, _stage_c_input_budget())

    import generesolver.config as _gcfg
    system = _CONCLUDE_SYSTEM
    if getattr(_gcfg, "CONCLUDE_ANNOTATION_CAVEAT", False):
        system = system + _CONCLUDE_ANNOTATION_CAVEAT_TEXT

    try:
        parsed = llm_client.complete_structured(
            system, user, schema_model=_ConclusionResponse,
            max_tokens=_STAGE_C_MAX_TOKENS,
        )
    except Exception as exc:
        logger.warning("reasoning-lead conclusion failed: %s — no verdict", exc)
        return None
    if parsed is None:
        return None

    idx = int(parsed.chosen_id) - 1
    if not (0 <= idx < len(top)):
        logger.warning("reasoning-lead conclusion: chosen_id %s out of range — no verdict",
                       parsed.chosen_id)
        return None

    chosen = top[idx]
    confidence = parsed.confidence.strip().lower()
    if confidence not in _CONCLUDE_CONFIDENCE:
        confidence = "uncertain"
    alternatives = []
    for aid in parsed.alternative_ids[:2]:
        ai = int(aid) - 1
        if 0 <= ai < len(top) and ai != idx:
            alternatives.append({"disease_id": top[ai]["disease_id"], "name": top[ai].get("name")})

    conclusion = {
        "most_likely_disease_id": chosen["disease_id"],
        "most_likely_disease_name": chosen.get("name"),
        "genes": list(chosen.get("genes") or []),
        "confidence": confidence,
        "discriminating_evidence": (parsed.discriminating_evidence or "").strip()[:400],
        "confirmatory_test": (parsed.confirmatory_test or "").strip()[:200],
        "alternatives": alternatives,
        "flag_review": bool(parsed.flag_review),
        "adjudicated_slate_ids": [d["disease_id"] for d in top],
    }
    _LAST_ENGINES["conclusion"] = conclusion
    return conclusion


def _apply_decisive_conclusion(
    ranked: list[dict[str, Any]],
    conclusion: dict[str, Any],
) -> list[dict[str, Any]]:
    chosen_id = conclusion.get("most_likely_disease_id")
    pos = next((i for i, d in enumerate(ranked) if d["disease_id"] == chosen_id), None)
    if pos is None or pos == 0:
        return ranked
    promoted = ranked[pos]
    top_pp = ranked[0].get("posttest_prob")
    promoted_pp = promoted.get("posttest_prob")
    if isinstance(top_pp, (int, float)) and isinstance(promoted_pp, (int, float)):
        ranked[0]["posttest_prob"], promoted["posttest_prob"] = promoted_pp, top_pp
    reordered = [promoted] + ranked[:pos] + ranked[pos + 1:]
    return reordered


def _llm_names_only(
    present_ids: list[str],
    absent_ids: list[str],
    top_n: int,
) -> list[dict[str, Any]]:
    names = _stage_a_hypothesize(present_ids, absent_ids or [])
    _LAST_ENGINES["stage_a_llm"] = list(names)
    _LAST_ENGINES["mode"] = "llm_names_only"
    top = names[:top_n]
    n = len(top) or 1
    out: list[dict[str, Any]] = [
        {"disease_id": "", "name": nm, "genes": [],
         "posttest_prob": round((n - i) / n, 4), "supporting_terms": []}
        for i, nm in enumerate(top)
    ]
    _LAST_ENGINES["ranked_ids"] = ["" for _ in out]
    logger.info("reasoner[llm_names_only]: returned %d LLM-named diseases", len(out))
    return out


_RANK_NAMES_SYSTEM = (
    "You are a consultant clinical geneticist giving a precise, decisive differential "
    "diagnosis for a rare-disease case. Commit to your single best answer first."
)


class _RankedNamesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    diagnoses: list[str] = Field(default_factory=list)


def _llm_ranked_names(
    present_ids: list[str],
    absent_ids: list[str],
    top_n: int,
) -> list[dict[str, Any]]:

    user = (
        "A patient presents with the following standardised (HPO) phenotype.\n\n"
        f"PRESENT findings:\n{_terms_block(present_ids)}\n"
        + (f"\nEXPLICITLY ABSENT findings:\n{_terms_block(absent_ids)}\n" if absent_ids else "")
        + "\nGive your RANKED differential diagnosis. The FIRST entry MUST be the SINGLE most "
        "likely diagnosis — your single best answer — then the next most likely, and so on. "
        "Be decisive.\n"
        "Name SPECIFIC diseases exactly as catalogued in OMIM / Orphanet — NOT umbrella "
        "categories, groups, or pathways. WRONG: 'Urea cycle disorders', 'Organic acidemias', "
        "'Mitochondrial disorders'. RIGHT: 'Ornithine transcarbamylase deficiency', "
        "'Isovaleric acidemia', 'Mitochondrial complex IV deficiency'. For an inborn error of "
        "metabolism, name the specific enzyme deficiency and subtype (full disease name, not "
        "the gene, not a bare acronym).\n"
        f"Return up to {top_n} diagnoses, most likely FIRST.\n"
        'Return ONLY JSON: {"diagnoses": ["<most likely>", "<next most likely>", ...]}'
    )
    try:
        parsed = llm_client.complete_structured(
            _RANK_NAMES_SYSTEM, user, schema_model=_RankedNamesResponse,
            max_tokens=_STAGE_A_MAX_TOKENS,
        )
        raw = list(parsed.diagnoses) if parsed else []
    except Exception as exc:
        logger.warning("reasoner[llm_ranked_only]: LLM failed (%s)", exc)
        raw = []

    out_names: list[str] = []
    seen: set[str] = set()
    for nm in raw:
        nm = (nm or "").strip()
        k = nm.lower()
        if nm and k not in seen:
            seen.add(k)
            out_names.append(nm)
    top = out_names[:top_n]
    n = len(top) or 1
    out: list[dict[str, Any]] = [
        {"disease_id": "", "name": nm, "genes": [],
         "posttest_prob": round((n - i) / n, 4), "supporting_terms": []}
        for i, nm in enumerate(top)
    ]
    _LAST_ENGINES["mode"] = "llm_ranked_only"
    _LAST_ENGINES["stage_a_llm"] = list(top)
    _LAST_ENGINES["ranked_ids"] = ["" for _ in out]
    logger.info("reasoner[llm_ranked_only]: returned %d LLM-ranked diseases", len(out))
    return out


def rank_differential(
    present_ids: list[str],
    absent_ids: list[str] | None = None,
    *,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    present_ids = [h for h in (present_ids or []) if h]
    absent_ids = [h for h in (absent_ids or []) if h]
    if not present_ids:
        return []

    _LAST_ENGINES.clear()
    _LAST_ENGINES["failed"] = []

    if getattr(__import__("generesolver.config", fromlist=["LLM_NAMES_ONLY"]),
               "LLM_NAMES_ONLY", False):
        return _llm_names_only(present_ids, absent_ids, top_n)

    if getattr(__import__("generesolver.config", fromlist=["LLM_RANKED_ONLY"]),
               "LLM_RANKED_ONLY", False):
        return _llm_ranked_names(present_ids, absent_ids, top_n)

    if getattr(__import__("generesolver.config", fromlist=["DX_PIPELINE"]), "DX_PIPELINE", False):
        from generesolver.dx import pipeline as _dx
        return _dx.diagnose(present_ids, absent_ids, top_n=top_n)

    if getattr(__import__("generesolver.config", fromlist=["DX_AGENTIC"]), "DX_AGENTIC", False):
        from generesolver.dx import agent as _dx_agent
        return _dx_agent.diagnose_agentic(present_ids, absent_ids, top_n=top_n)

    if getattr(__import__("generesolver.config", fromlist=["PUBCASEFINDER_ENABLED"]),
               "PUBCASEFINDER_ENABLED", False):
        return _lr_pubcasefinder_fusion(present_ids, absent_ids, top_n)

    names = _stage_a_hypothesize(present_ids, absent_ids)
    logger.info("reasoner: Stage A proposed %d candidate diseases", len(names))
    _LAST_ENGINES["stage_a_llm"] = list(names)

    engine_candidates = _gather_engine_candidates(present_ids)

    bio_engine = [
        c for c in engine_candidates
        if "biochemical" in set(c.get("sources") or [])
    ]

    logger.info("reasoner: recall engines surfaced %d candidate(s)", len(engine_candidates))
    by_source: dict[str, list[str]] = {}
    for c in engine_candidates:
        for s in (c.get("sources") or []):
            by_source.setdefault(s, []).append(c["disease_id"])
    _LAST_ENGINES["by_source"] = by_source

    if not names and not engine_candidates:
        return []

    grounded = _stage_b_ground(
        names,
        present_ids,
        absent_ids,
        engine_candidates=engine_candidates,
    )
    logger.info("reasoner: Stage B grounded %d candidates (%d from engines)",
                len(grounded), len(engine_candidates))
    _LAST_ENGINES["grounded_ids"] = [b["disease_id"] for b in grounded]
    if not grounded:
        return []

    ranked = _stage_c_rank(grounded, present_ids, top_n)
    logger.info("reasoner: Stage C ranked %d diseases", len(ranked))
    _LAST_ENGINES["ranked_ids"] = [d["disease_id"] for d in ranked]

    import generesolver.config as _gcfg
    playoff_resolved = False
    if getattr(_gcfg, "CLUSTER_PLAYOFF_ENABLED", False) and ranked:
        ranked = _apply_cluster_playoff(ranked, present_ids, absent_ids)
        _LAST_ENGINES["ranked_ids"] = [d["disease_id"] for d in ranked]
        playoff_resolved = bool(_LAST_ENGINES.get("cluster_playoff"))

    if getattr(_gcfg, "REASONING_LEAD_CONCLUSION", False) and ranked:
        conclusion = _conclude_diagnosis(ranked, present_ids, absent_ids)
        if (conclusion and getattr(_gcfg, "REASONING_LEAD_DECISIVE", False)
                and not playoff_resolved):
            ranked = _apply_decisive_conclusion(ranked, conclusion)
            _LAST_ENGINES["ranked_ids"] = [d["disease_id"] for d in ranked]

    return ranked
