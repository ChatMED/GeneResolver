from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from generesolver.dx import tools, unified_scorer
from generesolver.integrations import hpo_db, hpo_a

logger = logging.getLogger(__name__)

_MAX_TURNS = 8
_SEED_N = 80
_EVIDENCE_DETAIL_N = 25

_SYSTEM = (
    "You are a clinical geneticist diagnosing a rare disease. You do NOT recall database codes "
    "from memory, so you MUST look up every disease you seriously consider — your final answer "
    "may only contain diseases you have looked up.\n\n"
    "Respond every turn with EXACTLY ONE JSON object and nothing else:\n"
    '  {"action":"search_disease","query":"<disease name or synonym>"}   look a disease up by name\n'
    '  {"action":"search_by_gene","gene":"<symbol>"}                      diseases caused by a gene\n'
    '  {"action":"get_disease_profile","disease_id":"<OMIM/ORPHA id>"}    full profile to check fit\n'
    '  {"action":"final","ranked":["<disease name>", ...]}               your ranked differential\n\n'
    "Method: each finding is annotated with how many diseases it characterises. ANCHOR on the "
    "SPECIFIC findings (few diseases) — a finding characterising thousands barely narrows the "
    "diagnosis. From those specific findings, DIAGNOSE FIRST: use your medical knowledge to "
    "name the actual disease (or the enzyme/gene it implicates) BEFORE searching. Then ground "
    "that diagnosis with search_disease or search_by_gene (use the exact names the tool "
    "returns). NEVER search a patient FINDING verbatim as if it were a disease name (e.g. do "
    "NOT search 'Alopecia' or 'Skin rash' or 'Increased lactate' — these are symptoms, not "
    "diagnoses, and searching them wastes a turn and finds nothing useful). Do NOT repeat a "
    "search. When ready, give final: up to 10 diseases, most likely FIRST, using "
    "only names you looked up."
)


def _spec(hpo_id: str) -> int:
    try:
        return len(hpo_a.get_hpo_to_diseases_map().get(int(hpo_id.split(":")[1])) or ())
    except (ValueError, IndexError):
        return 0


def _label(hpo_id: str) -> str:
    try:
        return str(hpo_db._get_ontology().get_hpo_object(int(hpo_id.split(":")[1])).name)
    except Exception:
        return hpo_id


def _norm(s: str) -> str:
    s = re.sub(r"\(.*?\)", " ", (s or "").lower())
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _parse_action(txt: str) -> dict | None:
    start = txt.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(txt)):
            ch = txt[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(txt[start:i + 1])
                        except json.JSONDecodeError:
                            break
        start = txt.find("{", start + 1)
    return None


def _client_model():
    from openai import OpenAI
    base = os.environ.get("GENERESOLVER_MEDGEMMA_BASE_URL", "")
    model = os.environ.get("GENERESOLVER_MEDGEMMA_MODEL", "google/medgemma-27b-it")
    key = os.environ.get("GENERESOLVER_MEDGEMMA_API_KEY", "not-needed")
    return OpenAI(base_url=base, api_key=key), model


def _to_int(hpo_id: str) -> int | None:
    try:
        return int(str(hpo_id).split(":")[1])
    except (ValueError, IndexError):
        return None


def diagnose_agentic(present_ids: list[str], absent_ids: list[str] | None = None,
                     *, top_n: int = 10, max_turns: int = _MAX_TURNS,
                     trace: list | None = None) -> list[dict[str, Any]]:
    present_ids = [h for h in (present_ids or []) if h]
    absent_ids = [h for h in (absent_ids or []) if h]
    if not present_ids:
        return []
    client, model = _client_model()

    present_ints = {i for i in (_to_int(h) for h in present_ids) if i is not None}
    absent_ints = {i for i in (_to_int(h) for h in absent_ids) if i is not None}
    case_tools = tools.build_case_tools(present_ints, absent_ints)
    findings = "\n".join(
        f"- {_label(h)} (characterises {_spec(h)} diseases)"
        for h in sorted(present_ids, key=_spec)
    )
    absent_block = ""
    if absent_ids:
        absent_block = "\nEXPLICITLY ABSENT (rule out diseases that would cause these):\n" + \
            "\n".join(f"- {_label(h)}" for h in absent_ids)

    grounded: dict[str, dict] = {}
    log_lr_by_id: dict[str, float] = {}

    try:
        seeded = unified_scorer.score_diseases(present_ids, absent_ids, top_n=_SEED_N)
    except Exception as exc:
        logger.warning("agent: unified_scorer seeding failed: %s", exc)
        seeded = []
    for s in seeded:
        grounded[_norm(s.name)] = {"disease_id": s.disease_id, "name": s.name, "genes": s.genes}
        log_lr_by_id[s.disease_id] = s.log_lr
    seeded_ids = {s.disease_id for s in seeded}
    if trace is not None:
        trace.append({"turn": "seed", "n_seeded": len(seeded)})

    user = (
        f"Patient findings:\n{findings}{absent_block}\n\n"
        f"A database scorer has already pre-screened {len(seeded)} candidate diseases with any "
        "phenotype or gene evidence for this patient (you'll see them in an evidence table before "
        "you rank). You do NOT need to search for those to ground them — they're already grounded. "
        "Use search_disease / search_by_gene only to ADD a disease you believe is likely from your "
        "own medical knowledge that might be missing or under-scored. Then give your final ranked "
        "differential."
    )
    messages = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user}]

    def _record(result):
        for r in (result if isinstance(result, list) else [result]):
            if isinstance(r, dict) and r.get("disease_id"):
                grounded[_norm(r["name"])] = {
                    "disease_id": r["disease_id"], "name": r["name"], "genes": r.get("genes") or []}

    for turn in range(max_turns):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=400, temperature=0)
            txt = resp.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("agent chat failed: %s", exc)
            break
        act = _parse_action(txt)
        messages.append({"role": "assistant", "content": txt})
        if trace is not None:
            trace.append({"turn": turn + 1, "raw": txt[:300], "action": act})

        if act is None:
            messages.append({"role": "user",
                             "content": "That was not a single valid JSON action. Send ONE JSON action now."})
            continue
        action = act.get("action")
        if action == "final":
            return _confirm_and_finalize(client, model, messages, act.get("ranked") or [],
                                          grounded, present_ints, absent_ints, top_n, trace,
                                          log_lr_by_id, seeded_ids)
        fn = case_tools.get(action)
        if fn is None:
            messages.append({"role": "user",
                             "content": "Unknown action. Use search_disease, search_by_gene, get_disease_profile, or final."})
            continue
        try:
            result = fn(act)
        except Exception as exc:
            result = [{"error": str(exc)}]
        _record(result)
        left = max_turns - turn - 1
        if left <= 2:
            nudge = (f"\n(Only {left} turn(s) left and you have grounded {len(grounded)} disease(s)"
                     ' — start concluding: respond {"action":"final","ranked":[...]} soon.)')
        else:
            nudge = f"\n(Grounded {len(grounded)} disease(s). Give final when your differential is ready.)"
        messages.append({"role": "user", "content": "TOOL RESULT:\n" + json.dumps(result)[:1600] + nudge})

    if grounded:
        table = _build_evidence_table(grounded, present_ints, absent_ints, log_lr_by_id, seeded_ids)
        fmsgs = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content":
                f"Patient findings:\n{findings}\n\n{table}\n\n"
                "Give your final ranked differential now, most likely FIRST. JSON only: "
                '{"action":"final","ranked":["<name>", ...]}'},
        ]
        try:
            resp = client.chat.completions.create(
                model=model, messages=fmsgs, max_tokens=400, temperature=0)
            reply_txt = resp.choices[0].message.content or ""
            act = _parse_action(reply_txt)
            if trace is not None:
                trace.append({"turn": "finalize", "action": act})
            if act and act.get("ranked"):
                fmsgs.append({"role": "assistant", "content": reply_txt})
                return _confirm_and_finalize(client, model, fmsgs, act["ranked"], grounded,
                                              present_ints, absent_ints, top_n, trace,
                                              log_lr_by_id, seeded_ids)
        except Exception as exc:
            logger.warning("agent finalize call failed: %s", exc)
    return _finalize(
        sorted((g["name"] for g in grounded.values()),
               key=lambda n: -log_lr_by_id.get(grounded[_norm(n)]["disease_id"], -999)),
        grounded, top_n)


def _build_evidence_table(grounded: dict[str, dict], present_ints: set[int],
                          absent_ints: set[int], log_lr_by_id: dict[str, float] | None = None,
                          seeded_ids: set[str] | None = None) -> str:
    from generesolver.dx import tools as _tools

    log_lr_by_id = log_lr_by_id or {}
    seeded_ids = seeded_ids or set()

    own, seed = [], []
    for g in grounded.values():
        profile = _tools.get_disease_profile(g["disease_id"], present_ints, absent_ints)
        row = (profile["n_present_matched"], -profile["n_absent_contradicted"],
               profile["name"], profile, log_lr_by_id.get(g["disease_id"]))
        (seed if g["disease_id"] in seeded_ids else own).append(row)
    own.sort(key=lambda r: (-r[0], r[1]))
    seed.sort(key=lambda r: -(r[4] if r[4] is not None else -999))

    def _detail_line(n_match, neg_contra, name, profile, llr, tag: str) -> str:
        contra = -neg_contra
        line = f"- {name}: matches {n_match} present finding(s){tag}"
        if profile["matched_present_findings"]:
            line += f" ({', '.join(profile['matched_present_findings'][:6])})"
        if contra:
            line += (f"; CONTRADICTS {contra} absent finding(s) "
                     f"({', '.join(profile['contradicted_absent_findings'][:4])}) — weigh against it")
        return line

    lines = []
    if own:
        lines.append("=== YOUR OWN HYPOTHESES (you grounded these via search — trust your "
                     "clinical reasoning here, a database score below does not override it) ===")
        for n_match, neg_contra, name, profile, llr in own:
            tag = f" | database log-LR={llr:+.2f}" if llr is not None else ""
            lines.append(_detail_line(n_match, neg_contra, name, profile, llr, tag))
    if seed:
        lines.append("\n=== DATABASE PRE-SCREEN (log-LR ranked; a LOW/NEGATIVE score can mean "
                     "incomplete annotation, NOT low likelihood — weigh alongside your own "
                     "knowledge, especially for enzyme/biochemical findings) ===")
        for i, (n_match, neg_contra, name, profile, llr) in enumerate(seed):
            tag = f" | log-LR={llr:+.2f}"
            if i < _EVIDENCE_DETAIL_N:
                lines.append(_detail_line(n_match, neg_contra, name, profile, llr, tag))
            else:
                lines.append(f"- {name}{tag}, {n_match} matched")
    return "\n".join(lines)


def _confirm_and_finalize(client, model, messages: list[dict], proposed_ranked: list[str],
                          grounded: dict[str, dict], present_ints: set[int], absent_ints: set[int],
                          top_n: int, trace: list | None,
                          log_lr_by_id: dict[str, float] | None = None,
                          seeded_ids: set[str] | None = None) -> list[dict[str, Any]]:
    if not grounded:
        return _finalize(proposed_ranked, grounded, top_n)

    table = _build_evidence_table(grounded, present_ints, absent_ints, log_lr_by_id, seeded_ids)
    messages.append({"role": "user", "content":
        f"{table}\n\nYour proposed ranking was: {proposed_ranked}\n"
        "Reconsider against the evidence table above — in particular, demote anything that "
        "CONTRADICTS an absent finding, and prefer candidates matching more present findings. "
        'Give your CONFIRMED final ranking now. JSON only: {"action":"final","ranked":["<name>", ...]}'})
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, max_tokens=400, temperature=0)
        act = _parse_action(resp.choices[0].message.content or "")
        if trace is not None:
            trace.append({"turn": "confirm", "action": act, "evidence_table": table})
        if act and act.get("action") == "final" and act.get("ranked"):
            return _finalize(act["ranked"], grounded, top_n)
    except Exception as exc:
        logger.warning("agent confirm-final call failed: %s", exc)
    return _finalize(proposed_ranked, grounded, top_n)


def _finalize(ranked_names: list[str], grounded: dict[str, dict], top_n: int) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for name in ranked_names:
        k = _norm(name)
        g = grounded.get(k) or next(
            (v for kk, v in grounded.items() if kk and (kk in k or k in kk)), None)
        if g:
            if g["disease_id"] in seen:
                continue
            seen.add(g["disease_id"])
            out.append({"disease_id": g["disease_id"], "name": g["name"],
                        "genes": g["genes"], "supporting_terms": []})
        else:
            out.append({"disease_id": "", "name": name, "genes": [], "supporting_terms": []})
        if len(out) >= top_n:
            break
    n = len(out) or 1
    for i, d in enumerate(out):
        d["posttest_prob"] = round((n - i) / n, 4)
    return out