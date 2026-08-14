from __future__ import annotations

import logging

from generesolver.integrations import llm_client
from generesolver.schemas.biochem import Direction
from generesolver.schemas.case import CaseState
from generesolver.schemas.report import ProvenanceItem
from generesolver.schemas.trace import ConfirmatoryTest, TestPower

logger = logging.getLogger(__name__)

_MAX_TESTS = 6

_SYSTEM = (
    "You are a clinical geneticist deciding what to do NEXT for this patient.\n\n"
    "You are given the complete case record: the phenotype, the biochemistry, any variants, the "
    "etiology verdict, the candidate diagnoses, the mechanism that was proposed, and the full list "
    "of databases that were actually consulted.\n\n"
    "Propose the tests that would CONFIRM OR DENY the leading hypotheses. Rules:\n"
    "- Derive every test from THIS patient's findings. Do not emit a generic panel for the disease "
    "category; if a test is not indicated by something in this record, leave it out.\n"
    "- A test is only worth proposing if its result would change what the clinician does next.\n"
    "- State what a POSITIVE would establish and what a NEGATIVE would rule out. A test whose "
    "negative result means nothing is a weak test — say so.\n\n"
    "Rate each test's power literally, and do NOT inflate it:\n"
    "  decisive     — a result EITHER WAY settles it: positive confirms, negative rules out. Use "
    "sparingly; this is the test to do first.\n"
    "  supportive   — moves confidence, settles nothing on its own.\n"
    "  exclusionary — mainly useful to rule a possibility out.\n"
    "If NO single test can settle this case, say that plainly and mark them supportive. Claiming a "
    "test is decisive when it is not tells a clinician to trust a result that cannot bear the "
    "weight — that is the most harmful thing you can do here.\n\n"
    "Reply as ONE JSON object:\n"
    '{"tests": [{"test": "<specific test>", "rationale": "<why, citing this patient\'s findings>", '
    '"confirms": ["<what a positive establishes>"], "refutes": ["<what a negative rules out>"], '
    '"power": "<decisive|supportive|exclusionary>"}],\n'
    ' "reasoning": "<2-3 sentences: what is still unresolved and how this plan resolves it>"}'
)


def _record(state: CaseState) -> str:
    parts: list[str] = []

    pheno = [t.label for t in state.hpo_profile.terms
             if getattr(t, "label", None) and t.status.value == "present"]
    if pheno:
        parts.append("PHENOTYPE:\n" + "\n".join(f"- {p}" for p in pheno))

    labs = [m.summary() for m in state.biochem_panel.measurements
            if m.direction != Direction.UNKNOWN]
    if labs:
        parts.append("BIOCHEMISTRY:\n" + "\n".join(f"- {l}" for l in labs))

    demo = []
    if state.age_years is not None:
        demo.append(f"Age at presentation: {state.age_years} years")
    if state.consanguinity is not None:
        demo.append(f"Consanguinity: {state.consanguinity}")
    if state.family_history:
        demo.append(f"Family history: {state.family_history}")
    if demo:
        parts.append("DEMOGRAPHICS:\n" + "\n".join(f"- {d}" for d in demo))

    if state.acmg_assessments:
        parts.append("VARIANTS:\n" + "\n".join(
            f"- {c.computed_acmg_assessment.gene} {c.computed_acmg_assessment.variant_id}: "
            f"ACMG {c.computed_acmg_assessment.classification}"
            for c in state.acmg_assessments[:10]))
    elif state.has_vcf:
        parts.append("VARIANTS: a VCF was analysed but no variant reached an ACMG call.")
    else:
        parts.append("VARIANTS: none — NO genomic data was available for this case.")

    if state.case_class:
        cc = state.case_class
        parts.append(
            f"ETIOLOGY VERDICT: {cc.decision.value if cc.decision else 'uncertain'} "
            f"(mode={cc.mode.value}, margin={cc.margin:.2f})"
            + (f"\n{cc.conclusion}" if cc.conclusion else "")
        )
    if state.branch_result:
        br = state.branch_result
        conds = br.get("conditions") or []
        if conds:
            parts.append("CANDIDATE CONDITIONS:\n" + "\n".join(
                f"- {c.get('name')} ({c.get('likelihood', '?')}): "
                f"for={c.get('supporting', [])} against={c.get('against', [])}"
                for c in conds[:5]))
        if br.get("mechanism"):
            parts.append("PROPOSED MECHANISM:\n" + br["mechanism"])
        if br.get("limitations"):
            parts.append("GAPS IN THE INVESTIGATION (a test may be needed precisely because of "
                         "these):\n" + "\n".join(f"- {l}" for l in br["limitations"]))
    if state.ranked_genes:
        parts.append("RANKED CANDIDATE GENES:\n" + "\n".join(
            f"- {g.rank}. {g.gene} (score {g.final_score:.3f})"
            for g in state.ranked_genes[:5]))
    if state.ranked_diseases:
        top = state.ranked_diseases[:5]
        parts.append("RANKED CANDIDATE DISEASES:\n" + "\n".join(
            f"- {d.get('name', d.get('disease_id', '?'))}" for d in top
            if isinstance(d, dict)))

    prior: list[str] = []
    integ = (state.diagnostics.get("final_integrated") or {}).get(
        "final_diagnostic_hypotheses") or []
    for h in integ[:3]:
        if isinstance(h, dict):
            prior.extend(str(t) for t in (h.get("missing_decisive_tests") or []))
    conclusion = (state.diagnostics.get("reasoner_engines") or {}).get("conclusion") or {}
    if isinstance(conclusion, dict) and conclusion.get("confirmatory_test"):
        prior.append(str(conclusion["confirmatory_test"]))
    if state.differential_decision and state.differential_decision.decisive_tests:
        prior.extend(state.differential_decision.decisive_tests)
    if prior:
        parts.append("TESTS ALREADY SUGGESTED BY OTHER COMPONENTS (adopt, refine or reject "
                     "these — do not simply repeat them):\n"
                     + "\n".join(f"- {t}" for t in dict.fromkeys(prior)))

    if state.trace.retrievals:
        ok = state.trace.sources_used()
        failed = sorted({r.source for r in state.trace.retrievals if r.failed})
        line = "DATABASES CONSULTED: " + (", ".join(ok) or "none")
        if failed:
            line += f"\nDATABASES THAT FAILED (their evidence is MISSING): {', '.join(failed)}"
        parts.append(line)

    return "\n\n".join(parts)


def _power(value) -> TestPower:
    try:
        return TestPower(str(value).lower().strip())
    except ValueError:
        return TestPower.SUPPORTIVE


def run(state: CaseState) -> CaseState:
    record = _record(state)

    try:
        res = llm_client.complete_json(_SYSTEM, record, max_tokens=1000)
    except Exception as exc:
        logger.warning("Next-steps synthesis failed: %s", exc)
        state.add_warning(f"Confirmatory-test plan could not be generated: {exc}")
        state.mark_done("next_steps")
        return state

    if not isinstance(res, dict):
        state.add_warning("Confirmatory-test plan could not be generated (malformed response).")
        state.mark_done("next_steps")
        return state

    for t in (res.get("tests") or [])[:_MAX_TESTS]:
        if not isinstance(t, dict) or not t.get("test"):
            continue
        state.next_steps.tests.append(ConfirmatoryTest(
            test=str(t["test"]),
            rationale=str(t.get("rationale", "")),
            confirms=[str(x) for x in (t.get("confirms") or [])],
            refutes=[str(x) for x in (t.get("refutes") or [])],
            power=_power(t.get("power")),
            source="next_steps",
        ))
    state.next_steps.reasoning = str(res.get("reasoning", "")).strip()

    decisive = state.next_steps.decisive()
    logger.info(
        "Next steps: %d test(s), %d decisive.", len(state.next_steps.tests), len(decisive)
    )
    state.trace.add_step(
        step="next_steps",
        claim=(f"{len(state.next_steps.tests)} confirmatory test(s) proposed"
               + (f"; decisive: {decisive[0].test}" if decisive else "; none decisive")),
        grounded_in=state.trace.sources_used() or ["case record"],
    )
    state.provenance.append(ProvenanceItem(
        component="next_steps",
        tool_name="LLM confirm-or-deny plan synthesized from the full case record",
        tool_version="no canned per-class test lists",
    ))
    state.mark_done("next_steps")
    return state
