from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx

from common import LABELS, read_json, write_json
from llm_adapter import add_source_dir, complete_json
from pubmed import article_text, get_article_passages

TRANSIENT_ERRORS = (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout)

VIGNETTE_PROMPT = """From this case report about a patient with
target_disease, do two things in one response:

1. Extract patient facts using only what's reported: patient_key (short
   identifying label, e.g. "37M with..."), presentation_facts,
   family_history, exposure_history, prediagnostic_findings.
2. Write a concise first-presentation clinical case from those facts
   only. First-presentation is a first doctor visit and runned tests, 
   the treatment, final testing and differentials are not established yet. Final diagnosis
   and tests result for the final diagnosis should not be included in the vignette. 

Return a JSON object with: patient_key; presentation_facts; family_history;
exposure_history; prediagnostic_findings; and vignette (a single plain-text
string, not a nested object)."""

VERIFY_MECHANISM_PROMPT = """Read the case report and find its own final
explanation for why THIS patient has target_disease specifically -- not a
separate complication, or an incidental trigger, that merely occurred in an
already-diagnosed patient. expected_mechanism is target_disease's
established cause; keep it unless the article's own final explanation for
target_disease itself specifically differs. Pick ONE label:

single_gene (one gene's own variant or small indel -- including a deletion
confined to that one gene, e.g. one exon of it); chromosomal (an aneuploidy,
or a deletion/duplication/rearrangement spanning multiple genes or a
chromosome segment, even if inherited); somatic (postzygotic -- acquired in
only some of the patient's cells/tissues, not inherited); multifactorial
(polygenic and/or an established combination of genetic predisposition with
an environmental/infectious/autoimmune trigger); non_genetic (infectious,
toxic/drug, nutritional, mechanical, autoimmune, or other acquired cause,
not rooted in an inherited or postzygotic DNA change).

Return JSON: label; somatic_subgroup ("single_gene" or "multifactorial" --
only if label is "somatic", else ""); reason (explain the specific
mechanism/finding that justifies this label, one or two sentences)."""


def call_with_retry(fn, *args, retries: int = 5, backoff: float = 2.0, **kwargs):
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except TRANSIENT_ERRORS as exc:
            last_error = exc
            wait = backoff * (2**attempt)
            print(f"    (transient error, retry {attempt + 1}/{retries} in {wait:.1f}s: {exc})")
            time.sleep(wait)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 429:
                raise
            last_error = exc
            wait = float(exc.response.headers.get("Retry-After") or backoff * (2**attempt))
            print(f"    (429, retry {attempt + 1}/{retries} in {wait:.1f}s)")
            time.sleep(wait)
    raise last_error


def check_leakage(vignette: str, disease_name: str, synonyms: list[str]) -> tuple[bool, list[str]]:
    text = vignette.casefold()
    hits = [name for name in [disease_name, *synonyms] if name and len(name) >= 4 and name.casefold() in text]
    return bool(hits), hits


def load_cached_article(pmcid: str | None, articles_dir: Path) -> str | None:
    if not pmcid:
        return None
    path = articles_dir / f"{pmcid}.json"
    if not path.exists():
        return None
    try:
        return article_text(read_json(path))
    except Exception:
        return None


def facts_fallback_text(record: dict) -> str:
    parts = [f"Patient: {record.get('patient_key', '')}"]
    for label, key in (
        ("Presentation", "presentation_facts"),
        ("Family history", "family_history"),
        ("Exposure history", "exposure_history"),
        ("Prediagnostic findings", "prediagnostic_findings"),
    ):
        items = record.get(key) or []
        if items:
            parts.append(f"{label}: " + "; ".join(str(x) for x in items))
    if record.get("vignette"):
        parts.append(f"Vignette: {record['vignette']}")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("work/adult_primary_cases.json"))
    parser.add_argument("--diseases", type=Path, default=Path("work/citation_seed_diseases.json"))
    parser.add_argument("--output", type=Path, default=Path("output/gold_vignettes_v2.json"))
    parser.add_argument("--source-dir", type=Path, default=Path("../.."))
    parser.add_argument("--client", default="generesolver.integrations.openai_client")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--min-words", type=int, default=70)
    parser.add_argument(
        "--reverify", action="store_true",
        help="For cases already in --output: re-run only the mechanism-verification "
             "call (reuses the existing vignette/patient facts as-is) and update "
             "gold_label if it disagrees. Use this to backfill the mechanism check "
             "onto records generated before it existed.",
    )
    parser.add_argument(
        "--articles-dir", type=Path, default=Path("work/articles"),
        help="Local BioC-passages cache of {pmcid}.json files, checked before any live "
             "PMC fetch (see load_cached_article). Free, no rate limit, no 503s.",
    )
    parser.add_argument(
        "--allow-live-fetch", action="store_true",
        help="When a pmcid has no local article cache, fall back to fetching it live: "
             "NCBI's BioC endpoint first, then Europe PMC's fullTextXML if that fails "
             "(see get_article_passages in pubmed.py -- NCBI's BioC gateway has been "
             "seen returning intermittent 503s). OFF by default: without this flag, "
             "cases with no cached article either use --reverify's own already-extracted "
             "facts as a weaker stand-in (see facts_fallback_text) or, for brand-new "
             "cases with no facts yet, are skipped with a note rather than making a "
             "network call.",
    )
    args = parser.parse_args()
    add_source_dir(args.source_dir)

    cases = [c for c in read_json(args.cases) if c.get("accepted")]
    disease_by_name = {d["name"]: d for d in read_json(args.diseases)}

    done: dict[str, dict] = {}
    if args.output.exists():
        done = {r["pmid"]: r for r in read_json(args.output)}
    output: list[dict] = [] if args.reverify else list(done.values())

    headers = {"User-Agent": "gold-vignette-generation/1.0"}
    with httpx.Client(timeout=45, follow_redirects=True, headers=headers) as client:
        for case in cases:
            pmid = case["pmid"]
            if pmid in done and not args.reverify:
                continue

            if pmid in done and args.reverify:
                record = done[pmid]
                gold_label = record.get("gold_label")
                somatic_subgroup = record.get("somatic_subgroup", "")
                reverify_disease = disease_by_name.get(record.get("disease", case["disease"]), {})
                try:
                    content = load_cached_article(case.get("pmcid"), args.articles_dir)
                    basis = "cached_article"
                    if not content and args.allow_live_fetch and case.get("pmcid"):
                        article_passages = call_with_retry(get_article_passages, client, case["pmcid"])
                        content = article_text(article_passages) if article_passages else ""
                        basis = "live_refetch"
                    if not content:
                        content = facts_fallback_text(record)
                        basis = "extracted_facts_only"
                    verify = call_with_retry(
                        complete_json,
                        client_module=args.client,
                        model=args.model,
                        system=VERIFY_MECHANISM_PROMPT,
                        payload={
                            "target_disease": record.get("disease", case["disease"]),
                            "expected_mechanism": reverify_disease.get("expected_mechanism", ""),
                            "article": content,
                        },
                        max_tokens=300,
                    )
                    corrected_label = str(verify.get("label") or "").strip()
                    mechanism_reason = str(verify.get("reason") or "").strip()
                    label_corrected = bool(corrected_label in LABELS and corrected_label != gold_label)
                    if label_corrected:
                        print(f"{pmid}: LABEL CORRECTED {gold_label} -> {corrected_label} [{basis}] — {mechanism_reason}")
                        if corrected_label == "somatic":
                            somatic_subgroup = str(verify.get("somatic_subgroup") or "").strip()
                        gold_label = corrected_label
                    else:
                        print(f"{pmid}: confirmed {gold_label} [{basis}]")
                    record = {
                        **record,
                        "case_id": f"{gold_label[:2].upper()}_{pmid}",
                        "gold_label": gold_label,
                        "somatic_subgroup": somatic_subgroup,
                        "mechanism_reason": mechanism_reason,
                        "label_corrected": label_corrected,
                        "reverify_basis": basis,
                    }
                except Exception as exc:
                    print(f"{pmid}: reverify failed — {exc}; keeping existing record")
                output.append(record)
                write_json(args.output, output)
                continue

            disease = disease_by_name.get(case["disease"], {})
            gold_label = disease.get("target_class", case.get("target_class"))
            somatic_subgroup = disease.get("somatic_subgroup", "")

            if not case.get("pmcid"):
                print(f"{pmid}: SKIPPED — no pmcid on record (was this case not full_text?)")
                continue
            content = load_cached_article(case["pmcid"], args.articles_dir)
            if content:
                print(f"{pmid}: using cached article ({args.articles_dir}/{case['pmcid']}.json)")
            elif args.allow_live_fetch:
                try:
                    article_passages = call_with_retry(get_article_passages, client, case["pmcid"])
                except Exception as exc:
                    print(f"{pmid}: fetch failed — {exc}")
                    continue
                if not article_passages:
                    print(f"{pmid}: SKIPPED — full text unavailable at generation time")
                    continue
                content = article_text(article_passages)
            else:
                print(f"{pmid}: SKIPPED — no local article cache and --allow-live-fetch not set")
                continue

            try:
                result = call_with_retry(
                    complete_json,
                    client_module=args.client,
                    model=args.model,
                    system=VIGNETTE_PROMPT,
                    payload={"target_disease": case["disease"], "article": content[:8000]},
                    max_tokens=900,
                )
            except Exception as exc:
                print(f"{pmid}: LLM call failed — {exc}")
                continue

            vignette = result.get("vignette", "")
            if not isinstance(vignette, str):
                vignette = vignette.get("text", "") if isinstance(vignette, dict) else str(vignette)

            synonyms = disease.get("synonyms") or []
            if isinstance(synonyms, str):
                synonyms = synonyms.split("|") if synonyms else []
            leakage_found, leakage_terms = check_leakage(vignette, case["disease"], synonyms)
            word_count = len(vignette.split())

            mechanism_reason = ""
            label_corrected = False
            try:
                verify = call_with_retry(
                    complete_json,
                    client_module=args.client,
                    model=args.model,
                    system=VERIFY_MECHANISM_PROMPT,
                    payload={
                        "target_disease": case["disease"],
                        "expected_mechanism": disease.get("expected_mechanism", ""),
                        "article": content,
                    },
                    max_tokens=300,
                )
                corrected_label = str(verify.get("label") or "").strip()
                mechanism_reason = str(verify.get("reason") or "").strip()
                if corrected_label in LABELS and corrected_label != gold_label:
                    print(f"{pmid}: LABEL CORRECTED {gold_label} -> {corrected_label} — {mechanism_reason}")
                    label_corrected = True
                    if corrected_label == "somatic":
                        somatic_subgroup = str(verify.get("somatic_subgroup") or "").strip()
                    gold_label = corrected_label
            except Exception as exc:
                print(f"{pmid}: mechanism verification failed — {exc}; keeping disease-default label")

            record = {
                "case_id": f"{gold_label[:2].upper()}_{pmid}",
                "pmid": pmid,
                "pmcid": case["pmcid"],
                "disease": case["disease"],
                "gold_label": gold_label,
                "somatic_subgroup": somatic_subgroup,
                "mechanism_reason": mechanism_reason,
                "label_corrected": label_corrected,
                "patient_key": result.get("patient_key"),
                "presentation_facts": result.get("presentation_facts", []),
                "family_history": result.get("family_history", []),
                "exposure_history": result.get("exposure_history", []),
                "prediagnostic_findings": result.get("prediagnostic_findings", []),
                "vignette": vignette,
                "word_count": word_count,
                "leakage_found": leakage_found,
                "leakage_terms": leakage_terms,
                "meets_min_words": word_count >= args.min_words,
                "qa_status": "ok" if not leakage_found and word_count >= args.min_words else "needs_review",
            }
            output.append(record)
            write_json(args.output, output)
            print(f"{case['disease']} | {pmid}: {record['qa_status']}"
                  f"{' — LEAKAGE: ' + ', '.join(leakage_terms) if leakage_found else ''}")

    ok = sum(1 for r in output if r["qa_status"] == "ok")
    print(f"\nWrote {len(output)} vignettes to {args.output} ({ok} ok, {len(output) - ok} need review)")
    from collections import Counter
    print("gold_label counts:", dict(Counter(r["gold_label"] for r in output if r["qa_status"] == "ok")))


if __name__ == "__main__":
    main()