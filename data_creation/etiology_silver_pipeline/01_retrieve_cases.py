from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx

from common import read_json, write_json
from llm_adapter import add_source_dir, complete_json
from pubmed import article_text, get_article_passages, pmcid_batch

PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TRANSIENT_ERRORS = (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout)

CHECK_PROMPT = """Check this case report against two requirements for
target_disease. source_content_level tells you whether you're reading the
full article ("full_text") or only the abstract ("abstract_only") — treat
abstract_only judgments as lower confidence, since age and primary-vs-
incidental status are often clearer in the full case presentation.

1. is_adult: the patient is an adult (18+) at the time of the reported
   presentation. If age isn't explicitly stated, infer conservatively from
   clear contextual cues (occupation, "elderly", specific decade of life,
   etc.) — if genuinely unclear, set is_adult to false rather than guessing.
2. is_primary_diagnosis: target_disease is THE main condition this case
   report is about — not a secondary/incidental finding, past medical
   history item, or comorbidity in a report primarily about something else.
   A report titled/focused on a different condition that merely mentions
   the patient also has target_disease does NOT qualify.
3. is_incidental_findings: target_disease is not merely an incidental finding in the report.

Return a JSON object with: is_adult; age_reported (string, e.g. "34 years"
or "elderly" or "not stated"); is_primary_diagnosis; confidence (low,
moderate, or high); patient_key (a short identifying label for this
patient, e.g. "58F with..."); and exclusion_reason (empty string if both
checks pass)."""


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


def search_adult_case_reports(client: httpx.Client, disease_name: str, batch_size: int, start: int = 0) -> list[str]:
    term = (
        f'"{disease_name}"[Title] AND "Case Reports"[Publication Type] '
        f'AND ("Adult"[MeSH Terms] OR "Middle Aged"[MeSH Terms] OR "Aged"[MeSH Terms]) '
        f'NOT ("Infant"[MeSH Terms] OR "Child"[MeSH Terms] OR "Adolescent"[MeSH Terms])'
    )
    resp = call_with_retry(
        client.get, PUBMED_ESEARCH,
        params={"db": "pubmed", "term": term, "retmax": batch_size, "retstart": start, "retmode": "json", "sort": "relevance"},
    )
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def fetch_content(client: httpx.Client, pmid: str, known_pmcid: str | None) -> tuple[str, str | None, str]:
    pmc_id = known_pmcid
    if pmc_id:
        rows = call_with_retry(get_article_passages, client, pmc_id)
        time.sleep(0.3)
        if rows:
            return article_text(rows), pmc_id, "full_text"
    resp = call_with_retry(
        client.get, PUBMED_EFETCH,
        params={"db": "pubmed", "id": pmid, "rettype": "abstract", "retmode": "text"},
    )
    resp.raise_for_status()
    time.sleep(0.3)
    return resp.text.strip(), pmc_id, "abstract_only"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("work/citation_seed_diseases.json"))
    parser.add_argument("--output", type=Path, default=Path("work/adult_primary_cases.json"))
    parser.add_argument("--source-dir", type=Path, default=Path("../.."))
    parser.add_argument("--client", default="generesolver.integrations.openai_client")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--max-per-disease", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=15, help="results pulled per search page")
    parser.add_argument(
        "--max-screened", type=int, default=200,
        help="safety cap on total candidates screened per disease before giving up (not a target — retrieval keeps paging until this or a genuinely exhausted search)",
    )
    parser.add_argument(
        "--full-text-only", action="store_true",
        help="Skip abstract-only candidates entirely (no LLM call spent on them) — "
        "recommended, since age/primary-diagnosis judgment and vignette content "
        "are both less reliable from an abstract than a full case presentation.",
    )
    args = parser.parse_args()
    add_source_dir(args.source_dir)

    diseases = read_json(args.input)
    results: list[dict] = read_json(args.output) if args.output.exists() else []
    done_by_disease: dict[str, list[dict]] = {}
    for r in results:
        done_by_disease.setdefault(r["disease"], []).append(r)

    headers = {"User-Agent": "adult-case-retrieval/1.0"}
    with httpx.Client(timeout=45, follow_redirects=True, headers=headers) as client:
        for disease in diseases:
            name = disease["name"]
            already = [r for r in done_by_disease.get(name, []) if r.get("accepted")]
            if len(already) >= args.max_per_disease:
                continue
            seen_pmids = {r["pmid"] for r in done_by_disease.get(name, [])}
            accepted_count = len(already)
            total_screened = 0
            page_start = 0
            while accepted_count < args.max_per_disease and total_screened < args.max_screened:
                try:
                    pmids = search_adult_case_reports(client, name, args.batch_size, start=page_start)
                except Exception as exc:
                    print(f"{name}: SEARCH FAILED — {exc}")
                    break
                new_pmids = [p for p in pmids if p not in seen_pmids]
                if not pmids:
                    break
                page_start += args.batch_size
                if not new_pmids:
                    continue
                try:
                    pmcid_map = call_with_retry(pmcid_batch, client, new_pmids)
                except Exception as exc:
                    print(f"  id-conversion failed for this page — {exc}")
                    pmcid_map = {}
                time.sleep(0.3)
                for pmid in new_pmids:
                    if accepted_count >= args.max_per_disease or total_screened >= args.max_screened:
                        break
                    seen_pmids.add(pmid)
                    total_screened += 1
                    try:
                        content, pmc_id, content_level = fetch_content(client, pmid, pmcid_map.get(pmid))
                    except Exception as exc:
                        print(f"  {pmid}: fetch failed — {exc}")
                        continue
                    if args.full_text_only and content_level == "abstract_only":
                        record = {
                            "disease": name,
                            "target_class": disease["target_class"],
                            "pmid": pmid,
                            "pmcid": pmc_id,
                            "content_level": content_level,
                            "accepted": False,
                            "check": {"exclusion_reason": "abstract_only, skipped (--full-text-only)"},
                        }
                        results.append(record)
                        write_json(args.output, results)
                        print(f"{name} | {pmid}: skip (abstract_only, no LLM call spent)")
                        continue
                    try:
                        check = call_with_retry(
                            complete_json,
                            client_module=args.client,
                            model=args.model,
                            system=CHECK_PROMPT,
                            payload={
                                "target_disease": name,
                                "target_class": disease["target_class"],
                                "source_content_level": content_level,
                                "article": content[:8000],
                            },
                            max_tokens=400,
                        )
                    except Exception as exc:
                        print(f"  {pmid}: LLM check failed — {exc}")
                        continue
                    accepted = bool(check.get("is_adult")) and bool(check.get("is_primary_diagnosis"))
                    if accepted:
                        accepted_count += 1
                    record = {
                        "disease": name,
                        "target_class": disease["target_class"],
                        "pmid": pmid,
                        "pmcid": pmc_id,
                        "content_level": content_level,
                        "accepted": accepted,
                        "check": check,
                    }
                    results.append(record)
                    write_json(args.output, results)
                    status = "ACCEPTED" if accepted else f"skip ({check.get('exclusion_reason', 'unclear')})"
                    print(f"{name} | {pmid}: {status} ({accepted_count}/{args.max_per_disease})")
            if accepted_count < args.max_per_disease:
                print(f"  -> {name}: only {accepted_count}/{args.max_per_disease} after screening {total_screened} (search exhausted or cap hit)")

    print(f"\nWrote {len(results)} candidate records to {args.output}")


if __name__ == "__main__":
    main()