from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import httpx

from common import bool_value, read_json, write_json
from llm_adapter import add_source_dir, complete_json
from pubmed import article_text, fulltext, passages, pmcid, summary

PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

TRANSIENT_ERRORS = (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout)


def call_with_retry(fn, *args, retries: int = 4, backoff: float = 1.5, **kwargs):
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except TRANSIENT_ERRORS as exc:
            last_error = exc
            wait = backoff * (2**attempt)
            print(f"    (transient error on {fn.__name__}, retry {attempt + 1}/{retries} in {wait:.1f}s: {exc})")
            time.sleep(wait)
    raise last_error


def _title_terms(name: str, synonyms: list[str]) -> str:
    terms = [name] + list(synonyms)
    return " OR ".join(f'"{t}"[Title]' for t in terms)


def _esearch_pubmed(client: httpx.Client, term: str, retmax: int) -> list[str]:
    response = client.get(
        PUBMED_ESEARCH,
        params={"db": "pubmed", "term": term, "retmax": retmax, "retmode": "json", "sort": "relevance"},
    )
    response.raise_for_status()
    return response.json().get("esearchresult", {}).get("idlist", [])


def search_disease_literature(
    client: httpx.Client, name: str, synonyms: list[str], screened: int
) -> tuple[str, list[str]]:
    title_terms = _title_terms(name, synonyms)
    tiers = [
        f'({title_terms}) AND Review[Publication Type] NOT "Case Reports"[Publication Type]',
        f'({title_terms}) AND (Review[Publication Type] OR "genetics"[Subheading]) NOT "Case Reports"[Publication Type]',
        f'({title_terms}) AND (Review[Publication Type] OR "genetics"[Subheading])',
        f'({title_terms})',
    ]
    for tier, query in enumerate(tiers):
        ids = call_with_retry(_esearch_pubmed, client, query, screened)
        if ids:
            return f"tier{tier}: {query}", ids
    return tiers[-1], []


CONFIRM_PROMPT = """You are checking whether a source actually supports a
claimed disease etiology classification, not merely whether it mentions the
disease. Read the supplied source content (may be a full article, or only a
title/journal/year if full text was unavailable — treat title-only sources
with much lower confidence and say so).

Return a JSON object with:
- relevant (JSON boolean): does this source address the disease's etiology/
  inheritance/mechanism at all?
- supports_target_class (JSON boolean): does the source explicitly state or
  clearly support the claimed etiology_class / mechanism_note for this
  disease?
- contradicts_target_class (JSON boolean): does the source make a general
  claim about this disease's ESTABLISHED etiology that conflicts with the
  claimed etiology_class (e.g. a review stating the disease is polygenic
  when the claim is single_gene, or disputing monogenic causation as a
  field-level consensus)?
- confidence (low, moderate, or high): lower if the assessment is based on
  title/journal only rather than full text.
- reason: one or two sentences explaining the call.
- short_supporting_quote: at most 15 words from the source if one directly
  supports the claim, else empty string.

Use JSON booleans. If the source does not address etiology at all, set
relevant to false and leave supports_target_class /
contradicts_target_class false."""


def load_seed(path: Path) -> list[dict]:
    rows = read_json(path)
    for row in rows:
        row.setdefault("synonyms", [])
    return rows


def gather_candidates(
    client, disease: dict, screened: int, max_candidates: int
) -> list[dict]:
    try:
        query, pmids = search_disease_literature(
            client, disease["name"], disease.get("synonyms", []), screened
        )
    except TRANSIENT_ERRORS as exc:
        print(f"  SEARCH FAILED for {disease['name']} after retries: {exc}")
        return []
    candidates: list[dict] = []
    for pmid in pmids[:max_candidates]:
        try:
            meta = call_with_retry(summary, client, pmid)
        except TRANSIENT_ERRORS as exc:
            print(f"  summary() failed for PMID {pmid} after retries, skipping candidate: {exc}")
            continue
        time.sleep(0.4)
        title = meta.get("title", "")
        journal = meta.get("source") or meta.get("fulljournalname") or ""
        year = (meta.get("pubdate") or "")[:4]
        doi = next(
            (x.get("value") for x in meta.get("articleids", []) if x.get("idtype") == "doi"),
            None,
        )
        text = ""
        title_only = True
        pmc_id = None
        try:
            pmc_id = call_with_retry(pmcid, client, pmid)
        except TRANSIENT_ERRORS as exc:
            print(f"  pmcid() failed for PMID {pmid} after retries, falling back to title-only: {exc}")
        time.sleep(0.4)
        if pmc_id:
            bioc = None
            try:
                bioc = call_with_retry(fulltext, client, pmc_id)
            except TRANSIENT_ERRORS as exc:
                print(f"  fulltext() failed for PMC{pmc_id} after retries, falling back to title-only: {exc}")
            time.sleep(0.4)
            if bioc:
                text = article_text(passages(bioc))
                title_only = False
        candidates.append(
            {
                "pmid": pmid,
                "pmcid": pmc_id,
                "doi": doi,
                "title": title,
                "journal": journal,
                "year": year,
                "title_only": title_only,
                "content": text or f"Title: {title}\nJournal: {journal} ({year})",
                "query": query,
            }
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, default=Path("work/citation_seed_diseases.json"))
    parser.add_argument("--source-dir", type=Path, default=Path("../.."))
    parser.add_argument("--client", default="generesolver.integrations.openai_client")
    parser.add_argument("--model", default=None)
    parser.add_argument("--screened", type=int, default=15, help="PubMed results screened per disease")
    parser.add_argument("--max-candidates", type=int, default=5, help="Candidates evaluated per disease")
    parser.add_argument("--accept-per-disease", type=int, default=2)
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    args = parser.parse_args()
    add_source_dir(args.source_dir)
    diseases = load_seed(args.seed)

    detail_path = args.work_dir / "disease_citations.json"
    summary_path = args.work_dir / "disease_citations_summary.csv"
    done_names = set()
    detail: list[dict] = []
    if detail_path.exists():
        detail = read_json(detail_path)
        done_names = {d["disease"] for d in detail}

    headers = {"User-Agent": "etiology-citation-curation/1.0"}
    with httpx.Client(timeout=45, follow_redirects=True, headers=headers) as client:
        for disease in diseases:
            if disease["name"] in done_names:
                continue
            try:
                candidates = gather_candidates(client, disease, args.screened, args.max_candidates)
                assessments: list[dict] = []
                for candidate in candidates:
                    result = complete_json(
                        client_module=args.client,
                        model=args.model,
                        system=CONFIRM_PROMPT,
                        payload={
                            "disease": disease["name"],
                            "etiology_class": disease["target_class"],
                            "mechanism_note": disease.get("expected_mechanism", ""),
                            "source_title_only": candidate["title_only"],
                            "source_content": candidate["content"][:12000],
                        },
                        max_tokens=500,
                    )
                    assessments.append({**candidate, "assessment": result})
            except Exception as exc:
                print(f"{disease['name']}: SKIPPED after error, will retry on next run — {exc}")
                continue
            accepted = [
                a for a in assessments
                if bool_value(a["assessment"].get("relevant"))
                and bool_value(a["assessment"].get("supports_target_class"))
                and not bool_value(a["assessment"].get("contradicts_target_class"))
            ]
            accepted.sort(
                key=lambda a: (
                    a["title_only"],
                    {"high": 0, "moderate": 1, "low": 2}.get(
                        str(a["assessment"].get("confidence", "")).strip().casefold(), 3
                    ),
                )
            )
            contradicted = [
                a for a in assessments
                if bool_value(a["assessment"].get("contradicts_target_class"))
            ]
            record = {
                "disease": disease["name"],
                "target_class": disease["target_class"],
                "somatic_subgroup": disease.get("somatic_subgroup", ""),
                "candidates_evaluated": len(assessments),
                "accepted": accepted[: args.accept_per_disease],
                "contradicted": contradicted,
                "all_assessments": assessments,
            }
            detail.append(record)
            write_json(detail_path, detail)
            status = (
                "CONTRADICTED" if contradicted and not accepted
                else "confirmed" if accepted
                else "not_found"
            )
            print(f"{disease['name']}: {status} ({len(accepted)} accepted, {len(contradicted)} contradicting)")

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "disease", "target_class", "somatic_subgroup", "status",
            "citation_1_pmid", "citation_1_title", "citation_1_title_only",
            "citation_2_pmid", "citation_2_title", "citation_2_title_only",
            "contradicted_pmids",
        ])
        for record in detail:
            accepted = record["accepted"]
            contradicted_pmids = "|".join(c["pmid"] for c in record["contradicted"])
            status = (
                "CONTRADICTED" if record["contradicted"] and not accepted
                else "confirmed" if accepted
                else "not_found"
            )
            c1 = accepted[0] if len(accepted) > 0 else {}
            c2 = accepted[1] if len(accepted) > 1 else {}
            writer.writerow([
                record["disease"], record["target_class"], record["somatic_subgroup"], status,
                c1.get("pmid", ""), c1.get("title", ""), c1.get("title_only", ""),
                c2.get("pmid", ""), c2.get("title", ""), c2.get("title_only", ""),
                contradicted_pmids,
            ])
    print(f"\nWrote {len(detail)} disease records to {detail_path}")
    print(f"Wrote summary to {summary_path}")
    not_found = [r["disease"] for r in detail if not r["accepted"]]
    if not_found:
        print(f"\n{len(not_found)} diseases with no accepted citation — review manually: {not_found}")


if __name__ == "__main__":
    main()