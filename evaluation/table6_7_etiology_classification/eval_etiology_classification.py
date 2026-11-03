from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_creation" / "etiology_silver_pipeline"))
# Retrieved titles contain non-ASCII text; a cp1252 console would raise inside RAG retrieval.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from evaluation.common import kegg
import pubmed
from common import SOMATIC_SUBGROUP, read_json
from llm_adapter import add_source_dir, complete_json

LABELS = ("single_gene", "chromosomal", "multifactorial", "non_genetic")
CONDITIONS = ((False, False), (True, False), (False, True), (True, True))

TRANSIENT_ERRORS = (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout)

KEYWORD_PROMPT = """Extract short search terms from this clinical vignette
that would help find relevant medical literature and biochemical-database
entries about its underlying cause. Split them into two lists:

- "clinical_terms" (2-4 items): key clinical, pathological, or imaging
  findings (e.g. "hypotonia", "coarse facial features", "hepatosplenomegaly").
- "biochemical_terms" (2-4 items): named analytes, metabolites, or lab
  abnormalities ONLY -- the substance/marker name alone (e.g. "ammonia",
  "propionylcarnitine", "lactate"), not the value or direction. Empty list
  if the vignette states no biochemical/lab findings.

Do NOT name or infer a specific diagnosis, gene, or syndrome -- these terms
are sent to external literature and biochemical-database searches and must
not leak the answer. Each term should be a short, generic medical word or
phrase that could apply to more than one disease.
Return JSON: {"clinical_terms":["", ...],"biochemical_terms":["", ...]}."""

CLASSIFY_PROMPT = """You are a clinical-genetics etiology classifier.

Classify the PRIMARY cause and mechanism, of the main disease for the patient described in the clinical
vignette into exactly ONE of these four labels:

1. "single_gene"
   One gene is sufficient to cause the disorder. Possible molecular mechanisms are:
   -Loss of function: too little or no functional protein.
   -Gain of function: the protein gains abnormal activity.
   -Dominant negative: the abnormal protein interferes with the normal protein.
   -Haploinsufficiency: one functional copy does not produce enough protein.
   -Toxic protein effect: the altered protein accumulates or damages cells.
   When seeing the case narrative, phenotypes, test, lab results, assume the molecular mechanism and detect
   whether the combination of the manifestations are connect in some of these mechanisms. If Yes than is single gene
   label if not do not assume single_gene. Same for mentioned diseases.

2. "chromosomal"
   The cause is a large-scale genomic or chromosome abnormality, 
   usually affecting multiple genes or chromosome regulation.
   Common mechanisms include:
    -Aneuploidy: an extra or missing chromosome.
    -Large copy-number variant: deletion or duplication of a chromosome segment.
    -Translocation: chromosome segments exchange positions.
    -Inversion: a chromosome segment is reversed.
    -Imprinting defect: the active parental copy is missing or incorrectly regulated.
    -Uniparental disomy: both chromosome copies come from the same parent.
    When assessing the case, search for confirmation of these mechanisms. If the underlying cause and manifestations are
    explainable by these mechanisms the case is chromosomal

3. "multifactorial"
   No single variant, and no chromosome-level abnormality, is sufficient to
   explain the disease. Possible mechanisms are:
   -Polygenic risk: many variants of small individual effect combine to cross
    a liability threshold; no one variant is sufficient on its own.
   -Gene-environment interaction: a genetic predisposition only produces
    disease together with a specific environmental, developmental, or
    lifestyle factor -- neither alone is sufficient.
   -Postzygotic/somatic process without one driver: a somatic-origin
    condition where no single confirmed clone or variant explains it.
   -Multifactorial developmental process: the phenotype arises from the
    combined effect of several contributing physiological or developmental
    processes, with no one sufficient genetic or chromosomal lesion identified.
   When assessing the case, search for confirmation of these mechanisms the
   same way you would for single_gene or chromosomal -- do not use this label
   as a fallback merely because no mechanism was confirmed; only use it when
   the evidence itself points to combined/distributed causation.

4. "non_genetic"
   The primary cause is acquired, with no pathogenic genetic or chromosomal
   alteration as the primary driver. Possible mechanisms are:
   -Infectious: an active or past infection is the direct cause.
   -Toxic/iatrogenic: a drug, toxin, or medical treatment/procedure is the
    direct cause.
   -Acquired nutritional/metabolic: a dietary deficiency or excess (not an
    inherited metabolic gene defect) is the direct cause.
   -Traumatic/mechanical: physical injury or mechanical disruption is the
    direct cause.
   -Vascular/ischemic: acquired vascular compromise (clot, hemorrhage,
    occlusion) is the direct cause.
   -Acquired autoimmune/inflammatory: an acquired immune-mediated process is
    the direct cause, not an inherited immune-gene defect.
   -Phenocopy: an acquired condition mimics a genetic disorder's phenotype
    without a primary genetic cause.
   A genetic susceptibility combined with one of these factors belongs under
   multifactorial's gene-environment-interaction mechanism, not here.

Return only valid JSON on one line, without Markdown or additional text.
Work out "mechanism_explained" FIRST, then "reasoning", THEN commit to
"label" last -- the label must follow from the mechanism you already stated,
not the other way around:
{"mechanism_explained":"one sentence naming the single specific mechanism from above that best matches the case",
"reasoning":"one sentence on why that mechanism implies the label below, do not overexplain",
"label":"single_gene|chromosomal|multifactorial|non_genetic"}
"""

CLASSIFY_PROMPT_RAG_SUFFIX = """

Below are excerpts retrieved from two external sources, using only clinical/
biochemical terms extracted from the vignette (never the diagnosis):
- "Literature": PubMed case-report abstracts.
- "Biochemical database": KEGG COMPOUND/DISEASE entries.

Both are keyword matches and may be irrelevant, incomplete, or describe a
different disease that happens to share a term with this vignette. Treat
them strictly as supportive context: use a passage only if it plausibly
explains a mechanism consistent with THIS vignette's own findings, and never
let it override explicit evidence stated in the vignette itself. If nothing
retrieved is relevant, ignore it and decide from the vignette alone."""

_PUBMED_RATE_LIMIT_SECONDS = 0.34


def call_with_retry(fn, *args, retries: int = 6, backoff: float = 2.0, **kwargs):
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except TRANSIENT_ERRORS as exc:
            last_error = exc
            wait = backoff * (2**attempt)
            print(f"    (transient error on {fn.__name__}, retry {attempt + 1}/{retries} in {wait:.1f}s: {exc})")
            time.sleep(wait)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 429:
                raise
            last_error = exc
            retry_after = exc.response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else backoff * (2**attempt)
            print(f"    (429 rate limited on {fn.__name__}, retry {attempt + 1}/{retries} in {wait:.1f}s)")
            time.sleep(wait)
    raise last_error


def effective_label(record: dict) -> str | None:
    label = record.get("gold_label")
    if label == "somatic":
        mechanism = record.get("mechanism_type")
        subgroup = record.get("somatic_subgroup") or SOMATIC_SUBGROUP.get(mechanism)
        return subgroup if subgroup in LABELS else None
    return label if label in LABELS else None


def load_cases(path: Path) -> list[dict]:
    records = read_json(path)
    cases = []
    for r in records:
        label = effective_label(r)
        if not label or not r.get("vignette") or not r.get("meets_min_words", True):
            continue
        cases.append({**r, "effective_label": label})
    return cases


def pick_few_shot_pool(cases: list[dict], per_class: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = {label: [] for label in LABELS}
    for c in cases:
        by_class[c["effective_label"]].append(c)
    pool: list[dict] = []
    pool_ids: set[str] = set()
    for label in LABELS:
        rng.shuffle(by_class[label])
        chosen = by_class[label][:per_class]
        pool.extend(chosen)
        pool_ids.update(c["case_id"] for c in chosen)
    remaining = [c for c in cases if c["case_id"] not in pool_ids]
    return pool, remaining


def few_shot_block(pool: list[dict]) -> str:
    lines = ["Labeled examples (for calibration only -- these are NOT the case to classify):"]
    for ex in pool:
        lines.append(f"---\nVignette: {ex['vignette']}\nLabel: {ex['effective_label']}")
    return "\n".join(lines)


def extract_keywords(vignette: str, client_module: str, model: str | None) -> dict[str, list[str]]:
    result = call_with_retry(
        complete_json, client_module=client_module, model=model,
        system=KEYWORD_PROMPT, payload={"vignette": vignette}, max_tokens=300,
    )
    clinical = [str(t) for t in (result.get("clinical_terms") or []) if str(t).strip()][:4]
    biochemical = [str(t) for t in (result.get("biochemical_terms") or []) if str(t).strip()][:4]
    return {"clinical_terms": clinical, "biochemical_terms": biochemical}


_ABSTRACT_MAX_CHARS = 1200


def retrieve_literature_context(keywords: dict[str, list[str]], client: httpx.Client, max_articles: int) -> str:
    terms = keywords["clinical_terms"] + keywords["biochemical_terms"]
    if not terms:
        return ""
    _, pmids = call_with_retry(pubmed.search, client, terms[0], terms[1:], max_articles * 3)
    time.sleep(_PUBMED_RATE_LIMIT_SECONDS)
    snippets = []
    for pmid in pmids[:max_articles]:
        try:
            meta = call_with_retry(pubmed.summary, client, pmid)
            time.sleep(_PUBMED_RATE_LIMIT_SECONDS)
            body = call_with_retry(pubmed.abstract, client, pmid)
            time.sleep(_PUBMED_RATE_LIMIT_SECONDS)
        except Exception as exc:
            print(f"    (PMID {pmid}: lookup failed, skipping — {exc})")
            continue
        title = meta.get("title") or ""
        body = body[:_ABSTRACT_MAX_CHARS] or title
        if not body:
            continue
        snippets.append(f"[PMID {pmid}] {title}\n{body}")
    if not snippets:
        return ""
    return "Literature (PubMed case-report abstracts):\n" + "\n\n".join(snippets)


_KEGG_RATE_LIMIT_SECONDS = 0.34


def retrieve_kegg_context(keywords: dict[str, list[str]], client: httpx.Client, max_entries: int) -> str:
    compound_hits: dict[str, str] = {}
    for term in keywords["biochemical_terms"]:
        try:
            hits = call_with_retry(kegg.find_compound, client, term)
        except Exception as exc:
            print(f"    (KEGG compound lookup for {term!r} failed, skipping — {exc})")
            continue
        time.sleep(_KEGG_RATE_LIMIT_SECONDS)
        for entry_id, name in hits[:max_entries]:
            compound_hits.setdefault(entry_id, name)

    disease_hits: dict[str, str] = {}
    for term in keywords["clinical_terms"]:
        try:
            hits = call_with_retry(kegg.find_disease, client, term)
        except Exception as exc:
            print(f"    (KEGG disease lookup for {term!r} failed, skipping — {exc})")
            continue
        time.sleep(_KEGG_RATE_LIMIT_SECONDS)
        for entry_id, name in hits[:max_entries]:
            disease_hits.setdefault(entry_id, name)

    lines = []
    if compound_hits:
        lines.append("Compounds/metabolites:")
        lines.extend(f"  [{eid}] {name}" for eid, name in list(compound_hits.items())[:max_entries])
    if disease_hits:
        lines.append("Diseases (with KEGG category, when available):")
        for entry_id, name in list(disease_hits.items())[:max_entries]:
            try:
                fields = call_with_retry(kegg.get_entry, client, entry_id)
                time.sleep(_KEGG_RATE_LIMIT_SECONDS)
            except Exception:
                fields = {}
            category = fields.get("CATEGORY", "")
            suffix = f" -- category: {category}" if category else ""
            lines.append(f"  [{entry_id}] {name}{suffix}")
    if not lines:
        return ""
    return "Biochemical database (KEGG COMPOUND/DISEASE):\n" + "\n".join(lines)


def retrieve_rag_context(keywords: dict[str, list[str]], client: httpx.Client, max_articles: int) -> str:
    parts = [
        retrieve_literature_context(keywords, client, max_articles),
        retrieve_kegg_context(keywords, client, max_articles),
    ]
    parts = [p for p in parts if p]
    return "\n\n".join(parts)


def classify(
    vignette: str, few_shot_text: str | None, rag_text: str | None,
    client_module: str, model: str | None,
) -> tuple[str, str]:
    prompt = CLASSIFY_PROMPT
    payload: dict = {"vignette": vignette}
    if few_shot_text:
        payload["labeled_examples"] = few_shot_text
    if rag_text:
        prompt = prompt + CLASSIFY_PROMPT_RAG_SUFFIX
        payload["retrieved_context"] = rag_text
    result = call_with_retry(
        complete_json, client_module=client_module, model=model,
        system=prompt, payload=payload, max_tokens=900,
    )
    label = str(result.get("label") or "").strip()
    reasoning = str(result.get("reasoning") or "")
    mechanism = str(result.get("mechanism_explained") or "")
    return label, reasoning, mechanism


def load_done(out_path: Path) -> set[tuple[str, bool, bool, int]]:
    done: set[tuple[str, bool, bool, int]] = set()
    if not out_path.exists():
        return done
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done.add((row["case_id"], row["few_shot"], row["rag"], row["repeat"]))
    return done


def main() -> None:
    from generesolver.env import load_project_env
    load_project_env(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--vignettes", type=Path,
                         default=ROOT / "data_creation" / "etiology_silver_pipeline" / "output" / "gold_vignettes_v2.json")
    parser.add_argument("--out", type=Path,
                         default=ROOT / "data_creation" / "etiology_silver_pipeline" / "output" / "predictions.jsonl")
    parser.add_argument("--source-dir", type=Path, default=ROOT)
    parser.add_argument("--client", default="generesolver.integrations.medgemma_client")
    parser.add_argument("--model", default=None)
    parser.add_argument("--few-shot-per-class", type=int, default=2)
    parser.add_argument("--rag-max-articles", type=int, default=3,
                         help="max hits kept per RAG source: PubMed abstracts, "
                              "and separately KEGG compound/disease entries")
    parser.add_argument("--repeats", type=int, default=5,
                         help="independent runs per configuration (paper: 5); the MedGemma response "
                              "cache is always disabled so every run is a fresh model call")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N cases (smoke test)")
    args = parser.parse_args()
    # Repeats must be real model calls: never serve MedGemma responses from the on-disk cache.
    os.environ["GENERESOLVER_LLM_CACHE_DISABLE"] = "1"
    add_source_dir(args.source_dir)

    cases_all = load_cases(args.vignettes)
    few_shot_pool, eval_cases = pick_few_shot_pool(cases_all, args.few_shot_per_class, args.seed)
    if args.limit:
        eval_cases = eval_cases[: args.limit]
    few_shot_text = few_shot_block(few_shot_pool)
    print(
        f"{len(cases_all)} usable case(s) ({len(few_shot_pool)} reserved as the few-shot "
        f"pool, {len(eval_cases)} to evaluate) across {len(LABELS)} classes."
    )

    done = load_done(args.out)
    if done:
        print(f"Resuming: {len(done)} prediction(s) already recorded in {args.out}.")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Keyed by (case, repeat): keyword extraction is a MedGemma call, so each run extracts and
    # retrieves afresh; within one run both RAG configurations share the same context (paired).
    rag_cache: dict[tuple[str, int], str] = {}
    total_written = len(done)

    headers = {"User-Agent": "etiology-silver-dataset/1.0"}
    with httpx.Client(timeout=45, follow_redirects=True, headers=headers) as client:
        for repeat in range(args.repeats):
            for case in eval_cases:
                for few_shot, rag in CONDITIONS:
                    key = (case["case_id"], few_shot, rag, repeat)
                    if key in done:
                        continue

                    rag_text = None
                    if rag:
                        rag_key = (case["case_id"], repeat)
                        if rag_key not in rag_cache:
                            try:
                                keywords = extract_keywords(case["vignette"], args.client, args.model)
                                rag_cache[rag_key] = retrieve_rag_context(
                                    keywords, client, args.rag_max_articles
                                )
                            except Exception as exc:
                                print(f"{case['case_id']}: RAG retrieval failed, will retry on next run — {exc}")
                                continue
                        rag_text = rag_cache[rag_key]

                    fs_text = few_shot_text if few_shot else None
                    try:
                        label, reasoning, mechanism = classify(
                            case["vignette"], fs_text, rag_text, args.client, args.model
                        )
                    except Exception as exc:
                        print(f"{case['case_id']} few_shot={few_shot} rag={rag}: SKIPPED after error, will retry on next run — {exc}")
                        continue

                    if label not in LABELS:
                        print(
                            f"{case['case_id']} few_shot={few_shot} rag={rag}: model returned "
                            f"invalid label {label!r} — SKIPPED (not coerced), will retry on next run."
                        )
                        continue

                    row = {
                        "case_id": case["case_id"], "gold_label": case["effective_label"],
                        "few_shot": few_shot, "rag": rag, "prediction": label,
                        "repeat": repeat, "reasoning": reasoning, "mechanism": mechanism
                    }
                    done.add(key)
                    total_written += 1
                    with args.out.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                    print(
                        f"{case['case_id']} few_shot={few_shot} rag={rag} repeat={repeat}: "
                        f"gold={case['effective_label']} pred={label}"
                    )

    print(f"\nDone. {total_written} prediction(s) total in {args.out}")


if __name__ == "__main__":
    main()
