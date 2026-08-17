from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_PATH = Path(__file__).resolve().parent / "generated_deid_dataset.jsonl"
N_CASES = 100

GENERATION_PROMPT = """You are generating a synthetic clinical de-identification
test set. Produce ONE realistic but entirely FICTIONAL
clinical referral or consultation note per item, written
in the style of a rare-disease genetics referral. The
note may include clinically relevant findings such as
seizures, developmental delay, dysmorphic features,
metabolic abnormalities, and skeletal or connective-
tissue findings.

Every patient, relative, clinician, facility,
location, and identifying number must be invented.
Do not use the name of a real person, public figure,
hospital, clinic, or other healthcare institution.
Do not reuse a real medical record number or any
other real identifying information.

Embed a natural mixture of information that a
clinical de-identification system should detect and
mask. The possible information fields include:

- patient, relative, or clinician names;
- street addresses, cities, postal codes, and other
  detailed geographic locations;
- dates linked to an individual, including dates of
  birth, admission dates, discharge dates, and
  consultation dates;
- telephone and fax numbers;
- email addresses;
- medical record numbers;
- health-plan or insurance identifiers;
- account numbers;
- professional licence or certificate numbers;
- vehicle identifiers;
- medical-device identifiers or serial numbers;
- URLs;
- IP addresses;
- biometric identifiers represented in text;
- other unique identifying numbers or codes;
- ages above 89 years.

Vary the information fields included in each note.
Do not include every field in every note. Include a
minority of notes, approximately one in six, with no
identifying information so that the dataset also
contains true-negative examples.

For EACH note, output the exact text substrings that
should be detected and masked. Copy every annotated
substring VERBATIM from the note, preserving the
same characters and capitalization. These
annotations will be checked programmatically against
the note text. For example, if the note contains
"DOB 03/15/1985", the annotated token must be
exactly "03/15/1985", rather than "March 15, 1985" or
"1985".

Return one JSON object per line, without Markdown
fencing:
{"id": "PHI_0NN", "note": "...",
"expected_pii_tokens": ["...", ...]}"""


def _verify_tokens(note: str, tokens: list[str]) -> list[str]:
    note_lower = note.lower()
    return [t for t in tokens if t.lower() not in note_lower]


def generate_one(item_id: str, backend: str, model: str | None) -> dict:
    user_message = (
        f'Generate item "{item_id}". Return exactly one JSON object '
        "(not a list, no markdown fencing) matching the schema in your instructions."
    )
    if backend == "anthropic":
        from generesolver.integrations import anthropic_client as backend_client
    elif backend == "openai":
        from generesolver.integrations import openai_client as backend_client
    else:
        raise ValueError(f"unknown backend: {backend}")

    record = backend_client.complete_json(
        system=GENERATION_PROMPT,
        user=user_message,
        max_tokens=4000,
        model=model,
    )
    record.setdefault("id", item_id)
    record.setdefault("expected_pii_tokens", [])
    return record


def run(n_cases: int, backend: str, model: str | None) -> None:
    existing: dict[str, dict] = {}
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing[rec["id"]] = rec

    mismatch_count = 0
    negative_control_count = 0

    for i in range(1, n_cases + 1):
        item_id = f"PHI_{i:03d}"
        if item_id in existing:
            print(f"skip {item_id} (already generated)")
            continue

        print(f"generating {item_id}...")
        try:
            record = generate_one(item_id, backend, model)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        tokens = record.get("expected_pii_tokens") or []
        bad_tokens = _verify_tokens(record.get("note", ""), tokens)
        if bad_tokens:
            mismatch_count += 1
            print(f"  VERIFICATION FAILED — {len(bad_tokens)} token(s) not found verbatim in note: {bad_tokens}")
        if not tokens:
            negative_control_count += 1

        existing[item_id] = record
        with OUTPUT_PATH.open("w", encoding="utf-8") as f:
            for rec in existing.values():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(existing)}/{n_cases} cases in {OUTPUT_PATH}")
    print(f"Negative controls (no PII tokens): {negative_control_count} "
          f"(paper's original corpus: 16/100)")
    print(f"Cases with verification mismatches: {mismatch_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=N_CASES, help="Number of cases to generate (default 100).")
    parser.add_argument("--backend", choices=["anthropic", "openai"], default="anthropic",
                         help="LLM backend (paper used Claude Sonnet 5, i.e. 'anthropic' — the default).")
    parser.add_argument("--model", default=None, help="Override model id (else backend default).")
    args = parser.parse_args()
    run(args.n, args.backend, args.model)


if __name__ == "__main__":
    main()
