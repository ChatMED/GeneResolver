from __future__ import annotations

import argparse
from pathlib import Path

import httpx

from common import read_json, write_json
from pubmed import pmcid_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("work/adult_primary_cases.json"))
    parser.add_argument("--output", type=Path, default=None, help="defaults to --input (in place)")
    args = parser.parse_args()
    out_path = args.output or args.input

    records: list[dict] = read_json(args.input)
    todo = [r for r in records if r.get("accepted") and not r.get("pmcid")]
    pmids = [r["pmid"] for r in todo]
    print(f"{len(records)} records total, {len(todo)} accepted+pmcid-missing to re-resolve "
          f"({-(-len(pmids) // 200)} batch request(s))")

    headers = {"User-Agent": "adult-case-retrieval/1.0"}
    with httpx.Client(timeout=45, follow_redirects=True, headers=headers) as client:
        resolved_map = pmcid_batch(client, pmids)

    resolved = 0
    for record in todo:
        pc = resolved_map.get(record["pmid"])
        if pc:
            record["pmcid"] = pc
            resolved += 1
            print(f"  {record['pmid']}: -> {pc}")
        else:
            print(f"  {record['pmid']}: no PMC copy")

    write_json(out_path, records)
    print(f"\nResolved {resolved}/{len(todo)} previously-null pmcids. Wrote {out_path}")


if __name__ == "__main__":
    main()
