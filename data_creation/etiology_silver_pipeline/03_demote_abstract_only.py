from __future__ import annotations

import argparse
from pathlib import Path

from common import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("work/adult_primary_cases.json"))
    parser.add_argument("--output", type=Path, default=None, help="defaults to --input (in place)")
    args = parser.parse_args()
    out_path = args.output or args.input

    records: list[dict] = read_json(args.input)
    demoted = 0
    for r in records:
        if r.get("accepted") and r.get("content_level") == "abstract_only":
            r["accepted"] = False
            r["check"] = {**(r.get("check") or {}),
                          "exclusion_reason": "demoted: abstract_only acceptance superseded by "
                                               "--full-text-only re-search (2026-08-10)"}
            demoted += 1

    write_json(out_path, records)
    still_accepted = sum(1 for r in records if r.get("accepted"))
    print(f"Demoted {demoted} abstract_only-accepted records to accepted=False. "
          f"{still_accepted} records remain accepted (full_text). Wrote {out_path}")


if __name__ == "__main__":
    main()
