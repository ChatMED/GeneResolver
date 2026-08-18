"""Emit exactly ONE case's pheval gene-result parquet, in a fresh interpreter.

Why this exists: as a fallback path for the rare genuinely-transient write hiccup, tried after
in-process retries in _verify_and_retry_emission(). Real root cause of the FileNotFoundError
this used to chase (see eval_genome_vcf_ranking.py's _ensure_truth_placeholders docstring):
generate_gene_result() reads its own target path to learn ground truth before overwriting it,
so that path must already exist (as a placeholder or a real result) -- it must NEVER be
deleted before calling it.

Usage: python -m evaluation.stage5_genome_vcf_ranking._emit_single_case <raw_dir> <results_dir> <case_id>
Exit 0 on success (correct row count on disk), exit 1 otherwise.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
# NOTE: do NOT wrap sys.stdout/sys.stderr here -- importing evaluation.eval_genome_vcf_ranking
# below already does this at module scope. Wrapping twice creates two TextIOWrapper objects
# around the same underlying buffer; the first one gets garbage-collected and closes that
# shared buffer in its finalizer, so the second (the one still assigned to sys.stdout) then
# fails with "ValueError: I/O operation on closed file" on any later print.

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import evaluation.stage5_genome_vcf_ranking.eval_genome_vcf_ranking as ev  # noqa: E402  (this import already seeds
# executed_results.add(ResultType.GENE) at module scope -- see that file's comment -- so the
# destructive create_empty_pheval_result() sweep can never fire from this subprocess either)


def main() -> None:
    raw_dir = Path(sys.argv[1])
    results_dir = Path(sys.argv[2])
    case_id = sys.argv[3]

    cfg = ev.CONFIG
    gid = ev.GeneIdentifierUpdater(gene_identifier="ensembl_id", identifier_map=ev.create_gene_identifier_map())

    pp = cfg.LIRICAL_PP / f"{case_id}.json"
    raw_f = raw_dir / f"{case_id}.json"
    ranked = json.loads(raw_f.read_text(encoding="utf-8"))

    ev._emit_pheval(ranked, pp, results_dir, gid)

    got_n = ev._parquet_row_count(results_dir, case_id)
    if ev._row_count_ok(got_n, len(ranked)):
        print(f"OK {case_id}: {got_n} rows")
        sys.exit(0)
    print(f"FAIL {case_id}: expected {len(ranked)} (or +1) rows, got {got_n}")
    sys.exit(1)


if __name__ == "__main__":
    main()
