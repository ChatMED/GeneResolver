#!/usr/bin/env python
"""
GeneResolver — full setup: fetch all the data the shipped pipeline needs, check the
environment/config, then run the demo case to confirm everything actually works.

    python setup_quickstart.py

WHAT THIS FETCHES BY DEFAULT (auto-downloaded, idempotent — safe to re-run):
  - HPO consortium annotations (phenotype.hpoa, phenotype_to_genes.txt)
  - Orphanet Orphadata products (disease x-refs, gene associations, prevalence, HPO freq)
  - Reactome NCBI gene -> pathway mapping
  - gnomAD v4.1 gene constraint metrics (~150 MB)
  - MONDO disease-identifier crosswalk (mondo.sssom.tsv, ~13 MB)
  - ClinVar VCF (GRCh37 + GRCh38) and variant_summary.txt
  - MANE GRCh38 transcript summary
  - ClinGen gene-validity + haploinsufficiency curation
  - UniProt HumsaVar, gnomAD exomes sites, REVEL, UCSC phyloP + simpleRepeat

This now covers every task mode, including the ACMG/VCF arm's non-Docker inputs.

OPT-IN (large, only needed for local on-premise VEP annotation — pass --with-vep-cache):
  - Ensembl VEP local cache + Docker image (~23 GB) — see scripts/download_vep_cache.py

NOT FETCHED HERE (evaluation-only corpora, not part of this repo):
  - The GSC+ phenotype-extraction gold-standard corpus, PhEval Phenopacket Store corpus,
    and the monarch_pheval sibling repo

Usage:
    python setup_quickstart.py                  # data + config check + demo run
    python setup_quickstart.py --skip-demo       # data + config check only
    python setup_quickstart.py --with-vep-cache  # also fetch the ~23 GB local VEP cache
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_DOWNLOAD_STEPS = [
    ("scripts.download_hpo_annotations", "HPO consortium annotations"),
    ("scripts.download_orphanet", "Orphanet (Orphadata)"),
    ("scripts.download_reactome", "Reactome gene->pathway mapping"),
    ("scripts.download_gnomad_constraint", "gnomAD v4.1 constraint metrics (~150 MB)"),
    ("scripts.download_mondo_sssom", "MONDO disease-identifier crosswalk (~13 MB)"),
    ("scripts.download_clinvar", "ClinVar VCF (GRCh37 + GRCh38)"),
    ("scripts.download_clinvar_variant_summary", "ClinVar variant_summary.txt"),
    ("scripts.download_mane", "MANE GRCh38 transcript summary"),
    ("scripts.download_clingen_gene_validity", "ClinGen gene-disease validity"),
    ("scripts.download_gnomad_exomes_sites", "gnomAD exomes sites"),
    ("scripts.download_revel", "REVEL missense scores"),
    ("scripts.download_uniprot_humsavar", "UniProt HumsaVar"),
    ("scripts.download_ucsc_phylop", "UCSC phyloP conservation"),
    ("scripts.download_ucsc_simple_repeat", "UCSC simpleRepeat"),
]

def _run_step(module: str, label: str) -> bool:
    print(f"\n--- {label} ({module}) ---")
    result = subprocess.run([sys.executable, "-m", module], cwd=ROOT)
    ok = result.returncode == 0
    if not ok:
        print(f"  [!] {label} failed (exit {result.returncode}) — continuing with the rest.",
              file=sys.stderr)
    return ok


def _check_config() -> None:
    print("\n--- Configuration check ---")
    backend = os.environ.get("GENERESOLVER_LLM_BACKEND", "medgemma")
    print(f"  GENERESOLVER_LLM_BACKEND = {backend!r} (default: medgemma, local)")
    if backend == "medgemma":
        base_url = os.environ.get("GENERESOLVER_MEDGEMMA_BASE_URL")
        print(f"  GENERESOLVER_MEDGEMMA_BASE_URL = {base_url or '(not set — client default)'}")
        if not base_url:
            print("    -> if the demo hangs or fails to connect, set this to your local "
                  "vLLM MedGemma endpoint.")
    elif backend == "claude":
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        print(f"  ANTHROPIC_API_KEY = {'set' if has_key else 'NOT SET'}")
        if not has_key:
            print("    -> required for backend=claude. Set it or switch back to medgemma.")
    local_mode = os.environ.get("GENERESOLVER_LOCAL_MODE", "0")
    print(f"  GENERESOLVER_LOCAL_MODE = {local_mode!r} "
          f"(1 = fully on-premise: disables gnomAD/PubCaseFinder/Monarch API calls, "
          f"forces local VEP)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-demo", action="store_true",
                     help="fetch data + check config only, don't run the demo case at the end")
    ap.add_argument("--with-vep-cache", action="store_true",
                     help="also fetch the ~23 GB local VEP cache + Docker image "
                          "(needed for the +VCF arm's on-premise path)")
    args = ap.parse_args()

    steps = list(_DOWNLOAD_STEPS)
    print("=== GeneResolver data setup ===")
    print(f"{len(steps)} data sources, then a config check"
          + ("" if args.skip_demo else " and a demo run")
          + (", plus the VEP cache" if args.with_vep_cache else "") + ".\n")

    results = [_run_step(m, label) for m, label in steps]
    if args.with_vep_cache:
        results.append(_run_step("scripts.download_vep_cache", "VEP local cache + Docker image (~23 GB)"))
    _check_config()

    n_ok = sum(results)
    print(f"\n=== data setup: {n_ok}/{len(results)} steps OK ===")
    if n_ok < len(results):
        print("Some steps failed — re-run this script to retry just those (the rest are "
              "already cached and will be skipped), or fetch manually per the error above.")

    if not args.with_vep_cache:
        print("\nNot fetched (opt-in / evaluation-only, not part of this repo):")
        print("  - Ensembl VEP local cache + Docker image (~23 GB, +VCF arm only) — "
              "pass --with-vep-cache (see scripts/download_vep_cache.py)")
        print("  - GSC+ phenotype-extraction corpus, PhEval Phenopacket Store corpus, "
              "monarch_pheval sibling repo")

    if args.skip_demo:
        return

    print("\n=== running the demo case to confirm the pipeline works ===")
    subprocess.run([sys.executable, "run_generesolver.py"], cwd=ROOT)


if __name__ == "__main__":
    main()
