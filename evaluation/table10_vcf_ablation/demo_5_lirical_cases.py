from __future__ import annotations

import copy
import sys
from pathlib import Path

from pheval.utils.phenopacket_utils import PhenopacketUtil, phenopacket_reader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.table10_vcf_ablation import eval_genome_vcf_ranking as evr

N_CASES = 5
_SNAPSHOT_KEY = "_demo_pre_acmg_ranked_genes"


def _causal_gene(pp: Path) -> str | None:
    genes = PhenopacketUtil(phenopacket_reader(pp)).diagnosed_genes()
    return genes[0].gene_symbol if genes else None


def _install_gene_ranker_snapshot() -> None:
    from generesolver.components import gene_ranker

    _orig_run = gene_ranker.run

    def _wrapped(state):
        state = _orig_run(state)
        state.diagnostics[_SNAPSHOT_KEY] = copy.deepcopy(state.ranked_genes)
        return state

    gene_ranker.run = _wrapped


def _run_once(case_id: str, hpos: list[str], vcf_path: str):
    from generesolver import orchestrator
    from generesolver.schemas.case import CaseInput, CaseState
    inp = CaseInput(case_id=case_id, hpo_terms=list(hpos), vcf_path=vcf_path)
    state = CaseState(case_id=case_id, input=inp)
    state.stop_after_step = "acmg_rerank"
    return orchestrator.run(state)


def _rank_of(ranked_genes, gene: str) -> int | None:
    for g in ranked_genes:
        if g.gene == gene:
            return g.rank
    return None


def _top(ranked_genes, k: int = 5) -> str:
    top = sorted(ranked_genes, key=lambda g: g.rank)[:k]
    return ", ".join(f"{g.rank}.{g.gene}({g.final_score:.3f})" for g in top)


def main() -> None:
    import os
    os.environ.setdefault("GENERESOLVER_LLM_BACKEND", evr.CONFIG.BACKEND)

    print("== setup: VEP cache + GWAS cache + local-mode patches ==")
    evr._cache_load(evr.CONFIG.BASE / "vep_cache_local.json.gz")
    evr._gwas_cache_load(evr.CONFIG.BASE / "gwas_cache.json.gz")
    evr._install_patches()
    _install_gene_ranker_snapshot()
    evr._local_vep_selftest()

    evr.CONFIG.LIMIT = N_CASES
    vcf_dir = evr.CONFIG.BASE / "spiked_vcfs"
    cases = evr._load_cases(evr.CONFIG, vcf_dir)
    print(f"\n== {len(cases)} case(s) selected ==")

    for cs in cases:
        causal_gene = _causal_gene(cs.pp)
        print("\n" + "=" * 90)
        print(f"case: {cs.case_id}")
        print(f"  input HPO terms ({len(cs.hpos)}): {', '.join(cs.hpos[:8])}"
              + (" ..." if len(cs.hpos) > 8 else ""))
        print(f"  spiked VCF: {cs.vcf}")
        print(f"  causal gene (phenopacket truth): {causal_gene}")

        result = _run_once(cs.case_id, cs.hpos, str(cs.vcf))

        pre_ranked = result.diagnostics.get(_SNAPSHOT_KEY) or []
        pre_rank = _rank_of(pre_ranked, causal_gene) if causal_gene else None
        print(f"  PRE-ACMG  (gene_ranker) top 5: {_top(pre_ranked)}")
        print(f"  PRE-ACMG  causal gene rank: {pre_rank}")

        post_ranked = result.ranked_genes or []
        post_rank = _rank_of(post_ranked, causal_gene) if causal_gene else None
        print(f"  POST-ACMG (acmg_rerank) top 5: {_top(post_ranked)}")
        print(f"  POST-ACMG causal gene rank: {post_rank}")

        if pre_rank is not None and post_rank is not None:
            delta = pre_rank - post_rank
            verdict = "IMPROVED" if delta > 0 else ("WORSE" if delta < 0 else "UNCHANGED")
            print(f"  -> rank change: {pre_rank} -> {post_rank} ({verdict})")

        causal_assessments = [
            a for a in (result.acmg_assessments or [])
            if a.computed_acmg_assessment.gene == causal_gene
        ]
        if causal_assessments:
            print(f"  ACMG assessment(s) for {causal_gene}:")
            for a in causal_assessments:
                ca = a.computed_acmg_assessment
                print(f"    disease={ca.disease_id} classification={ca.classification} "
                      f"points={ca.total_points} counted_criteria={ca.counted_criteria} "
                      f"not_evaluated={ca.criteria_not_evaluated[:6]}...")
        else:
            print(f"  ACMG assessment(s) for {causal_gene}: none produced "
                  "(no matching annotated variant / variant-disease context)")

    evr._cache_save()
    evr._gwas_cache_save()
    print("\n== done ==")


if __name__ == "__main__":
    sys.exit(main())
