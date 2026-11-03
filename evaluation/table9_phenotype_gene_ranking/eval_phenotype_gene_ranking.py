from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from pheval.post_processing.post_processing import SortOrder, generate_gene_result
from pheval.runners.runner import PhEvalRunner
from pheval.utils.phenopacket_utils import (
    GeneIdentifierUpdater,
    PhenopacketUtil,
    create_gene_identifier_map,
    phenopacket_reader,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.common import pheval_scoring, stats

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

_GENE_RESULT_SCHEMA = {"gene_symbol": pl.Utf8, "gene_identifier": pl.Utf8, "score": pl.Float64}


def generesolver_rank_genes(hpo_ids: list[str], fast: bool = True) -> list[tuple[str, float]]:
    from generesolver import orchestrator
    from generesolver.schemas.case import CaseInput, CaseState

    state = CaseState(case_id="pheval", input=CaseInput(case_id="pheval", hpo_terms=list(hpo_ids)))
    if fast:
        state.stop_after_step = "gene_ranker"
    state = orchestrator.run(state)
    return [(g.gene, float(getattr(g, "final_score", 0.0) or 0.0))
            for g in sorted(state.ranked_genes or [], key=lambda x: x.rank)]


@dataclass
class GeneResolverPhEvalRunner(PhEvalRunner):
    input_dir: Path
    testdata_dir: Path
    tmp_dir: Path
    output_dir: Path
    config_file: Path
    version: str

    def prepare(self):
        os.environ.setdefault("GENERESOLVER_LLM_BACKEND", "medgemma")
        os.environ.setdefault("GENERESOLVER_CASE_BRANCHING", "0")
        self._gid = GeneIdentifierUpdater(
            gene_identifier="ensembl_id", identifier_map=create_gene_identifier_map())
        self._raw = Path(self.tmp_dir) / "generesolver_raw"
        self._raw.mkdir(parents=True, exist_ok=True)
        (Path(self.output_dir) / "pheval_gene_results").mkdir(parents=True, exist_ok=True)

    def _phenopacket_dir(self) -> Path:
        d = Path(self.testdata_dir) / "phenopackets"
        return d if d.exists() else Path(self.testdata_dir)

    def run(self):
        pp_dir = self._phenopacket_dir()
        files = sorted(pp_dir.glob("*.json"))
        cached = sum(1 for pp in files if (self._raw / f"{pp.stem}.json").exists())
        todo = len(files) - cached
        print(f"  [run] {len(files)} cases total — {cached} already cached (resuming), "
              f"{todo} left to run", file=sys.stderr)
        errors = 0
        done = 0
        bar = tqdm(files, desc="PhEval phenotype-only", unit="case", file=sys.stderr)
        for pp in bar:
            out = self._raw / f"{pp.stem}.json"
            if out.exists():
                continue
            try:
                ppkt = phenopacket_reader(pp)
                hpo = [pf.type.id for pf in PhenopacketUtil(ppkt).observed_phenotypic_features()]
                ranked = generesolver_rank_genes(hpo)
            except Exception as exc:
                tqdm.write(f"  [run] {pp.stem}: ERROR {exc}", file=sys.stderr)
                ranked = []
                errors += 1
            tmp = out.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(ranked))
            os.replace(tmp, out)
            done += 1
            bar.set_postfix(errors=errors)
        if todo:
            print(f"  [run] done — {done} cases run this session, {errors} errors, "
                  f"{cached + done}/{len(files)} total cached", file=sys.stderr)

    def post_process(self):
        pp_dir = self._phenopacket_dir()
        for pp in sorted(pp_dir.glob("*.json")):
            raw_f = self._raw / f"{pp.stem}.json"
            try:
                ranked = json.loads(raw_f.read_text()) if raw_f.exists() else []
            except Exception:
                ranked = []
            rows = []
            for sym, score in ranked:
                try:
                    ident = self._gid.find_identifier(sym)
                except Exception:
                    ident = None
                rows.append({"gene_symbol": sym, "gene_identifier": ident or sym, "score": float(score)})
            df = pl.DataFrame(rows, schema=_GENE_RESULT_SCHEMA) if rows else pl.DataFrame(schema=_GENE_RESULT_SCHEMA)
            generate_gene_result(
                results=df, sort_order=SortOrder.DESCENDING,
                output_dir=Path(self.output_dir), result_path=pp, phenopacket_dir=pp_dir)


def _stage_corpus(corpus_pp: Path, n: int, dest: Path) -> int:
    import shutil
    pp_out = dest / "phenopackets"
    pp_out.mkdir(parents=True, exist_ok=True)
    files = sorted(corpus_pp.glob("*.json"))[:n] if n else sorted(corpus_pp.glob("*.json"))
    for f in files:
        if not (pp_out / f.name).exists():
            shutil.copy(f, pp_out / f.name)
    return len(files)


_CORPORA = ROOT / "external" / "pheval_corpora" / "corpora"
PHENO_ONLY = _CORPORA / "phenopacket_store_0.1.11_phenotypes"
GENOMIC = _CORPORA / "phenopacket_store_0.1.11_variants"


def _write_run_yaml(base: Path, run_id: str, phenopacket_dir: Path, results_dir: Path) -> Path:
    yaml_text = f"""\
benchmark_name: {run_id}
runs:
  - run_identifier: {run_id}
    phenopacket_dir: {phenopacket_dir.as_posix()}
    results_dir: {results_dir.as_posix()}
    gene_analysis: true
    variant_analysis: false
    disease_analysis: false
    threshold:
    score_order: descending
plot_customisation:
  gene_plots:
    plot_type: bar_cumulative
    rank_plot_title: GeneResolver — gene rank
    roc_curve_title: GeneResolver — ROC
    precision_recall_title: GeneResolver — precision/recall
  disease_plots:
    plot_type: bar_cumulative
    rank_plot_title: GeneResolver — disease rank
    roc_curve_title: GeneResolver — ROC
    precision_recall_title: GeneResolver — precision/recall
  variant_plots:
    plot_type: bar_cumulative
    rank_plot_title: GeneResolver — variant rank
    roc_curve_title: GeneResolver — ROC
    precision_recall_title: GeneResolver — precision/recall
"""
    out_path = base / "run_config.yaml"
    out_path.write_text(yaml_text, encoding="utf-8")
    return out_path


def _main():
    if len(sys.argv) <= 1:
        sys.exit(
            "usage: python -m evaluation.table9_phenotype_gene_ranking.eval_phenotype_gene_ranking N [corpus_dir]\n"
            "  N = 0 runs the FULL 4,677-case corpus (hours). Pass a small number (e.g. 20) for a quick\n"
            "  check instead. No default is assumed on purpose so this never launches the full run by\n"
            "  accident -- pass 0 explicitly if that's really what you want."
        )
    n = int(sys.argv[1])
    corpus = Path(sys.argv[2]) if len(sys.argv) > 2 else PHENO_ONLY
    corpus_pp = corpus / "phenopackets"
    if not corpus_pp.exists():
        sys.exit(f"corpus not found: {corpus_pp}")
    base = Path(os.environ.get("GENERESOLVER_STAGE4_RUN_DIR") or ROOT / "eval_runs" / f"_pheval_{corpus.name}")
    out = base / "results"
    out.mkdir(parents=True, exist_ok=True)
    testdata = base / "testdata" if n else corpus
    if n:
        _stage_corpus(corpus_pp, n, base / "testdata")
    print(f"corpus: {corpus.name}  ({n or len(list(corpus_pp.glob('*.json')))} phenopackets)")
    (base / "config.yaml").write_text(
        "tool: generesolver\ntool_version: 0.1\nvariant_analysis: false\n"
        "gene_analysis: true\ndisease_analysis: false\n"
        "tool_specific_configuration_options:\n", encoding="utf-8")
    runner = GeneResolverPhEvalRunner(
        input_dir=base, testdata_dir=testdata, tmp_dir=base / "tmp",
        output_dir=out, config_file=base / "config.yaml", version="0.1")
    runner.prepare(); runner.run(); runner.post_process()
    res = list(out.rglob("*gene_result.parquet")) + list(out.rglob("*pheval_gene_result.tsv"))
    print(f"wrote {len(res)} PhEval gene-result files -> {out/'pheval_gene_results'}")
    if res:
        print(f"sample: {res[0].name}")

    run_yaml = _write_run_yaml(
        base, run_id=f"generesolver_{corpus.name}",
        phenopacket_dir=testdata / "phenopackets",
        results_dir=out,
    )
    print(f"\nWrote a ready-to-use pheval-utils config -> {run_yaml}")

    metrics: dict = {}
    try:
        summary = pheval_scoring.run_benchmark(run_yaml, base / "benchmark")
        hits = pheval_scoring.per_case_hits_from_results(out)
        r1_ci = stats.bootstrap_ci(pheval_scoring.hit_vector(hits, k=1))
        metrics = {"summary": summary, "recall_at_1_ci95": [r1_ci[1], r1_ci[2]],
                   "n_scored_for_ci": len(hits)}
        n_total = summary.get("total") or summary.get("number_of_samples") or len(hits)
        print(f"\nScored {n_total} case(s) with the paper's PhEval metric definitions:")
        print(f"  Recall@1={summary.get('top1', 0) / n_total:.3f}  "
              f"(95% CI {r1_ci[1]:.3f}-{r1_ci[2]:.3f})  "
              f"Recall@3={summary.get('top3', 0) / n_total:.3f}  "
              f"Recall@5={summary.get('top5', 0) / n_total:.3f}  "
              f"Recall@10={summary.get('top10', 0) / n_total:.3f}  "
              f"MRR={summary.get('mrr', 0):.3f}")
    except Exception as exc:
        raise RuntimeError(
            f"PhEval scoring failed: {exc}. Retry with "
            f"`cd {base / 'benchmark'} && pheval-utils benchmark -r {run_yaml}`"
        ) from exc

    metrics_path = base / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    _main()
