from __future__ import annotations

import gzip
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import polars as pl

from pheval.post_processing.post_processing import (
    PhenopacketTruthSet,
    ResultType,
    SortOrder,
    create_empty_pheval_result,
    executed_results,
    generate_gene_result,
)
from pheval.utils.phenopacket_utils import (
    GeneIdentifierUpdater,
    PhenopacketUtil,
    create_gene_identifier_map,
    phenopacket_reader,
)

# CRITICAL: generate_gene_result() -> create_empty_pheval_result() sweeps EVERY phenopacket
# in the corpus and overwrites its parquet with an empty "not found" placeholder, but ONLY
# guards against doing this twice via a module-level `executed_results` set -- so the *first*
# generate_gene_result() call in any fresh process destructively clobbers every already-correct
# result already on disk for this result type before writing the one real result it was asked
# for. Pre-seeding the guard here makes that sweep permanently a no-op: every case we care
# about is emitted explicitly by our own loop below anyway, so we never needed the sweep's
# false-negative bookkeeping in the first place. This must run before the first
# generate_gene_result() call in this process (verified directly against pheval-utils
# 0.5.x source -- see post_processing.py's create_empty_pheval_result/executed_results).
executed_results.add(ResultType.GENE)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.common import pheval_scoring, stats

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

os.environ.setdefault("GENERESOLVER_LOCAL_MODE", "1")

os.environ.setdefault("GENERESOLVER_CASE_BRANCHING", "0")

from generesolver.components import vcf_input as _vcf_input
_vcf_input._MAX_VARIANTS = 0


# pheval-utils' generate_gene_result() has a real race under tight back-to-back calls
# with no delay -- verified: an unpaced loop silently dropped the parquet write for 317
# of 375 cases. Pacing calls, then verifying + individually retrying anything still
# missing, eliminates it (both steps needed -- pacing alone still leaves stragglers).
_EMIT_DELAY_S = 1.0


class CONFIG:
    WARM_TEMPLATE = True
    _CORPUS_ROOT = (ROOT / "external" / "monarch_pheval"
                    if (ROOT / "external" / "monarch_pheval").exists()
                    else ROOT.parent / "monarch_pheval")
    LIRICAL_PP = _CORPUS_ROOT / "corpora" / "lirical" / "default" / "phenopackets"
    TEMPLATE = _CORPUS_ROOT / "testdata" / "template_vcf" / "template_exome_hg19.vcf.gz"
    # The revised variant filter never writes into the original (reported) run directory.
    FILTER_PROFILE = os.environ.get("GENERESOLVER_VARIANT_FILTER_PROFILE", "original").strip().lower()
    BASE = Path(os.environ.get(
        "GENERESOLVER_STAGE5_RUN_DIR",
        ROOT / "eval_runs" / ("_lirical_vcf" if FILTER_PROFILE == "original"
                              else f"_lirical_vcf_{FILTER_PROFILE}_filter")))
    # Optional earlier run whose spiked VCFs and VEP/gnomAD/GWAS caches are COPIED (never
    # modified) into BASE. VEP flags are identical across filter profiles, so reuse is valid.
    CACHE_SEED_DIR = os.environ.get("GENERESOLVER_STAGE5_CACHE_SEED_DIR") or None
    LIMIT = int(os.environ.get("GENERESOLVER_VCF_LIMIT", "0"))
    RUN_PHENO_ONLY_ARM = True
    BACKGROUND_LIMIT = 0
    CACHE_SAVE_EVERY = 150
    BACKEND = os.environ.get("GENERESOLVER_LLM_BACKEND", "medgemma")


_CACHE: dict[str, dict] = {}
_CACHE_PATH: Path | None = None
_CACHE_DIRTY = 0


def _cache_load(path: Path) -> None:
    global _CACHE, _CACHE_PATH
    _CACHE_PATH = path
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            _CACHE = json.load(f)
        print(f"  [cache] loaded {len(_CACHE)} VEP annotations from {path.name}")


def _cache_save() -> None:
    if _CACHE_PATH is None:
        return
    tmp = _CACHE_PATH.with_suffix(".tmp.gz")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(_CACHE, f)
    tmp.replace(_CACHE_PATH)


_GNOMAD_POPMAX_CACHE: dict[str, dict] = {}
_GNOMAD_POPMAX_CACHE_PATH: Path | None = None


def _gnomad_popmax_cache_load(path: Path) -> None:
    global _GNOMAD_POPMAX_CACHE, _GNOMAD_POPMAX_CACHE_PATH
    _GNOMAD_POPMAX_CACHE_PATH = path
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            _GNOMAD_POPMAX_CACHE = json.load(f)
        print(f"  [cache] loaded {len(_GNOMAD_POPMAX_CACHE)} gnomAD-popmax/SpliceAI lookups from {path.name}")


def _gnomad_popmax_cache_save() -> None:
    if _GNOMAD_POPMAX_CACHE_PATH is None:
        return
    tmp = _GNOMAD_POPMAX_CACHE_PATH.with_suffix(".tmp.gz")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(_GNOMAD_POPMAX_CACHE, f)
    tmp.replace(_GNOMAD_POPMAX_CACHE_PATH)


_GWAS_CACHE: dict[str, list[dict]] = {}
_GWAS_CACHE_PATH: Path | None = None
_GWAS_MAX_WORKERS = 5


def _gwas_cache_load(path: Path) -> None:
    global _GWAS_CACHE, _GWAS_CACHE_PATH
    _GWAS_CACHE_PATH = path
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            _GWAS_CACHE = json.load(f)
        print(f"  [cache] loaded {len(_GWAS_CACHE)} GWAS Catalog lookups from {path.name}")


def _gwas_cache_save() -> None:
    if _GWAS_CACHE_PATH is None:
        return
    tmp = _GWAS_CACHE_PATH.with_suffix(".tmp.gz")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(_GWAS_CACHE, f)
    tmp.replace(_GWAS_CACHE_PATH)


def _install_gwas_patches() -> None:
    from concurrent.futures import ThreadPoolExecutor

    from generesolver.integrations import gwas_catalog
    from generesolver.components import gwas_lookup as gwas_lookup_mod
    from generesolver.schemas.variant import GWASEvidence

    _orig_lookup = gwas_catalog.lookup

    def _cached_lookup(rsid: str | None = None, gene: str | None = None) -> list[dict]:
        key = f"rsid:{rsid}" if rsid else f"gene:{gene}"
        if key in _GWAS_CACHE:
            return _GWAS_CACHE[key]
        result = _orig_lookup(rsid=rsid, gene=gene)
        _GWAS_CACHE[key] = result
        return result

    gwas_catalog.lookup = _cached_lookup

    def _parallel_run(state):
        genes: list[str] = [gr.gene for gr in state.ranked_genes]
        if state.input.candidate_gene and state.input.candidate_gene not in genes:
            genes.append(state.input.candidate_gene)
        if not genes:
            state.mark_done("gwas_lookup")
            return state

        results: dict[str, list[dict]] = {}
        with ThreadPoolExecutor(max_workers=_GWAS_MAX_WORKERS) as pool:
            futures = {pool.submit(gwas_catalog.lookup, gene=g): g for g in genes}
            for fut, g in futures.items():
                try:
                    results[g] = fut.result()
                except Exception as exc:
                    logging.getLogger(gwas_lookup_mod.__name__).warning(
                        "GWAS lookup failed for gene %s: %s", g, exc)
                    results[g] = []

        all_evidence = list(state.gwas_evidence)
        for gene in genes:
            for item in results.get(gene, []):
                all_evidence.append(GWASEvidence(
                    query=item.get("query", gene),
                    trait=item.get("trait"),
                    p_value=gwas_lookup_mod._safe_float(item.get("p_value")),
                    odds_ratio=gwas_lookup_mod._safe_float(item.get("odds_ratio")),
                    support_level=item.get("support_level", "supportive_context_only"),
                    source_db=item.get("source_db"),
                    warning=item.get("warning", GWASEvidence.model_fields["warning"].default),
                ))
        state.gwas_evidence = all_evidence
        state.mark_done("gwas_lookup")
        return state

    gwas_lookup_mod.run = _parallel_run


def _install_patches() -> None:
    import generesolver.config as gconfig
    from generesolver.integrations import vep_api
    from generesolver.components import variant_annotator as va
    from generesolver.components import vcf_input as vi

    _orig_annotate = vep_api.annotate_variants_local

    def _cached_annotate(variants, canonical_only=True, genome_build=None):
        global _CACHE_DIRTY
        build = genome_build or "GRCh37"
        # Existing cache entries are unprefixed GRCh37 keys; other builds get their own namespace.
        prefix = "" if build == "GRCh37" else f"{build}:"
        out, miss = {}, []
        for c, p, r, a in variants:
            k = f"{str(c).lstrip('chr')}:{p}:{r}:{a}"
            if prefix + k in _CACHE:
                out[k] = _CACHE[prefix + k]
            else:
                miss.append((c, p, r, a))
        if miss:
            got = _orig_annotate(miss, canonical_only, build)
            for c, p, r, a in miss:
                k = f"{str(c).lstrip('chr')}:{p}:{r}:{a}"
                ann = got.get(k, {})
                _CACHE[prefix + k] = ann
                out[k] = ann
            _CACHE_DIRTY += 1
        return out

    vep_api.annotate_variants = _cached_annotate

    from generesolver.integrations import gnomad_popmax_local
    _orig_gnomad_batch = gnomad_popmax_local.annotate_batch

    def _cached_gnomad_batch(variants, genome_build):
        out, miss = {}, []
        for c, p, r, a in variants:
            k = f"{str(c).lstrip('chr')}:{p}:{r}:{a}"
            if k in _GNOMAD_POPMAX_CACHE:
                out[k] = _GNOMAD_POPMAX_CACHE[k]
            else:
                miss.append((c, p, r, a))
        if miss:
            got = _orig_gnomad_batch(miss, genome_build)
            for c, p, r, a in miss:
                k = f"{str(c).lstrip('chr')}:{p}:{r}:{a}"
                ann = got.get(k, {})
                _GNOMAD_POPMAX_CACHE[k] = ann
                out[k] = ann
        return out

    gnomad_popmax_local.annotate_batch = _cached_gnomad_batch

    _orig_lpv = vi.load_pass_variants
    def _uncapped(vcf_path, max_variants=20_000_000):
        return _orig_lpv(vcf_path, max_variants)
    vi.load_pass_variants = _uncapped
    va.load_pass_variants = _uncapped

    _orig_lnv = va.load_normalized_variants
    def _uncapped_normalized(vcf_path, genome_build=None, max_variants=20_000_000, **kw):
        return _orig_lnv(vcf_path, genome_build, max_variants, **kw)
    va.load_normalized_variants = _uncapped_normalized

    _install_gwas_patches()


def _local_vep_selftest() -> None:
    from generesolver.integrations import vep_api
    res = vep_api.annotate_variants_local([("17", 41245466, "G", "A")])
    gene = (next(iter(res.values()), {}) or {}).get("gene") if res else None
    if gene != "BRCA1":
        raise SystemExit(
            f"Local VEP self-test FAILED — expected BRCA1, got {gene!r}.\n"
            "  Check: Docker Desktop running | image ensemblorg/ensembl-vep pulled |\n"
            "  cache at data/vep_cache/homo_sapiens/116_GRCh37/. Aborting before the run.")
    print("  [selftest] local VEP OK (BRCA1)")


def _iter_template_variants(template: Path, limit: int = 0):
    op = gzip.open(template, "rt") if str(template).endswith(".gz") else open(template, "rt")
    n = 0
    with op as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.split("\t")
            if len(c) < 5:
                continue
            filt = c[6] if len(c) > 6 else "PASS"
            if filt not in ("PASS", "."):
                continue
            yield (c[0], int(c[1]), c[3], c[4])
            n += 1
            if limit and n >= limit:
                return


def _warm_template_cache(cfg) -> None:
    from generesolver.integrations import vep_api
    variants = list(_iter_template_variants(cfg.TEMPLATE, cfg.BACKGROUND_LIMIT))
    todo = [v for v in variants if f"{v[0].lstrip('chr')}:{v[1]}:{v[2]}:{v[3]}" not in _CACHE]
    print(f"  [warm] template has {len(variants)} variants; {len(variants)-len(todo)} cached, "
          f"{len(todo)} to annotate")
    CHUNK = 20_000
    t0 = time.time()
    for i in range(0, len(todo), CHUNK):
        vep_api.annotate_variants(todo[i:i + CHUNK])
        _cache_save()
        done = min(i + CHUNK, len(todo))
        rate = done / max(time.time() - t0, 1)
        print(f"  [warm] {done}/{len(todo)} (~{rate:.0f} var/s, cache={len(_CACHE)})",
              file=sys.stderr)
    _cache_save()
    print(f"  [warm] done — {len(_CACHE)} annotations cached")


def _causal_variants(cfg) -> list[tuple[str, int, str, str]]:
    out: dict[tuple[str, int, str, str], None] = {}
    for pp in sorted(cfg.LIRICAL_PP.glob("*.json")):
        data = json.loads(pp.read_text(encoding="utf-8"))
        for interp in data.get("interpretations", []):
            for gi in interp.get("diagnosis", {}).get("genomicInterpretations", []):
                vr = gi.get("variantInterpretation", {}).get("variationDescriptor", {}).get("vcfRecord")
                if not vr:
                    continue
                out[(vr["chrom"], int(vr["pos"]), vr["ref"], vr["alt"])] = None
    return list(out)


def _warm_causal_variants(cfg) -> None:
    from generesolver.integrations import vep_api
    variants = _causal_variants(cfg)
    todo = [v for v in variants if f"{v[0].lstrip('chr')}:{v[1]}:{v[2]}:{v[3]}" not in _CACHE]
    print(f"  [warm-causal] {len(variants)} distinct causal variant(s) across all cases; "
          f"{len(variants) - len(todo)} cached, {len(todo)} to annotate")
    if todo:
        vep_api.annotate_variants(todo)
        _cache_save()
    print(f"  [warm-causal] done — {len(_CACHE)} annotations cached")


def _warm_gnomad_popmax_template(cfg) -> None:
    from generesolver.integrations import gnomad_popmax_local
    variants = list(_iter_template_variants(cfg.TEMPLATE, cfg.BACKGROUND_LIMIT))
    todo = [v for v in variants if f"{v[0].lstrip('chr')}:{v[1]}:{v[2]}:{v[3]}" not in _GNOMAD_POPMAX_CACHE]
    print(f"  [warm-gnomad] template has {len(variants)} variants; {len(variants)-len(todo)} cached, "
          f"{len(todo)} to annotate")
    CHUNK = 20_000
    t0 = time.time()
    for i in range(0, len(todo), CHUNK):
        gnomad_popmax_local.annotate_batch(todo[i:i + CHUNK], "GRCh37")
        _gnomad_popmax_cache_save()
        done = min(i + CHUNK, len(todo))
        rate = done / max(time.time() - t0, 1)
        print(f"  [warm-gnomad] {done}/{len(todo)} (~{rate:.0f} var/s, cache={len(_GNOMAD_POPMAX_CACHE)})",
              file=sys.stderr)
    _gnomad_popmax_cache_save()
    print(f"  [warm-gnomad] done — {len(_GNOMAD_POPMAX_CACHE)} annotations cached")


def _warm_gnomad_popmax_causal(cfg) -> None:
    from generesolver.integrations import gnomad_popmax_local
    variants = _causal_variants(cfg)
    todo = [v for v in variants if f"{v[0].lstrip('chr')}:{v[1]}:{v[2]}:{v[3]}" not in _GNOMAD_POPMAX_CACHE]
    print(f"  [warm-gnomad-causal] {len(variants)} distinct causal variant(s) across all cases; "
          f"{len(variants) - len(todo)} cached, {len(todo)} to annotate")
    if todo:
        gnomad_popmax_local.annotate_batch(todo, "GRCh37")
        _gnomad_popmax_cache_save()
    print(f"  [warm-gnomad-causal] done — {len(_GNOMAD_POPMAX_CACHE)} annotations cached")


def _rank(case_id: str, hpo_ids: list[str], vcf_path: str | None) -> list[tuple[str, float]]:
    from generesolver import orchestrator
    from generesolver.schemas.case import CaseInput, CaseState
    inp = CaseInput(case_id=case_id, hpo_terms=list(hpo_ids), vcf_path=vcf_path)
    state = CaseState(case_id=case_id, input=inp)
    state.stop_after_step = "acmg_rerank" if vcf_path else "gene_ranker"
    state = orchestrator.run(state)
    return [(g.gene, float(getattr(g, "final_score", 0.0) or 0.0))
            for g in sorted(state.ranked_genes or [], key=lambda x: x.rank)]


@dataclass
class _Case:
    pp: Path
    case_id: str
    hpos: list[str]
    vcf: Path


def _load_cases(cfg, vcf_dir: Path) -> list[_Case]:
    out = []
    for pp in sorted(cfg.LIRICAL_PP.glob("*.json")):
        vcf = vcf_dir / f"{pp.stem}.vcf.gz"
        if not vcf.exists():
            continue
        hpos = [p.type.id for p in PhenopacketUtil(phenopacket_reader(pp)).observed_phenotypic_features()]
        out.append(_Case(pp, pp.stem, hpos, vcf))
        if cfg.LIMIT and len(out) >= cfg.LIMIT:
            break
    return out


_SCHEMA = {"gene_symbol": pl.Utf8, "gene_identifier": pl.Utf8, "score": pl.Float64}


def _ensure_truth_placeholders(phenopacket_dir: Path, results_dir: Path) -> None:
    """The REAL root cause of the persistent 'parquet race' FileNotFoundError (confirmed by
    reading pheval-utils source directly): generate_gene_result() -> merge_gene_results() does
    `pl.read_parquet(output_file)` to fetch the ground-truth gene for THIS case -- reading from
    the very same path it's about to write to. It relies on create_empty_pheval_result() having
    already written a truth-only placeholder there first. That sweep is guarded by a
    module-level `executed_results` set so it only runs once per process -- fine for a single
    long-running run, but this codebase calls generate_gene_result() from many short-lived
    processes (retries, subprocess fallback) against the same results_dir, and each one starts
    with an empty guard: either it redundantly (and destructively) re-sweeps and clobbers every
    already-written real result back to a placeholder, or -- once we pre-seed the guard at
    import time to stop that -- it never writes the placeholder at all, so the read is a
    guaranteed FileNotFoundError for any case not yet processed. Neither behavior is a random
    race; both are deterministic consequences of the same design assumption (single process,
    one sweep, then per-case writes) being violated. The fix: do the placeholder sweep exactly
    ONCE ever per results_dir, explicitly, up front -- tracked by a disk sentinel so it survives
    across process restarts -- before any per-case emission (main loop, retry, or subprocess
    fallback) ever runs. Every later generate_gene_result() call can then assume its target
    file already exists (as a placeholder or a real result), so the internal read always
    succeeds and the FileNotFoundError can no longer occur at all.
    """
    sentinel = results_dir / ".truth_placeholders_written"
    if sentinel.exists():
        return
    gene_dir = results_dir / "pheval_gene_results"
    gene_dir.mkdir(parents=True, exist_ok=True)
    was_seeded = ResultType.GENE in executed_results
    executed_results.discard(ResultType.GENE)  # force the sweep to actually run this once
    try:
        create_empty_pheval_result(phenopacket_dir, gene_dir, ResultType.GENE)
    finally:
        if was_seeded:
            executed_results.add(ResultType.GENE)
    sentinel.write_text("ok", encoding="utf-8")
    print(f"  [truth] wrote ground-truth placeholders for {len(list(phenopacket_dir.glob('*.json')))} "
          f"phenopackets to {gene_dir}")


def _emit_pheval(ranked, pp: Path, out_dir: Path, gid: GeneIdentifierUpdater) -> None:
    # Idempotency guard: skip only if a REAL result (correct row count -- see _row_count_ok)
    # is already on disk. A file merely existing isn't enough: _ensure_truth_placeholders()
    # deliberately writes a 1-row ground-truth placeholder for every case up front, and that
    # placeholder must not be mistaken for an already-emitted real result or every case would
    # get silently skipped.
    existing = out_dir / "pheval_gene_results" / f"{pp.stem}-gene_result.parquet"
    if existing.exists():
        if _row_count_ok(_parquet_row_count(out_dir, pp.stem), len(ranked)):
            return
        # existing.exists() but wrong row count: generate_gene_result()'s merge_gene_results()
        # reads THIS SAME file to learn the ground truth before overwriting it -- if what's
        # there is a stale/corrupted REAL result rather than the truth-only placeholder, it
        # gets misread as truth (every candidate gene "is_in" the old result's full gene list),
        # which is exactly the all-true_positive corruption seen earlier this session. Never
        # delete it (that reintroduces the FileNotFoundError this whole design avoids) --
        # reset it back to the truth-only placeholder shape instead, then let the normal path
        # below correctly read-and-overwrite it.
        placeholder = PhenopacketTruthSet(cfg_pp_dir(pp)).classified_gene(pp.stem)
        placeholder.write_parquet(existing, compression="zstd")
    rows = []
    for sym, score in ranked:
        try:
            ident = gid.find_identifier(sym)
        except Exception:
            ident = None
        rows.append({"gene_symbol": sym, "gene_identifier": ident or sym, "score": float(score)})
    df = pl.DataFrame(rows, schema=_SCHEMA) if rows else pl.DataFrame(schema=_SCHEMA)
    (out_dir / "pheval_gene_results").mkdir(parents=True, exist_ok=True)
    # This race has proven stubborn -- for a small, reproducible subset of cases it recurs
    # even in a brand-new interpreter (not just under back-to-back calls in one long-running
    # process), so treat it as a slow-to-clear transient lock (most likely Windows Defender/AV
    # scanning newly-written parquet files in this directory) and retry patiently rather than
    # giving up after ~2s.
    _ATTEMPTS = 12
    for attempt in range(_ATTEMPTS):
        try:
            generate_gene_result(results=df, sort_order=SortOrder.DESCENDING, output_dir=out_dir,
                                 result_path=pp, phenopacket_dir=cfg_pp_dir(pp))
            return
        except FileNotFoundError:
            if attempt < _ATTEMPTS - 1:
                time.sleep(1.5)
            else:
                print(f"  [emit] {pp.stem}: parquet race after retries — skipping (raw saved)",
                      file=sys.stderr)


def cfg_pp_dir(pp: Path) -> Path:
    return pp.parent


def _spike_all(cfg, vcf_dir: Path) -> int:
    vcf_dir.mkdir(parents=True, exist_ok=True)
    have = len(list(vcf_dir.glob("*.vcf.gz")))
    total = len(list(cfg.LIRICAL_PP.glob("*.json")))
    if have >= (cfg.LIMIT or total):
        print(f"  [spike] {have} spiked VCFs present — skipping")
        return have
    print(f"  [spike] generating spiked VCFs via pheval-utils create-spiked-vcfs ...")
    cmd = [str(ROOT / ".venv" / "Scripts" / "pheval-utils"), "create-spiked-vcfs",
           "-P", str(cfg.LIRICAL_PP), "--hg19-template-vcf", str(cfg.TEMPLATE),
           "--output-dir", str(vcf_dir)]
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    import os
    subprocess.run(cmd, env={**os.environ, **env}, check=True)
    return len(list(vcf_dir.glob("*.vcf.gz")))


def _write_run_yaml(base: Path, suffix: str, run_id: str,
                     phenopacket_dir: Path, results_dir: Path) -> Path:
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
    out_path = base / f"run_config_{suffix}.yaml"
    out_path.write_text(yaml_text, encoding="utf-8")
    return out_path


def _parquet_row_count(results_dir: Path, case_id: str) -> int | None:
    f = results_dir / "pheval_gene_results" / f"{case_id}-gene_result.parquet"
    if not f.exists():
        return None
    import polars as _pl
    return _pl.read_parquet(f).height


def _row_count_ok(got_n: int | None, expected_n: int) -> bool:
    # See _bad_cases(): a correct emission is exactly len(candidates), or len(candidates)+1
    # when pheval-utils legitimately appended a rank=0 "not found" placeholder for the truth
    # gene.
    return got_n is not None and expected_n <= got_n <= expected_n + 1


def _verify_and_retry_emission(raw_dir: Path, results_dir: Path, cfg, gid, label: str) -> None:
    """pheval-utils' generate_gene_result() is unsafe under back-to-back calls in two
    distinct, separately-observed ways: it can silently skip the parquet write entirely
    (FileNotFoundError, case missing), or it can silently write a truncated 1-row 'not
    found' placeholder instead of the real N-row candidate list (case present but wrong
    row count) -- checking file existence alone misses the second failure mode entirely.
    This checks row count against the actual input candidate list for every case and
    individually re-emits (with backoff, deleting the bad file first) anything that
    doesn't match. Aborts rather than silently scoring a corrupted/partial arm."""
    result_dir = results_dir / "pheval_gene_results"
    raw_ids = sorted(f.stem for f in raw_dir.glob("*.json") if (cfg.LIRICAL_PP / f"{f.stem}.json").exists())

    def _bad_cases() -> list[str]:
        bad = []
        for case_id in raw_ids:
            expected_n = len(json.loads((raw_dir / f"{case_id}.json").read_text(encoding="utf-8")))
            got_n = _parquet_row_count(results_dir, case_id)
            if not _row_count_ok(got_n, expected_n):
                bad.append(case_id)
        return bad

    bad = _bad_cases()
    if not bad:
        print(f"  [{label}] {len(raw_ids)}/{len(raw_ids)} emitted with correct row counts, 0 bad")
        return
    print(f"  [{label}] {len(bad)} case(s) with missing/wrong-size parquet, retrying individually: {bad}")
    for case_id in bad:
        pp = cfg.LIRICAL_PP / f"{case_id}.json"
        ranked = json.loads((raw_dir / f"{case_id}.json").read_text(encoding="utf-8"))
        target = result_dir / f"{case_id}-gene_result.parquet"
        for attempt in range(4):
            # Do NOT unlink target here -- _emit_pheval() needs a truth-placeholder already at
            # that path to read (see _ensure_truth_placeholders / _emit_pheval's own reset
            # logic); deleting it just reintroduces the FileNotFoundError.
            _emit_pheval(ranked, pp, results_dir, gid)
            if _row_count_ok(_parquet_row_count(results_dir, case_id), len(ranked)):
                print(f"    {case_id}: recovered on in-process retry {attempt + 1}")
                break
            time.sleep(2.0)
        else:
            # In-process retries exhausted. Verified root cause: the exact same emission,
            # for the exact same case/data/output-dir, run in a FRESH interpreter succeeds
            # immediately -- this is contention against pheval_gene_results/ that only shows
            # up after hundreds of back-to-back calls in one long-running process, not
            # anything wrong with the case itself. Fall back to a one-shot subprocess.
            print(f"    {case_id}: still wrong after in-process retries, falling back to subprocess")
            for sub_attempt in range(3):
                proc = subprocess.run(
                    [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m",
                     "evaluation.table10_vcf_ablation._emit_single_case",
                     str(raw_dir), str(results_dir), case_id],
                    cwd=str(ROOT), capture_output=True, text=True,
                )
                if _row_count_ok(_parquet_row_count(results_dir, case_id), len(ranked)):
                    print(f"    {case_id}: recovered via subprocess (attempt {sub_attempt + 1})")
                    break
                print(f"    {case_id}: subprocess attempt {sub_attempt + 1} failed "
                      f"(rc={proc.returncode}); stderr tail: {proc.stderr[-300:]}")
                time.sleep(2.0)
            else:
                print(f"    {case_id}: STILL WRONG after in-process + subprocess retries")
    still_bad = _bad_cases()
    if still_bad:
        sys.exit(f"[{label}] ABORT: {len(still_bad)} case(s) never got a correctly-sized "
                  f"parquet after retries: {still_bad}. Fix and rerun (idempotent).")
    print(f"  [{label}] {len(raw_ids)}/{len(raw_ids)} emitted with correct row counts after retry, 0 bad")


_MANIFEST = "run_manifest.json"
_VEP_CACHE_FILE = "vep_cache_local_v2.json.gz"


def _guard_run_dir(cfg) -> None:
    """Never let one filter profile write into a directory holding another profile's results."""
    from generesolver import config as gconfig
    if gconfig.VARIANT_FILTER_PROFILE != cfg.FILTER_PROFILE:
        sys.exit("GENERESOLVER_VARIANT_FILTER_PROFILE changed after import; set it before launch.")
    man = cfg.BASE / _MANIFEST
    has_results = (cfg.BASE / "raw_with_vcf").exists() and any((cfg.BASE / "raw_with_vcf").iterdir())
    if man.exists():
        prev = json.loads(man.read_text()).get("variant_filter_profile")
        if prev != cfg.FILTER_PROFILE:
            sys.exit(f"ABORT: {cfg.BASE} holds results of filter profile {prev!r}; this run is "
                     f"{cfg.FILTER_PROFILE!r}. Use a separate GENERESOLVER_STAGE5_RUN_DIR.")
    elif has_results and cfg.FILTER_PROFILE != "original":
        sys.exit(f"ABORT: {cfg.BASE} already holds results from before run manifests existed "
                 "(original profile). Use a separate GENERESOLVER_STAGE5_RUN_DIR.")


def _seed_from(cfg) -> None:
    if not cfg.CACHE_SEED_DIR:
        return
    seed = Path(cfg.CACHE_SEED_DIR)
    if seed.resolve() == cfg.BASE.resolve():
        return
    if (seed / "spiked_vcfs").exists() and not (cfg.BASE / "spiked_vcfs").exists():
        shutil.copytree(seed / "spiked_vcfs", cfg.BASE / "spiked_vcfs")
    for name in (_VEP_CACHE_FILE, "gnomad_popmax_cache.json.gz", "gwas_cache.json.gz"):
        if (seed / name).exists() and not (cfg.BASE / name).exists():
            shutil.copy2(seed / name, cfg.BASE / name)
            print(f"  [seed] copied {name} from {seed} (source left untouched)")


def _write_manifest(cfg) -> None:
    from generesolver import config as gconfig
    from generesolver.integrations import vep_api, resource_versions
    meta = vep_api.annotation_metadata("GRCh37")
    # VEP cache entries carry no per-entry fingerprint; record the configuration that is
    # adding to this cache now, and refuse a later run whose configuration differs.
    fp_file = cfg.BASE / (_VEP_CACHE_FILE + ".fingerprint.json")
    if fp_file.exists():
        prev = json.loads(fp_file.read_text())
        if prev.get("fingerprint") != meta["fingerprint"]:
            sys.exit(f"ABORT: VEP configuration changed (cache built with {prev.get('fingerprint')}, "
                     f"now {meta['fingerprint']}). Use a fresh run dir or delete {_VEP_CACHE_FILE}.")
    else:
        fp_file.write_text(json.dumps({
            "fingerprint": meta["fingerprint"], "vep": meta,
            "note": ("Recorded when this file was first created. Entries already in the cache "
                     "at that point predate fingerprinting and are assumed, not proven, to come "
                     "from this configuration."),
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True,
                                text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT),
                                    capture_output=True, text=True).stdout.strip())
    except OSError:
        commit, dirty = None, None
    # Hash of the code actually imported, so a run from an uncommitted working copy can still
    # be matched to its exact source (the paper's ACMG engine could not be, see ACMG_RULES.md).
    import hashlib
    h, per_file = hashlib.sha256(), {}
    for p in sorted((ROOT / "generesolver").rglob("*.py")):
        b = p.read_bytes()
        h.update(p.relative_to(ROOT).as_posix().encode() + b"\0" + b)
        if p.name in ("acmg_engine.py", "variant_annotator.py", "gene_aggregator.py", "vep_api.py"):
            per_file[p.name] = hashlib.sha256(b).hexdigest()[:16]
    code_hash = {"generesolver_py_sha256": h.hexdigest(), "key_files": per_file}
    # A run directory belongs to ONE code version: a resumed run on edited code would mix
    # engine versions across cases (as happened on 2026-08-17). Abort instead.
    if (cfg.BASE / _MANIFEST).exists():
        prev_code = (json.loads((cfg.BASE / _MANIFEST).read_text()).get("code") or {}).get(
            "generesolver_py_sha256")
        if prev_code and prev_code != code_hash["generesolver_py_sha256"] \
                and os.environ.get("GENERESOLVER_STAGE5_ALLOW_CODE_CHANGE") != "1":
            sys.exit(f"ABORT: generesolver source changed since this run started "
                     f"({prev_code[:12]} -> {code_hash['generesolver_py_sha256'][:12]}). "
                     "Restore code_snapshot/ or use a fresh GENERESOLVER_STAGE5_RUN_DIR.")
    snap = cfg.BASE / "code_snapshot"
    if not snap.exists():
        for sub in ("generesolver", "evaluation"):
            shutil.copytree(ROOT / sub, snap / sub,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "runs", "data"))
    raw = cfg.BASE / "raw_with_vcf"
    if (cfg.BASE / _MANIFEST).exists():
        predates = bool(json.loads((cfg.BASE / _MANIFEST).read_text()).get("results_predate_manifest"))
    else:
        predates = raw.exists() and any(raw.iterdir())
    man = {
        "variant_filter_profile": cfg.FILTER_PROFILE,
        # True when this directory already held results before any manifest was written: the
        # values below describe the environment NOW, not necessarily when those results ran.
        "results_predate_manifest": predates,
        "rarity_af_ceiling": gconfig.RARITY_AF_CEILING,
        "spliceai_relevance_min": gconfig.SPLICEAI_RELEVANCE_MIN,
        "clinvar_triage": gconfig.CLINVAR_TRIAGE_ENABLED,
        "min_qual": float(os.environ.get("GENERESOLVER_MIN_QUAL", "20") or 0),
        "min_gq": gconfig.MIN_GENOTYPE_QUALITY, "min_dp": gconfig.MIN_READ_DEPTH,
        "vcf_max_variants": "uncapped",
        "llm_backend": os.environ.get("GENERESOLVER_LLM_BACKEND"),
        "vep": meta,
        "resources": resource_versions.collect("GRCh37"),
        "git_commit": commit, "git_dirty": dirty, "code": code_hash,
        "cache_seed_dir": cfg.CACHE_SEED_DIR,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (cfg.BASE / _MANIFEST).write_text(json.dumps(man, indent=2, default=str))
    print(f"  [manifest] {cfg.BASE / _MANIFEST} (profile={cfg.FILTER_PROFILE}, "
          f"clinvar_triage={gconfig.CLINVAR_TRIAGE_ENABLED})")


def main():
    import os
    os.environ.setdefault("GENERESOLVER_LLM_BACKEND", CONFIG.BACKEND)
    cfg = CONFIG
    cfg.BASE.mkdir(parents=True, exist_ok=True)
    _guard_run_dir(cfg)
    _seed_from(cfg)
    vcf_dir = cfg.BASE / "spiked_vcfs"
    raw_vcf = cfg.BASE / "raw_with_vcf"
    raw_ph = cfg.BASE / "raw_pheno_only"
    res_vcf = cfg.BASE / "results_with_vcf"
    res_ph = cfg.BASE / "results_pheno_only"
    for d in (raw_vcf, raw_ph):
        d.mkdir(parents=True, exist_ok=True)

    print("== 1. spike VCFs ==")
    _spike_all(cfg, vcf_dir)

    print("== 2. local VEP / cache / cap patches ==")
    # v2: v1 entries predate the gnomAD "absent" status used by PM2/BA1 and must not be reused.
    _cache_load(cfg.BASE / _VEP_CACHE_FILE)
    _gnomad_popmax_cache_load(cfg.BASE / "gnomad_popmax_cache.json.gz")
    _gwas_cache_load(cfg.BASE / "gwas_cache.json.gz")
    _install_patches()
    _local_vep_selftest()
    _write_manifest(cfg)

    if cfg.WARM_TEMPLATE:
        _warm_template_cache(cfg)
        _warm_causal_variants(cfg)
        _warm_gnomad_popmax_template(cfg)
        _warm_gnomad_popmax_causal(cfg)
    else:
        print("  [warm] skipped template-wide VEP warmup")

    print("== 2b. ensuring ground-truth placeholders exist (one-time, see _ensure_truth_placeholders) ==")
    res_vcf.mkdir(parents=True, exist_ok=True)
    res_ph.mkdir(parents=True, exist_ok=True)
    _ensure_truth_placeholders(cfg.LIRICAL_PP, res_vcf)
    if cfg.RUN_PHENO_ONLY_ARM:
        _ensure_truth_placeholders(cfg.LIRICAL_PP, res_ph)

    print("== 3. run GeneResolver (phenotype+VCF, and phenotype-only) ==")
    gid = GeneIdentifierUpdater(gene_identifier="ensembl_id", identifier_map=create_gene_identifier_map())
    cases = _load_cases(cfg, vcf_dir)
    total = len(cases)
    already = sum(1 for cs in cases if (raw_vcf / f"{cs.case_id}.json").exists()
                  and (not cfg.RUN_PHENO_ONLY_ARM or (raw_ph / f"{cs.case_id}.json").exists()))
    print(f"  {total} cases total — {already} already cached (resuming), "
          f"{total - already} left to run", file=sys.stderr)
    if total - already > 20:
        print(f"  [!] a full cold run of all {total} cases has taken up to ~16 hours "
              f"(local VEP/Docker annotation + LLM ranking per case, x2 arms) -- set "
              f"GENERESOLVER_VCF_LIMIT to a small number (e.g. 5) for a quick check instead.",
              file=sys.stderr)
    new_count = 0
    bar = tqdm(cases, desc="LIRICAL-385 (VCF + phenotype-only)", unit="case", file=sys.stderr)
    for cs in bar:
        rf = raw_vcf / f"{cs.case_id}.json"
        did_work = False
        if not rf.exists():
            try:
                ranked = _rank(cs.case_id, cs.hpos, str(cs.vcf))
            except Exception as exc:
                # No raw file on failure, so the resume check retries the case next run.
                tqdm.write(f"  [vcf] {cs.case_id}: ERROR {exc}", file=sys.stderr)
                continue
            rf.write_text(json.dumps(ranked))
            _cache_save()
            _gnomad_popmax_cache_save()
            _gwas_cache_save()
            did_work = True
        _emit_pheval(json.loads(rf.read_text()), cs.pp, res_vcf, gid)
        time.sleep(_EMIT_DELAY_S)

        if cfg.RUN_PHENO_ONLY_ARM:
            rp = raw_ph / f"{cs.case_id}.json"
            if not rp.exists():
                try:
                    ranked = _rank(cs.case_id, cs.hpos, None)
                except Exception as exc:
                    tqdm.write(f"  [ph] {cs.case_id}: ERROR {exc}", file=sys.stderr)
                    continue
                rp.write_text(json.dumps(ranked))
                did_work = True
            _emit_pheval(json.loads(rp.read_text()), cs.pp, res_ph, gid)
            time.sleep(_EMIT_DELAY_S)

        if did_work:
            new_count += 1
        bar.set_postfix(fresh=new_count)

    print("\n== done ==")
    print(f"phenotype+VCF results : {res_vcf}")
    print(f"phenotype-only results: {res_ph}")

    print("\n== 4. verifying parquet emission is complete (pheval-utils' own writer has a "
          "known race under back-to-back calls -- retry any case that silently didn't land) ==")
    _verify_and_retry_emission(raw_vcf, res_vcf, cfg, gid, "phenotype+VCF")
    if cfg.RUN_PHENO_ONLY_ARM:
        _verify_and_retry_emission(raw_ph, res_ph, cfg, gid, "phenotype-only")

    yaml_vcf = _write_run_yaml(cfg.BASE, "vcf", "generesolver_phenotype_vcf",
                                cfg.LIRICAL_PP, res_vcf)
    yaml_ph = None
    if cfg.RUN_PHENO_ONLY_ARM:
        yaml_ph = _write_run_yaml(cfg.BASE, "pheno_only", "generesolver_phenotype_only",
                                   cfg.LIRICAL_PP, res_ph)
    print(f"\nWrote ready-to-use pheval-utils config(s) -> {yaml_vcf}"
          + (f", {yaml_ph}" if yaml_ph else ""))

    metrics: dict = {}
    try:
        summary_vcf = pheval_scoring.run_benchmark(yaml_vcf, cfg.BASE / "benchmark_vcf")
        hits_vcf = pheval_scoring.per_case_hits_from_results(res_vcf)
        n_vcf = summary_vcf.get("total") or summary_vcf.get("number_of_samples") or len(hits_vcf)
        ci_vcf = stats.bootstrap_ci(pheval_scoring.hit_vector(hits_vcf, k=1))
        metrics["phenotype_vcf"] = {"summary": summary_vcf, "recall_at_1_ci95": [ci_vcf[1], ci_vcf[2]]}
        print(f"\nphenotype+VCF  — Recall@1={summary_vcf.get('top1', 0) / n_vcf:.3f} "
              f"(95% CI {ci_vcf[1]:.3f}-{ci_vcf[2]:.3f})  "
              f"Recall@5={summary_vcf.get('top5', 0) / n_vcf:.3f}  "
              f"MRR={summary_vcf.get('mrr', 0):.3f}")

        if cfg.RUN_PHENO_ONLY_ARM and yaml_ph is not None:
            summary_ph = pheval_scoring.run_benchmark(yaml_ph, cfg.BASE / "benchmark_pheno_only")
            hits_ph = pheval_scoring.per_case_hits_from_results(res_ph)
            n_ph = summary_ph.get("total") or summary_ph.get("number_of_samples") or len(hits_ph)
            ci_ph = stats.bootstrap_ci(pheval_scoring.hit_vector(hits_ph, k=1))
            metrics["phenotype_only"] = {"summary": summary_ph, "recall_at_1_ci95": [ci_ph[1], ci_ph[2]]}
            print(f"phenotype-only — Recall@1={summary_ph.get('top1', 0) / n_ph:.3f} "
                  f"(95% CI {ci_ph[1]:.3f}-{ci_ph[2]:.3f})  "
                  f"Recall@5={summary_ph.get('top5', 0) / n_ph:.3f}  "
                  f"MRR={summary_ph.get('mrr', 0):.3f}")

            common = sorted(set(hits_vcf) & set(hits_ph))
            a = pheval_scoring.hit_vector({c: hits_vcf[c] for c in common}, k=1)
            b = pheval_scoring.hit_vector({c: hits_ph[c] for c in common}, k=1)
            mc = stats.mcnemar(a, b)
            metrics["mcnemar_recall_at_1"] = {**mc, "n_paired_cases": len(common)}
            print(f"\nMcNemar (phenotype+VCF vs. phenotype-only, recall@1, n={len(common)} paired "
                  f"cases): genome corrects {mc['b']}, regresses {mc['c']}, p={mc['p_value']}")
    except Exception as exc:
        print(f"\n[score] could not auto-score with pheval-utils ({exc})")
        print("Score each arm yourself (NOT 'generate-benchmark-stats' — the subcommand is "
              "'benchmark'):")
        print(f"  pheval-utils benchmark -r {yaml_vcf} -o {cfg.BASE / 'benchmark_vcf'}")
        if yaml_ph is not None:
            print(f"  pheval-utils benchmark -r {yaml_ph} -o {cfg.BASE / 'benchmark_pheno_only'}")

    metrics_path = cfg.BASE / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {metrics_path}")


if __name__ == "__main__":
    main()
