"""Table 8: held-out ACMG/AMP validation, GeneResolver's engine and GeneBe on the same ClinVar variants.

Steps (run in order; each step's output is kept and never overwritten):

  sample   ClinVar >=2-star P, LP, LB and B coding/splice variants, selected with the filters of
           acmg_validation_balanced.py, restricted to variants already annotated by a Table 10 run
           (same VEP 116 / GRCh37 configuration, checked by fingerprint, so no VEP cache is needed)
           and excluding every variant of the 400-variant development set. The cached annotations
           are copied into the pipeline's VEP result cache (a backup is made first).
  score    run the current GeneResolver engine on the sampled variants
           (acmg_validation_balanced --rescore).
  genebe   query the GeneBe public API (https://genebe.net) for the same variants. They are public
           ClinVar records, not patient data. Resumable; the ACMG fields of each response are kept.
  report   both tools' classifications as published (all criteria, including those that draw on
           ClinVar), against the ClinVar class: overall, per consequence, agreement between tools.

    python -m evaluation.table8_acmg_validation.eval_acmg_validation sample --stage5-run eval_runs/<stage5 run>
    python -m evaluation.table8_acmg_validation.eval_acmg_validation score
    python -m evaluation.table8_acmg_validation.eval_acmg_validation genebe
    python -m evaluation.table8_acmg_validation.eval_acmg_validation report

Dataset in data_creation/acmg_validation/: heldout_214.json (the variant list), dev_400.json (the
development set, excluded) and genebe_responses.jsonl. Outputs in evaluation/paper_results/acmg_heldout/:
comparison.json and comparison.md; GeneResolver's per-variant result is
evaluation/paper_results/acmg_validation/acmg_validation_balanced_GRCh37_<tag>.json.

Both tools can use ClinVar for the variant being tested (GeneResolver PS1/PM5/PM1, GeneBe PP5 and
BP6), as they would in use; the report gives each tool's classification as returned. The variants
occur in the Stage 5 exome, so benign variants are mostly common background polymorphisms and
pathogenic ones mostly the spiked causal variants; the per-consequence table shows the mix.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from math import comb, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT = ROOT / "evaluation" / "paper_results" / "acmg_heldout"
DATA = ROOT / "data_creation" / "acmg_validation"
RECORDS = DATA / "heldout_214.json"
GENEBE = DATA / "genebe_responses.jsonl"
DEV_SET = DATA / "dev_400.json"
RESULT_DIR = ROOT / "evaluation" / "paper_results" / "acmg_validation"
GENEBE_URL = "https://api.genebe.net/cloud/api-public/v1/variants?genome=hg19"
GENEBE_FIELDS = ("acmg_classification", "acmg_criteria", "acmg_score", "gene_symbol", "transcript", "effect")
GOLD_CLASSES = ("pathogenic", "likely_pathogenic", "likely_benign", "benign")
CLASSES = ("pathogenic", "likely_pathogenic", "vus", "likely_benign", "benign", "not_classified")


# --------------------------------------------------------------------------------------- sample
def sample(stage5_run: Path, exclude: list[Path], per_class: int, seed: int) -> None:
    from evaluation.table8_acmg_validation import acmg_validation_balanced as v
    from generesolver.components.variant_disease_context_builder import _gene_disease_pairs
    from generesolver.components.vcf_input import _trim_variant
    from generesolver.integrations import vep_api

    if RECORDS.exists():
        sys.exit(f"{RECORDS} exists (the held-out set is never resampled)")
    fp_run = json.loads((stage5_run / "vep_cache_local_v2.json.gz.fingerprint.json")
                        .read_text(encoding="utf-8")).get("fingerprint")
    fp_now = vep_api.annotation_metadata("GRCh37")["fingerprint"]
    if fp_run != fp_now:
        sys.exit(f"VEP fingerprint differs: Stage 5 run {fp_run}, current configuration {fp_now}")
    with gzip.open(stage5_run / "vep_cache_local_v2.json.gz", "rt", encoding="utf-8") as f:
        stage5 = {k: a for k, a in json.load(f).items() if a}
    print(f"Stage 5 VEP cache: {len(stage5):,} annotated variants (fingerprint {fp_run})")

    excluded = {r["variant_id"] for p in exclude for r in json.loads(p.read_text(encoding="utf-8"))["per_variant"]}
    pools: dict[str, list[dict]] = {g: [] for g in v.GOLD.values()}
    with gzip.open(ROOT / "data" / "clinvar" / "clinvar_GRCh37.vcf.gz", "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            chrom, pos, vid, ref, alt, _, _, info = line.rstrip("\n").split("\t")[:8]
            if len(ref) > 50 or len(alt) > 50 or alt == "." or f"{chrom}:{pos}:{ref}:{alt}" in excluded:
                continue
            d = v._info(info)
            gold = v.GOLD.get(d.get("CLNSIG", ""))
            if not gold or d.get("CLNREVSTAT") not in v.TWO_STAR_PLUS:
                continue
            genes = [g.split(":")[0] for g in d.get("GENEINFO", "").split("|") if g]
            mcs = {m.split("|")[-1] for m in d.get("MC", "").split(",") if "|" in m}
            if len(genes) != 1 or not mcs & v.CODING_MC:
                continue
            tpos, tref, talt, _ = _trim_variant(int(pos), ref, alt)
            key = f"{chrom}:{tpos}:{tref}:{talt}"
            if key in stage5:
                pools[gold].append({
                    "variant_id": f"{chrom}:{pos}:{ref}:{alt}", "chrom": chrom, "pos": int(pos), "ref": ref,
                    "alt": alt, "gene": genes[0],
                    "diseases": [x for x in d.get("CLNDN", "").split("|") if x and x != "not_provided"],
                    "gold": gold, "revstat": d["CLNREVSTAT"], "clinvar_id": vid,
                    "clinvar_mc": sorted(mcs & v.CODING_MC), "_key": key})
    rng = random.Random(seed)
    records = []
    for gold, pool in pools.items():
        rng.shuffle(pool)
        kept = [r for r in pool if _gene_disease_pairs(r["gene"])][:per_class]
        print(f"  {gold:18s} in the Stage 5 cache={len(pool):>6}  sampled={len(kept)}")
        records.extend(kept)

    cache_path = vep_api._RESULT_CACHE_PATH
    backup = cache_path.with_name(cache_path.name + ".before_heldout")
    if cache_path.exists() and not backup.exists():
        shutil.copy2(cache_path, backup)
    cache = vep_api._load_result_cache() if cache_path.exists() else {}
    added = 0
    for r in records:
        key = r.pop("_key")
        if f"{fp_run}:GRCh37:{key}" not in cache:
            cache[f"{fp_run}:GRCh37:{key}"] = stage5[key]
            added += 1
    tmp = cache_path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(cache, f)
    tmp.replace(cache_path)
    print(f"VEP result cache: {added} annotation(s) copied from the Stage 5 run")

    DATA.mkdir(parents=True, exist_ok=True)
    RECORDS.write_text(json.dumps({"per_variant": records, "source": {
        "stage5_run": stage5_run.as_posix(), "vep_fingerprint": fp_run, "seed": seed,
        "excluded_from": [p.name for p in exclude], "counts": dict(Counter(r["gold"] for r in records))}},
        indent=2), encoding="utf-8")
    print(f"wrote {RECORDS} ({len(records)} variants)")


# ---------------------------------------------------------------------------------------- score
def score(tag: str) -> None:
    subprocess.run([sys.executable, "-m", "evaluation.table8_acmg_validation.acmg_validation_balanced", "--rescore", str(RECORDS),
                    "--tag", tag], check=True, cwd=ROOT)


# --------------------------------------------------------------------------------------- genebe
def _genebe_request(batch: list[dict]) -> list[dict]:
    body = json.dumps([{"chr": v["chrom"], "pos": v["pos"], "ref": v["ref"], "alt": v["alt"]} for v in batch])
    req = urllib.request.Request(GENEBE_URL, data=body.encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read().decode()).get("variants") or []
    if len(out) != len(batch):
        raise RuntimeError(f"GeneBe returned {len(out)} records for {len(batch)} variants")
    return out


def genebe(batch_size: int = 50, pause: float = 2.0) -> None:
    variants = json.loads(RECORDS.read_text(encoding="utf-8"))["per_variant"]
    done = ({json.loads(l)["variant_id"] for l in GENEBE.read_text(encoding="utf-8").splitlines() if l.strip()}
            if GENEBE.exists() else set())
    todo = [v for v in variants if v["variant_id"] not in done]
    print(f"GeneBe: {len(done)} saved, {len(todo)} to query")
    with GENEBE.open("a", encoding="utf-8") as fh:
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            for attempt in range(5):
                try:
                    recs = _genebe_request(batch)
                    break
                except (urllib.error.URLError, RuntimeError, TimeoutError) as exc:
                    print(f"  batch {i // batch_size}: {exc}; retry in {30 * (attempt + 1)}s")
                    time.sleep(30 * (attempt + 1))
            else:
                sys.exit("GeneBe unreachable after 5 attempts; rerun to resume")
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for v, rec in zip(batch, recs):
                fh.write(json.dumps({"variant_id": v["variant_id"], "queried_at": stamp,
                                     "response": {k: rec.get(k) for k in GENEBE_FIELDS}}) + "\n")
            fh.flush()
            print(f"  {min(i + batch_size, len(todo))}/{len(todo)}")
            time.sleep(pause)


# --------------------------------------------------------------------------------------- report
def _norm(label: str | None) -> str:
    s = (label or "").strip().lower().replace(" ", "_").replace("-", "_")
    if s in ("uncertain_significance", "uncertain", "vus"):
        return "vus"
    return s if s in CLASSES else "not_classified"


def _dir(c: str) -> str:
    return "P/LP" if c in ("pathogenic", "likely_pathogenic") else ("B/LB" if c in ("benign", "likely_benign") else "VUS")


def _wilson(k: int, n: int) -> list[float] | None:
    if not n:
        return None
    p, z = k / n, 1.96
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 3), round(centre + half, 3)]


def _accuracy(rows: list[tuple[str, str]]) -> dict:
    pg = [c for g, c in rows if _dir(g) == "P/LP"]
    bg = [c for g, c in rows if _dir(g) == "B/LB"]
    tp, tb = sum(_dir(c) == "P/LP" for c in pg), sum(_dir(c) == "B/LB" for c in bg)
    return {
        "n": len(rows), "n_P_LP": len(pg), "n_B_LB": len(bg),
        "P_LP_called_P_LP": round(tp / len(pg), 3) if pg else None, "P_LP_ci95": _wilson(tp, len(pg)),
        "B_LB_called_B_LB": round(tb / len(bg), 3) if bg else None, "B_LB_ci95": _wilson(tb, len(bg)),
        "overall_direction_correct": round((tp + tb) / len(rows), 3), "overall_ci95": _wilson(tp + tb, len(rows)),
        "opposite_direction": sum(_dir(c) == "B/LB" for c in pg) + sum(_dir(c) == "P/LP" for c in bg),
        "vus_or_not_classified": sum(_dir(c) == "VUS" for _, c in rows),
        "exact_class_agreement": round(sum(g == c for g, c in rows) / len(rows), 3),
        "confusion": {g: {c: sum(1 for gg, cc in rows if gg == g and cc == c) for c in CLASSES} for g in GOLD_CLASSES},
    }


def _mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else round(min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n), 4)


def _kappa(pairs: list[tuple[str, str]]) -> float | None:
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def report(tag: str, out: Path = OUT) -> dict:
    ours = {r["variant_id"]: r for r in json.loads(
        (RESULT_DIR / f"acmg_validation_balanced_GRCh37_{tag}.json").read_text(encoding="utf-8"))["per_variant"]}
    theirs = {}
    for line in GENEBE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            theirs[row["variant_id"]] = row["response"] or {}
    variants = [v for v in json.loads(RECORDS.read_text(encoding="utf-8"))["per_variant"]
                if v["variant_id"] in ours and v["variant_id"] in theirs]
    gr = {v["variant_id"]: ours[v["variant_id"]]["computed"] if ours[v["variant_id"]]["computed"] in CLASSES[:5]
          else "not_classified" for v in variants}
    gb = {v["variant_id"]: _norm(theirs[v["variant_id"]].get("acmg_classification")) for v in variants}
    gold = {v["variant_id"]: v["gold"] for v in variants}
    ids = list(gold)
    gr_ok = {i: _dir(gr[i]) == _dir(gold[i]) for i in ids}
    gb_ok = {i: _dir(gb[i]) == _dir(gold[i]) for i in ids}
    by_cons: dict[str, list[str]] = {}
    for i in ids:
        by_cons.setdefault(ours[i].get("consequence") or "not_annotated", []).append(i)
    summary = {
        "n": len(ids), "gold_counts": dict(Counter(gold.values())),
        "genebe_queried": sorted({json.loads(l).get("queried_at", "")[:10] for l in
                                   GENEBE.read_text(encoding="utf-8").splitlines() if l.strip()}),
        "generesolver": _accuracy([(gold[i], gr[i]) for i in ids]),
        "genebe": _accuracy([(gold[i], gb[i]) for i in ids]),
        "agreement_direction": round(sum(_dir(gr[i]) == _dir(gb[i]) for i in ids) / len(ids), 3),
        "kappa_direction": _kappa([(_dir(gr[i]), _dir(gb[i])) for i in ids]),
        "mcnemar_correct_direction": {"only_generesolver": sum(gr_ok[i] and not gb_ok[i] for i in ids),
                                      "only_genebe": sum(gb_ok[i] and not gr_ok[i] for i in ids)},
        "by_consequence": {c: {"n": len(v), "generesolver_correct": sum(gr_ok[i] for i in v),
                               "genebe_correct": sum(gb_ok[i] for i in v)}
                           for c, v in sorted(by_cons.items(), key=lambda x: -len(x[1]))},
        "generesolver_result": f"acmg_validation_balanced_GRCh37_{tag}.json",
    }
    m = summary["mcnemar_correct_direction"]
    m["p"] = _mcnemar(m["only_generesolver"], m["only_genebe"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def pct(x, ci):
        return f"{100 * x:.1f}% ({100 * ci[0]:.1f}–{100 * ci[1]:.1f})" if x is not None else "–"

    a, b = summary["generesolver"], summary["genebe"]
    md = [f"# Held-out ACMG/AMP validation: GeneResolver vs GeneBe ({summary['n']} ClinVar variants, GRCh37)", "",
          f"ClinVar classes: {summary['gold_counts']}. Classifications as returned by each tool (all criteria).", "",
          "| | GeneResolver | GeneBe |", "|---|---|---|",
          f"| P/LP called P/LP (95% CI) | {pct(a['P_LP_called_P_LP'], a['P_LP_ci95'])} | {pct(b['P_LP_called_P_LP'], b['P_LP_ci95'])} |",
          f"| B/LB called B/LB (95% CI) | {pct(a['B_LB_called_B_LB'], a['B_LB_ci95'])} | {pct(b['B_LB_called_B_LB'], b['B_LB_ci95'])} |",
          f"| Overall direction correct (95% CI) | {pct(a['overall_direction_correct'], a['overall_ci95'])} | {pct(b['overall_direction_correct'], b['overall_ci95'])} |",
          f"| Opposite-direction calls | {a['opposite_direction']} | {b['opposite_direction']} |",
          f"| VUS / not classified | {a['vus_or_not_classified']} | {b['vus_or_not_classified']} |",
          f"| Exact 5-class agreement | {100 * a['exact_class_agreement']:.1f}% | {100 * b['exact_class_agreement']:.1f}% |", "",
          f"Direction agreement between the tools {100 * summary['agreement_direction']:.1f}% "
          f"(κ {summary['kappa_direction']}); McNemar on correct direction: only GeneResolver "
          f"{m['only_generesolver']}, only GeneBe {m['only_genebe']}, p = {m['p']}.", "",
          "| Consequence | n | GeneResolver correct | GeneBe correct |", "|---|---|---|---|"]
    md += [f"| {c} | {r['n']} | {r['generesolver_correct']} | {r['genebe_correct']} |"
           for c, r in summary["by_consequence"].items()]
    (out / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--stage5-run", type=Path, required=True)
    s.add_argument("--exclude", type=Path, nargs="+", default=[DEV_SET])
    s.add_argument("--per-class", type=int, default=100)
    s.add_argument("--seed", type=int, default=20261102)
    for name in ("score", "report"):
        sub.add_parser(name).add_argument("--tag", default="v3.1_heldout")
    sub.add_parser("genebe")
    args = ap.parse_args()
    if args.cmd == "sample":
        sample(args.stage5_run, args.exclude, args.per_class, args.seed)
    elif args.cmd == "score":
        score(args.tag)
    elif args.cmd == "genebe":
        genebe()
    else:
        report(args.tag)


if __name__ == "__main__":
    main()
