from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GSC = ROOT / "data" / "gsc_plus"
HP_OBO = ROOT / "data" / "hp.obo"
PHENO_ROOT = "HP:0000118"

BOOTSTRAP_N = 2000
SEED = 42


@dataclass
class Result:
    n: int = 0
    mode: str = "fair"
    metrics: dict = field(default_factory=dict)


def _load_obo() -> tuple[dict[str, str], set[str]]:
    alt2prim: dict[str, str] = {}
    children: dict[str, set[str]] = defaultdict(set)
    cur = None
    parents: list[str] = []
    alts: list[str] = []
    obsolete = False
    repl = None

    def flush():
        if cur:
            for a in alts:
                alt2prim[a] = cur
            if obsolete and repl:
                alt2prim[cur] = repl
            for p in parents:
                children[p].add(cur)

    for line in HP_OBO.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line == "[Term]":
            flush(); cur, parents, alts, obsolete, repl = None, [], [], False, None
        elif line.startswith("id: HP:"):
            cur = line[4:].strip()
        elif line.startswith("alt_id: HP:"):
            alts.append(line[8:].strip())
        elif line.startswith("is_a: HP:"):
            parents.append(line[6:].split("!")[0].strip())
        elif line.startswith("is_obsolete: true"):
            obsolete = True
        elif line.startswith("replaced_by: HP:"):
            repl = line[13:].strip()
    flush()
    pheno, stack = set(), [PHENO_ROOT]
    while stack:
        t = stack.pop()
        if t in pheno:
            continue
        pheno.add(t)
        stack.extend(children.get(t, ()))
    return alt2prim, pheno


def _bootstrap_micro_f1_ci(counts: list[tuple[int, int, int]], n_boot: int, seed: int) -> list[float]:
    if not counts:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(counts)
    vals = []
    for _ in range(n_boot):
        tp = fp = fn = 0
        for _ in range(n):
            t, f, g = counts[rng.randrange(n)]
            tp += t; fp += f; fn += g
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        vals.append(2 * p * r / (p + r) if p + r else 0.0)
    vals.sort()
    return [round(vals[int(0.025 * n_boot)], 4), round(vals[int(0.975 * n_boot)], 4)]


def _extractors():
    ex = {}
    try:
        from generesolver.integrations.fasthpocr_extractor import extract_hpo_terms, is_available
        if is_available():
            ex["FastHPOCR"] = lambda t: {a["hpo_id"] for a in extract_hpo_terms(t) if a.get("hpo_id")}
    except Exception as e:
        print(f"  [skip] FastHPOCR: {e}", file=sys.stderr)
    try:
        import os
        from clinphen_src import get_phenotypes as cp
        names_file = os.path.join(os.path.dirname(cp.__file__), "data", "hpo_term_names.txt")
        hpo_to_name = {}
        for line in open(names_file, encoding="utf-8", errors="ignore"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                hpo_to_name[parts[0]] = parts[1]

        def clinphen_extract(t):
            out = cp.extract_phenotypes(t, hpo_to_name)
            return {("HP:" + m.group(1)) for m in re.finditer(r"HP[_:](\d{7})", str(out))}
        clinphen_extract("the patient had seizures")
        ex["ClinPhen"] = clinphen_extract
    except Exception as e:
        print(f"  [skip] ClinPhen: {e}", file=sys.stderr)
    return ex


def run(limit: int = 0, mode: str = "fair") -> Result:
    if not GSC.exists():
        sys.exit(f"GSC+ corpus not found at {GSC}")
    alt2prim, pheno = _load_obo()

    def norm(ids: set[str]) -> set[str]:
        out = {alt2prim.get(i, i) for i in ids}
        return {i for i in out if i in pheno} if mode == "fair" else out

    ann_dir, txt_dir = GSC / "Annotations", GSC / "Text"
    docs = sorted(p for p in ann_dir.iterdir() if p.is_file())
    if limit:
        docs = docs[:limit]

    def text_for(docid: str):
        for c in (txt_dir / docid, txt_dir / f"{docid}.txt"):
            if c.exists():
                return c.read_text(encoding="utf-8", errors="ignore")
        hits = list(txt_dir.glob(f"{docid}*"))
        return hits[0].read_text(encoding="utf-8", errors="ignore") if hits else None

    gold = {}
    for ann in docs:
        t = text_for(ann.name)
        if t is not None:
            ids = {("HP:" + m.group(1)) for m in
                   re.finditer(r"HP[_:](\d{7})", ann.read_text(encoding="utf-8", errors="ignore"))}
            gold[ann.name] = (t, norm(ids))

    extractors = _extractors()
    if not extractors:
        sys.exit("no HPO tagger available (FastHPOCR/ClinPhen both failed to import)")

    metrics: dict = {}
    items = list(gold.items())
    for tool, fn in extractors.items():
        TP = FP = FN = 0
        per_doc_counts: list[tuple[int, int, int]] = []
        for _docid, (text, g) in items:
            try:
                pred = norm(fn(text))
            except Exception:
                pred = set()
            tp, fp, fn_ = len(g & pred), len(pred - g), len(g - pred)
            TP += tp; FP += fp; FN += fn_
            per_doc_counts.append((tp, fp, fn_))
        p = TP / (TP + FP) if TP + FP else 0.0
        r = TP / (TP + FN) if TP + FN else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        ci = _bootstrap_micro_f1_ci(per_doc_counts, BOOTSTRAP_N, SEED)
        metrics[tool] = {"precision": round(p, 4), "recall": round(r, 4),
                          "F1": round(f1, 4), "F1_ci95": ci}
        print(f"  {tool:12} P={p:.3f}  R={r:.3f}  F1={f1:.3f}  ci95={ci}  (n={len(items)})")

    return Result(n=len(gold), mode=mode, metrics=metrics)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 228 GSC+ docs")
    ap.add_argument("--mode", choices=["fair", "raw"], default="fair",
                     help="fair = normalize alt-ids + restrict to phenotypic-abnormality terms")
    ap.add_argument("--out", default=None, help="write JSON report here")
    args = ap.parse_args()

    print(f"Phenotype extraction eval — GSC+ ({args.mode})\n")
    res = run(limit=args.limit, mode=args.mode)

    out = Path(args.out) if args.out else (
        ROOT / "eval_runs" / f"eval_phenotype_extraction_{datetime.now():%Y%m%d_%H%M%S}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(res), indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
