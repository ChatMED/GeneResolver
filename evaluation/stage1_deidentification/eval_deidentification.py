from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data_creation" / "deid" / "generated_deid_dataset.jsonl"

MODELS = [
    "StanfordAIMI/stanford-deidentifier-base",
    "openmed",
    "obi/deid_roberta_i2b2",
]

BOOTSTRAP_N = 2000
SEED = 42


@dataclass
class Result:
    n: int = 0
    metrics: dict = field(default_factory=dict)


def _bootstrap_ratio_ci(pairs: list[tuple[float, float]], n_boot: int, seed: int) -> list[float]:
    if not pairs:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(pairs)
    vals = []
    for _ in range(n_boot):
        a = b = 0.0
        for _ in range(n):
            x, y = pairs[rng.randrange(n)]
            a += x; b += y
        vals.append(a / b if b else 0.0)
    vals.sort()
    return [round(vals[int(0.025 * n_boot)], 4), round(vals[int(0.975 * n_boot)], 4)]


def _ner_mask_fn(model_id: str):
    from transformers import pipeline
    pipe = pipeline("token-classification", model=model_id, aggregation_strategy="simple")

    def mask(note: str) -> str:
        masked = note
        for e in sorted(pipe(note), key=lambda x: -int(x["start"])):
            masked = masked[:int(e["start"])] + "[REDACTED]" + masked[int(e["end"]):]
        return masked
    return mask


def _load_models(model_ids: list[str]) -> dict:
    models = {}
    for mid in model_ids:
        if mid == "openmed":
            try:
                from generesolver.integrations.privacy_filter import deidentify_text
                models["OpenMed-privacy-filter"] = lambda n: deidentify_text(n)[0]
            except Exception as e:
                print(f"  [skip] OpenMed: {e}", file=sys.stderr)
        else:
            try:
                models[mid.split("/")[-1]] = _ner_mask_fn(mid)
            except Exception as e:
                print(f"  [skip] {mid}: {e}", file=sys.stderr)
    return models


def run(limit: int = 0, models: list[str] | None = None, dataset: Path = DATA_PATH) -> Result:
    if not dataset.exists():
        sys.exit(f"de-id set not found at {dataset}")
    recs = [json.loads(l) for l in dataset.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        recs = recs[:limit]

    loaded = _load_models(models or MODELS)
    if not loaded:
        sys.exit("no de-id model could be loaded (check MODELS / downloads)")

    metrics: dict = {}
    for name, mask_fn in loaded.items():
        pairs, leaks = [], []
        total_tokens = total_removed = 0
        for r in recs:
            try:
                masked = mask_fn(r["note"])
            except Exception:
                masked = r["note"]
            toks = r.get("expected_pii_tokens", [])
            low = masked.lower()
            removed = sum(1 for t in toks if t.lower() not in low)
            total_tokens += len(toks)
            total_removed += removed
            pairs.append((removed, len(toks)))
            leaks += [t for t in toks if t.lower() in low]
        recall = total_removed / total_tokens if total_tokens else 0.0
        ci = _bootstrap_ratio_ci(pairs, BOOTSTRAP_N, SEED)
        metrics[name] = {
            "recall": round(recall, 4), "recall_ci95": ci,
            "leaks": f"{len(leaks)}/{total_tokens}",
        }
        print(f"  {name:32} recall={recall:.3f}  ci95={ci}  leaks={len(leaks)}/{total_tokens}")

    return Result(n=len(recs), metrics=metrics)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="0 = all cases")
    ap.add_argument("--models", nargs="*", default=None, help="override the model list")
    ap.add_argument("--out", default=None, help="write JSON report here")
    ap.add_argument("--dataset", default=None, help="path to a de-id test set jsonl (default: data_creation/deid/generated_deid_dataset.jsonl)")
    args = ap.parse_args()

    dataset = Path(args.dataset) if args.dataset else DATA_PATH
    print(f"De-identification eval — {dataset.name}\n")
    res = run(limit=args.limit, models=args.models, dataset=dataset)

    out = Path(args.out) if args.out else (
        ROOT / "eval_runs" / f"eval_deidentification_{datetime.now():%Y%m%d_%H%M%S}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(res), indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
