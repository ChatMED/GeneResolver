from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

LABELS = ("single_gene", "chromosomal", "multifactorial", "non_genetic")
CONDITIONS = ((True, False), (True, True), (False, False), (False, True))


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    raise ValueError(f"Not a boolean: {value!r}")


def load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    required = {"case_id", "gold_label", "few_shot", "rag", "prediction"}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Missing fields {sorted(missing)} in row {row}")
        row["few_shot"] = as_bool(row["few_shot"])
        row["rag"] = as_bool(row["rag"])
        row["repeat"] = int(row.get("repeat", 0))
        if row["gold_label"] not in LABELS or row["prediction"] not in LABELS:
            raise ValueError(f"Invalid label in case {row['case_id']}")
    return rows


def macro_f1(gold: list[str], pred: list[str]) -> float:
    scores = []
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        scores.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores)


def confusion_matrix(gold: list[str], pred: list[str]) -> dict[str, dict[str, int]]:
    """Four-way counts with gold labels as rows and predictions as columns."""
    matrix = {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}
    for actual, predicted in zip(gold, pred):
        matrix[actual][predicted] += 1
    return matrix


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def bootstrap_ci(values: list[float], rng: random.Random, b: int) -> tuple[float, float]:
    n = len(values)
    estimates = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(b)
    ]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    lower = min(b, c)
    tail = sum(math.comb(n, k) for k in range(lower + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def condition_name(condition: tuple[bool, bool]) -> str:
    few, rag = condition
    return f"{'few' if few else 'no-few'}+{'RAG' if rag else 'raw'}"


def analyze_repeat(rows: list[dict], repeat: int, bootstraps: int, seed: int) -> dict:
    subset = [r for r in rows if r["repeat"] == repeat]
    lookup: dict[tuple[bool, bool], dict[str, dict]] = defaultdict(dict)
    for row in subset:
        condition = (row["few_shot"], row["rag"])
        cid = row["case_id"]
        if cid in lookup[condition]:
            raise ValueError(f"Duplicate case {cid}, condition {condition}, repeat {repeat}")
        lookup[condition][cid] = row

    id_sets = [set(lookup[c]) for c in CONDITIONS]
    if not id_sets or any(ids != id_sets[0] for ids in id_sets[1:]):
        sizes = {condition_name(c): len(lookup[c]) for c in CONDITIONS}
        raise ValueError(f"Repeat {repeat} is not complete and paired: {sizes}")
    case_ids = sorted(id_sets[0])
    rng = random.Random(seed + repeat)

    result: dict = {"repeat": repeat, "n": len(case_ids), "conditions": {}, "comparisons": {}}
    correctness: dict[tuple[bool, bool], list[int]] = {}
    branch_correct: dict[tuple[bool, bool], list[int]] = {}
    for condition in CONDITIONS:
        records = [lookup[condition][cid] for cid in case_ids]
        gold = [r["gold_label"] for r in records]
        pred = [r["prediction"] for r in records]
        correct = [int(g == p) for g, p in zip(gold, pred)]
        branch = [int((g == "single_gene") == (p == "single_gene")) for g, p in zip(gold, pred)]
        correctness[condition] = correct
        branch_correct[condition] = branch
        acc_ci = bootstrap_ci(correct, rng, bootstraps)
        branch_ci = bootstrap_ci(branch, rng, bootstraps)
        confusion = confusion_matrix(gold, pred)
        single_gene_total = sum(confusion["single_gene"].values())
        single_gene_correct = confusion["single_gene"]["single_gene"]
        result["conditions"][condition_name(condition)] = {
            "correct": sum(correct),
            "accuracy": sum(correct) / len(correct),
            "accuracy_ci": acc_ci,
            "macro_f1": macro_f1(gold, pred),
            "branch_correct": sum(branch),
            "branch_accuracy": sum(branch) / len(branch),
            "branch_ci": branch_ci,
            "single_gene_correct": single_gene_correct,
            "single_gene_total": single_gene_total,
            "single_gene_recall": (
                single_gene_correct / single_gene_total if single_gene_total else None
            ),
            "confusion_matrix": confusion,
        }

    comparisons = {
        "few-shot effect, raw": ((True, False), (False, False)),
        "few-shot effect, RAG": ((True, True), (False, True)),
        "context effect, few-shot": ((True, True), (True, False)),
        "context effect, no-few-shot": ((False, True), (False, False)),
    }
    for name, (left, right) in comparisons.items():
        left_values = correctness[left]
        right_values = correctness[right]
        differences = [a - b for a, b in zip(left_values, right_values)]
        discordant_left = sum(a == 1 and b == 0 for a, b in zip(left_values, right_values))
        discordant_right = sum(a == 0 and b == 1 for a, b in zip(left_values, right_values))
        branch_left = branch_correct[left]
        branch_right = branch_correct[right]
        branch_b = sum(a == 1 and b == 0 for a, b in zip(branch_left, branch_right))
        branch_c = sum(a == 0 and b == 1 for a, b in zip(branch_left, branch_right))
        result["comparisons"][name] = {
            "left": condition_name(left),
            "right": condition_name(right),
            "accuracy_difference": sum(differences) / len(differences),
            "difference_ci": bootstrap_ci(differences, rng, bootstraps),
            "discordant_left_better": discordant_left,
            "discordant_right_better": discordant_right,
            "mcnemar_exact_p": exact_mcnemar_p(discordant_left, discordant_right),
            "branch_discordant_left_better": branch_b,
            "branch_discordant_right_better": branch_c,
            "branch_mcnemar_exact_p": exact_mcnemar_p(branch_b, branch_c),
        }
    return result


def print_report(result: dict) -> None:
    print(f"\nREPEAT {result['repeat']}  n={result['n']}")
    for name, metric in result["conditions"].items():
        lo, hi = metric["accuracy_ci"]
        blo, bhi = metric["branch_ci"]
        print(
            f"{name:12s}  accuracy={metric['correct']}/{result['n']}="
            f"{metric['accuracy']:.3f} (95% CI {lo:.3f}-{hi:.3f})  "
            f"macro-F1={metric['macro_f1']:.3f}  "
            f"branch={metric['branch_correct']}/{result['n']}="
            f"{metric['branch_accuracy']:.3f} (95% CI {blo:.3f}-{bhi:.3f})"
        )
        recall = metric["single_gene_recall"]
        recall_text = f"{recall:.3f}" if recall is not None else "undefined"
        print(
            f"  single-gene recall={metric['single_gene_correct']}/"
            f"{metric['single_gene_total']}={recall_text}"
        )
        print("  confusion matrix (gold rows, predicted columns):")
        print("    gold \\ pred       " + " ".join(f"{label:>14s}" for label in LABELS))
        for gold_label, counts in metric["confusion_matrix"].items():
            print(f"    {gold_label:18s}" + " ".join(
                f"{counts[pred_label]:14d}" for pred_label in LABELS
            ))
    for name, comparison in result["comparisons"].items():
        lo, hi = comparison["difference_ci"]
        print(
            f"{name}: Δ={comparison['accuracy_difference']:+.3f} "
            f"(95% CI {lo:+.3f} to {hi:+.3f}); "
            f"discordant={comparison['discordant_left_better']}/"
            f"{comparison['discordant_right_better']}; "
            f"McNemar p={comparison['mcnemar_exact_p']:.5f}; "
            f"branch discordant={comparison['branch_discordant_left_better']}/"
            f"{comparison['branch_discordant_right_better']}; "
            f"branch p={comparison['branch_mcnemar_exact_p']:.5f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise SystemExit("--bootstrap must be at least 100")

    rows = load_rows(args.predictions)
    repeats = sorted({row["repeat"] for row in rows})
    results = [
        analyze_repeat(rows, repeat, args.bootstrap, args.seed) for repeat in repeats
    ]
    for result in results:
        print_report(result)
    if args.json_output:
        args.json_output.write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
