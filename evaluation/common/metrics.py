from __future__ import annotations

import math

from . import idmap



def _rank_of_disease(pred_ids: list[str], gold_set: set[str]) -> int | None:
    for i, pid in enumerate(pred_ids):
        if pid in gold_set:
            return i + 1
    return None


def _rank_of_disease_sets(pred_sets: list[list[str]], gold_set: set[str]) -> int | None:
    for i, ids in enumerate(pred_sets):
        if gold_set & set(ids):
            return i + 1
    return None


def _rank_of_gene(pred_genes: list[str], gold_gene: str | None,
                  alias_map: dict | None) -> int | None:
    if not gold_gene:
        return None
    target = idmap.normalize_gene(gold_gene, alias_map)
    for i, g in enumerate(pred_genes):
        if idmap.normalize_gene(g, alias_map) == target:
            return i + 1
    return None


def _rank_of_variant(pred_variants: list[dict], gold_vid: str | None) -> int | None:
    if not gold_vid:
        return None
    g = _norm_vid(gold_vid)
    for i, v in enumerate(pred_variants):
        if _norm_vid(v.get("variant_id")) == g:
            return i + 1
    return None


def _norm_vid(vid: str | None) -> str | None:
    if not vid:
        return None
    parts = str(vid).replace("chr", "").split(":")
    if len(parts) != 4:
        return str(vid).lower().replace("chr", "")
    chrom, pos, ref, alt = parts
    return f"{chrom}:{pos}:{ref.upper()}:{alt.upper()}"



def case_ranks(records: list[dict], cross: dict, alias_map: dict) -> dict[str, dict]:
    out = {}
    for r in records:
        if r.get("error"):
            out[r["case_id"]] = {"disease": None, "gene": None, "variant": None, "errored": True}
            continue
        gold_dis = idmap.expand_gold(r.get("gold_disease_ids") or [], cross)
        out[r["case_id"]] = {
            "disease": _rank_of_disease(r.get("pred_disease_ids") or [], gold_dis),
            "gene": _rank_of_gene(r.get("pred_genes") or [], r.get("gold_gene"), alias_map),
            "variant": _rank_of_variant(r.get("pred_variants") or [], r.get("gold_variant_id")),
            "errored": False,
        }
    return out



def recall_at_k(ranks: list[int | None], k: int) -> float:
    scored = [r for r in ranks]
    if not scored:
        return float("nan")
    hits = sum(1 for r in scored if r is not None and r <= k)
    return hits / len(scored)


def mrr(ranks: list[int | None]) -> float:
    scored = [r for r in ranks]
    if not scored:
        return float("nan")
    return sum((1.0 / r) if r else 0.0 for r in scored) / len(scored)


def hit_vector(ranks: list[int | None], k: int) -> list[int]:
    return [1 if (r is not None and r <= k) else 0 for r in ranks]


def summarize_axis(case_rank_map: dict[str, dict], which: str, k_values) -> dict:
    ranks = [v[which] for v in case_rank_map.values()
             if not v.get("errored")]
    ranks = [r for cid, r in zip(case_rank_map.keys(), ranks)]
    out = {"n": len(ranks), "mrr": round(mrr(ranks), 4)}
    for k in k_values:
        out[f"recall@{k}"] = round(recall_at_k(ranks, k), 4)
    return out



def calibration(records: list[dict], case_rank_map: dict[str, dict],
                which: str = "disease", n_bins: int = 10) -> dict:
    from .config import CONFIDENCE_BAND_MIDPOINT
    pairs = []
    for r in records:
        if r.get("error"):
            continue
        rank = case_rank_map.get(r["case_id"], {}).get(which)
        correct = 1 if (rank == 1) else 0
        prob = r.get("top_disease_posttest")
        if prob is None:
            band = (r.get("confidence_band") or r.get("confidence_level") or "unknown")
            prob = CONFIDENCE_BAND_MIDPOINT.get(str(band).lower(), 0.05)
        pairs.append((float(prob), correct))
    if not pairs:
        return {"ece": float("nan"), "n": 0, "bins": []}

    bins = [[] for _ in range(n_bins)]
    for p, c in pairs:
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, c))
    ece, rows = 0.0, []
    for b in bins:
        if not b:
            rows.append(None)
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(c for _, c in b) / len(b)
        ece += (len(b) / len(pairs)) * abs(acc - conf)
        rows.append({"conf": round(conf, 3), "acc": round(acc, 3), "n": len(b)})
    return {"ece": round(ece, 4), "n": len(pairs), "bins": rows}



def safety_flagging(records: list[dict], should_flag_fn) -> dict:
    tp = fp = tn = fn = 0
    for r in records:
        if r.get("error"):
            continue
        flagged = bool(r.get("requires_human_review"))
        should = bool(should_flag_fn(r))
        if should and flagged:
            tp += 1
        elif should and not flagged:
            fn += 1
        elif not should and flagged:
            fp += 1
        else:
            tn += 1
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {"sensitivity": round(sens, 4), "specificity": round(spec, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}



_ACMG_CLASSES = ["pathogenic", "likely_pathogenic",
                 "variant_of_uncertain_significance", "likely_benign", "benign"]


def acmg_concordance(records: list[dict], gold_acmg_by_variant: dict[str, str]) -> dict:
    mat = {a: {b: 0 for b in _ACMG_CLASSES} for a in _ACMG_CLASSES}
    n = correct = 0
    for r in records:
        for v in (r.get("pred_variants") or []):
            vid = _norm_vid(v.get("variant_id"))
            gold = gold_acmg_by_variant.get(vid)
            pred = (v.get("acmg") or "").lower()
            if gold is None or pred not in _ACMG_CLASSES or gold not in _ACMG_CLASSES:
                continue
            mat[gold][pred] += 1
            n += 1
            correct += int(gold == pred)
    return {"n": n, "accuracy": round(correct / n, 4) if n else float("nan"),
            "matrix": mat}
