from __future__ import annotations

import math
import random


def bootstrap_ci(values: list[float], statistic=lambda v: sum(v) / len(v),
                 n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    values = [v for v in values if v is not None]
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    point = statistic(values)
    stats = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(sample))
    stats.sort()
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[int((1 - alpha / 2) * n_boot) - 1]
    return (round(point, 4), round(lo, 4), round(hi, 4))


def mcnemar(hits_a: list[int], hits_b: list[int]) -> dict:
    assert len(hits_a) == len(hits_b), "vectors must be aligned by case"
    b = sum(1 for a, x in zip(hits_a, hits_b) if a == 1 and x == 0)
    c = sum(1 for a, x in zip(hits_a, hits_b) if a == 0 and x == 1)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0, "stat": 0.0}
    if n < 25:
        k = min(b, c)
        p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
        p = min(1.0, p)
        return {"b": b, "c": c, "p_value": round(p, 5), "stat": None, "test": "exact"}
    stat = (abs(b - c) - 1) ** 2 / n
    p = _chi2_sf_1df(stat)
    return {"b": b, "c": c, "stat": round(stat, 4), "p_value": round(p, 5), "test": "chi2_cc"}


def wilcoxon_rr(ranks_a: list[int | None], ranks_b: list[int | None]) -> dict:
    rr_a = [(1.0 / r if r else 0.0) for r in ranks_a]
    rr_b = [(1.0 / r if r else 0.0) for r in ranks_b]
    diffs = [a - b for a, b in zip(rr_a, rr_b) if (a - b) != 0]
    if not diffs:
        return {"p_value": 1.0, "n_nonzero": 0}
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(rr_a, rr_b, zero_method="wilcox")
        return {"p_value": round(float(p), 5), "stat": round(float(stat), 4),
                "n_nonzero": len(diffs), "test": "scipy"}
    except Exception:
        absd = sorted((abs(d), i) for i, d in enumerate(diffs))
        ranks = [0.0] * len(diffs)
        for r, (_, i) in enumerate(absd, 1):
            ranks[i] = r
        w_plus = sum(rk for rk, d in zip(ranks, diffs) if d > 0)
        n = len(diffs)
        mean = n * (n + 1) / 4
        sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24) or 1.0
        z = (w_plus - mean) / sd
        p = 2 * (1 - _norm_cdf(abs(z)))
        return {"p_value": round(min(1.0, p), 5), "z": round(z, 4),
                "n_nonzero": n, "test": "normal_approx"}



def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _chi2_sf_1df(x: float) -> float:
    if x <= 0:
        return 1.0
    return 2 * (1 - _norm_cdf(math.sqrt(x)))
