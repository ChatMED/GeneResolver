from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_CORPUS_SUBSETS = ("MME", "HMS", "LIRICAL", "RAMEDIS", "PUMCH_ADM")

_HP_TOKEN = re.compile(r"\bHP:?\s*(\d{1,7})\b", re.IGNORECASE)
_DISEASE_TOKEN = re.compile(
    r"\b(OMIM|ORPHA|ORPHANET|MONDO|CCRD|DECIPHER|OMIMPS):\s*([A-Za-z0-9]+)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CorpusCase:
    case_id: str
    dataset: str
    hpo_ids: frozenset[int]
    gold_disease_ids: tuple[str, ...]


def _norm_disease(prefix: str, num: str) -> str:
    prefix = prefix.upper()
    if prefix in ("ORPHANET", "ORPHA"):
        prefix = "ORPHA"
    if prefix == "OMIMPS":
        prefix = "OMIM"
    return f"{prefix}:{num}"


def _extract_hpo_ints(value) -> set[int]:
    items = value if isinstance(value, (list, tuple)) else [value]
    text = " ".join(str(x) for x in items)
    return {int(n) for n in _HP_TOKEN.findall(text)}


def _extract_disease_ids(value) -> list[str]:
    items = value if isinstance(value, (list, tuple)) else [value]
    text = " ".join(str(x) for x in items)
    out, seen = [], set()
    for m in _DISEASE_TOKEN.finditer(text):
        did = _norm_disease(m.group(1), m.group(2))
        if did not in seen:
            seen.add(did)
            out.append(did)
    return out


def _load_subset(subset: str) -> list[CorpusCase]:
    from datasets import load_dataset

    try:
        ds = load_dataset("chenxz/RareBench", subset, split="test", trust_remote_code=True)
    except Exception as exc:
        logger.warning("case_corpus: subset %s unavailable (%s); skipping", subset, exc)
        return []
    out: list[CorpusCase] = []
    for i, row in enumerate(ds):
        hpo = _extract_hpo_ints(row.get("Phenotype"))
        gold = _extract_disease_ids(row.get("RareDisease"))
        if not hpo or not gold:
            continue
        out.append(CorpusCase(f"{subset}_{i:05d}", subset, frozenset(hpo), tuple(gold)))
    logger.info("case_corpus: loaded %d cases from %s", len(out), subset)
    return out


_corpus: list[CorpusCase] | None = None


def get_corpus(exclude_dataset: str | None = None) -> list[CorpusCase]:
    global _corpus
    if _corpus is None:
        cases: list[CorpusCase] = []
        for s in _CORPUS_SUBSETS:
            cases.extend(_load_subset(s))
        _corpus = cases
        logger.info("case_corpus: total %d reference cases across %d subsets",
                    len(cases), len(_CORPUS_SUBSETS))
    if exclude_dataset:
        ex = exclude_dataset.strip().upper()
        return [c for c in _corpus if c.dataset.upper() != ex]
    return list(_corpus)
