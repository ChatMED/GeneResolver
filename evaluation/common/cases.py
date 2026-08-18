from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import idmap


@dataclass
class EvalCase:
    case_id: str
    hpo_present: list[str] = field(default_factory=list)
    hpo_absent: list[str] = field(default_factory=list)
    vcf_path: str | None = None
    sex: str | None = None
    gold_disease_ids: list[str] = field(default_factory=list)
    gold_gene: str | None = None
    gold_variant: dict | None = None



def load_rarebench(subset: str, limit: int | None = None) -> list[EvalCase]:
    from datasets import load_dataset

    ds = load_dataset("chenxz/RareBench", subset, split="test", trust_remote_code=True)
    cases: list[EvalCase] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        hpo = idmap.extract_ids(row.get("Phenotype"), kind="hpo")
        gold = idmap.extract_ids(row.get("RareDisease"), kind="disease")
        if not hpo or not gold:
            continue
        cases.append(EvalCase(
            case_id=f"{subset}_{i:05d}",
            hpo_present=hpo,
            gold_disease_ids=gold,
        ))
    return cases



def _get(d: dict, *names):
    for n in names:
        if isinstance(d, dict) and n in d and d[n] not in (None, "", [], {}):
            return d[n]
    return None


def load_phenopackets(directory: Path, limit: int | None = None) -> list[EvalCase]:
    directory = Path(directory)
    files = sorted(directory.rglob("*.json"))
    cases: list[EvalCase] = []
    for i, fp in enumerate(files):
        if limit is not None and len(cases) >= limit:
            break
        try:
            pp = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        case = _parse_phenopacket(pp, fallback_id=fp.stem)
        if case and case.hpo_present and case.gold_variant:
            cases.append(case)
    return cases


def _parse_phenopacket(pp: dict, fallback_id: str) -> EvalCase | None:
    case_id = _get(pp, "id") or fallback_id

    present, absent = [], []
    for f in (_get(pp, "phenotypicFeatures", "phenotypic_features") or []):
        t = _get(f, "type") or {}
        hid = idmap.normalize_hpo_id(_get(t, "id") or "")
        if not hid:
            continue
        if _get(f, "excluded") is True or _get(f, "negated") is True:
            absent.append(hid)
        else:
            present.append(hid)

    subject = _get(pp, "subject") or {}
    sex = (_get(subject, "sex") or "").lower() or None
    sex = {"male": "male", "female": "female"}.get(sex, sex)

    gold_disease, gold_gene, gold_variant = None, None, None
    for interp in (_get(pp, "interpretations") or []):
        diag = _get(interp, "diagnosis") or {}
        dis = _get(diag, "disease") or {}
        gold_disease = idmap.normalize_disease_id(_get(dis, "id") or "") or gold_disease
        for gi in (_get(diag, "genomicInterpretations", "genomic_interpretations") or []):
            vi = _get(gi, "variantInterpretation", "variant_interpretation") or {}
            vd = _get(vi, "variationDescriptor", "variation_descriptor") or {}
            gctx = _get(vd, "geneContext", "gene_context") or {}
            sym = _get(gctx, "symbol") or _get(_get(diag, "geneDescriptor", "gene_descriptor") or {}, "symbol")
            if sym:
                gold_gene = idmap.normalize_gene(sym)
            vcf = _get(vd, "vcfRecord", "vcf_record")
            if vcf:
                chrom = str(_get(vcf, "chrom", "chromosome") or "").replace("chr", "")
                pos = _get(vcf, "pos", "position")
                ref = _get(vcf, "ref")
                alt = _get(vcf, "alt")
                build = (_get(vcf, "genomeAssembly", "genome_assembly") or "").lower()
                if chrom and pos and ref and alt:
                    gold_variant = {
                        "chrom": chrom, "pos": int(pos), "ref": ref, "alt": alt,
                        "build": _norm_build(build),
                    }
            if gold_variant:
                break
        if gold_disease and gold_variant:
            break

    return EvalCase(
        case_id=case_id,
        hpo_present=present,
        hpo_absent=absent,
        sex=sex,
        gold_disease_ids=[gold_disease] if gold_disease else [],
        gold_gene=gold_gene,
        gold_variant=gold_variant,
    )


def _norm_build(b: str) -> str:
    b = (b or "").lower()
    if b in ("grch38", "hg38", "38"):
        return "hg38"
    if b in ("grch37", "hg19", "37"):
        return "hg19"
    return b or "hg38"
