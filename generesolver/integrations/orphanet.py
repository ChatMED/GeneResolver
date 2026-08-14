from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from generesolver.config import DATA_DIR

logger = logging.getLogger(__name__)

ORPHANET_DIR: Path = DATA_DIR / "orphanet"
PRODUCT1_PATH: Path = ORPHANET_DIR / "en_product1.xml"
PRODUCT6_PATH: Path = ORPHANET_DIR / "en_product6.xml"

_ASSOCIATION_STRENGTH: dict[str, float] = {
    "Disease-causing germline mutation(s) in":          1.00,
    "Disease-causing germline mutation(s) (loss of function) in": 1.00,
    "Disease-causing germline mutation(s) (gain of function) in": 1.00,
    "Disease-causing somatic mutation(s) in":           0.85,
    "Major susceptibility factor in":                   0.65,
    "Modifying germline mutation in":                   0.50,
    "Part of a fusion gene in":                         0.55,
    "Role in the phenotype of":                         0.40,
    "Candidate gene tested in":                         0.25,
    "Biomarker tested in":                              0.20,
}

_VALIDATION_MULTIPLIER: dict[str, float] = {
    "Assessed":         1.00,
    "Not yet assessed": 0.70,
}

_MIN_SYNONYM_CHARS = 3


@dataclass(frozen=True)
class OrphanetDisease:
    orpha_id: str
    label: str
    synonyms: tuple[str, ...] = ()
    omim_ids: tuple[str, ...] = ()
    icd10_ids: tuple[str, ...] = ()
    icd11_ids: tuple[str, ...] = ()
    mesh_ids: tuple[str, ...] = ()
    mondo_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class OrphanetGeneAssociation:
    gene_symbol: str
    hgnc_id: str | None
    association_type: str
    validation_status: str

    @property
    def strength(self) -> float:
        return association_strength(self.association_type, self.validation_status)


_diseases: list[OrphanetDisease] | None = None
_name_index: dict[str, OrphanetDisease] | None = None
_orpha_index: dict[str, OrphanetDisease] | None = None
_disease_to_genes: dict[str, tuple[OrphanetGeneAssociation, ...]] | None = None
_gene_to_diseases: dict[str, tuple[tuple[str, OrphanetGeneAssociation], ...]] | None = None
_product6_disorder_names: dict[str, str] | None = None


def is_available() -> bool:
    return PRODUCT1_PATH.exists()


def is_gene_disease_available() -> bool:
    return PRODUCT6_PATH.exists()


def association_strength(association_type: str, validation_status: str) -> float:
    base = _ASSOCIATION_STRENGTH.get(association_type, 0.0)
    mult = _VALIDATION_MULTIPLIER.get(validation_status, 0.5)
    return base * mult


def _normalize_query(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _keep_synonym(syn: str) -> bool:
    if len(syn) >= _MIN_SYNONYM_CHARS:
        return True
    return syn.isalpha()


def _load() -> tuple[list[OrphanetDisease], dict[str, OrphanetDisease], dict[str, OrphanetDisease]]:
    if not PRODUCT1_PATH.exists():
        logger.warning(
            "Orphanet en_product1.xml not found at %s. Disease normalization "
            "will return raw mentions without ORPHA IDs. Run "
            "scripts/download_orphanet.py to fetch.",
            PRODUCT1_PATH,
        )
        return [], {}, {}

    diseases: list[OrphanetDisease] = []
    name_index: dict[str, OrphanetDisease] = {}
    orpha_index: dict[str, OrphanetDisease] = {}

    for _, elem in ET.iterparse(PRODUCT1_PATH, events=("end",)):
        if elem.tag != "Disorder":
            continue

        orpha_code = (elem.findtext("OrphaCode") or "").strip()
        name_elem = elem.find("Name")
        label = (name_elem.text or "").strip() if name_elem is not None else ""

        if not orpha_code or not label:
            elem.clear()
            continue

        synonyms: list[str] = []
        for syn in elem.findall("./SynonymList/Synonym"):
            if syn.text:
                stripped = syn.text.strip()
                if stripped:
                    synonyms.append(stripped)

        omim_ids: list[str] = []
        icd10_ids: list[str] = []
        icd11_ids: list[str] = []
        mesh_ids: list[str] = []
        mondo_ids: list[str] = []
        for ref in elem.findall("./ExternalReferenceList/ExternalReference"):
            source = (ref.findtext("Source") or "").strip()
            reference = (ref.findtext("Reference") or "").strip()
            if not source or not reference:
                continue
            if source == "OMIM":
                omim_ids.append(reference)
            elif source == "ICD-10":
                icd10_ids.append(reference)
            elif source == "ICD-11":
                icd11_ids.append(reference)
            elif source == "MeSH":
                mesh_ids.append(reference)
            elif source == "MONDO":
                mondo_ids.append(reference)

        disease = OrphanetDisease(
            orpha_id=f"ORPHA:{orpha_code}",
            label=label,
            synonyms=tuple(synonyms),
            omim_ids=tuple(omim_ids),
            icd10_ids=tuple(icd10_ids),
            icd11_ids=tuple(icd11_ids),
            mesh_ids=tuple(mesh_ids),
            mondo_ids=tuple(mondo_ids),
        )
        diseases.append(disease)
        orpha_index[disease.orpha_id] = disease

        key = _normalize_query(label)
        if key:
            name_index.setdefault(key, disease)
        for syn in synonyms:
            if not _keep_synonym(syn):
                continue
            syn_key = _normalize_query(syn)
            if syn_key:
                name_index.setdefault(syn_key, disease)

        elem.clear()

    logger.info(
        "Orphanet loaded: %d diseases, %d searchable names (canonical + synonyms) from %s",
        len(diseases), len(name_index), PRODUCT1_PATH.name,
    )
    return diseases, name_index, orpha_index


def _ensure_loaded() -> None:
    global _diseases, _name_index, _orpha_index
    if _diseases is None:
        _diseases, _name_index, _orpha_index = _load()


def normalize(text: str) -> OrphanetDisease | None:
    if not text:
        return None
    _ensure_loaded()
    if not _name_index:
        return None

    key = _normalize_query(text)
    if not key:
        return None

    hit = _name_index.get(key)
    if hit:
        return hit

    for suffix in (" syndrome", " disease", " disorder"):
        if key.endswith(suffix):
            stripped = key[: -len(suffix)].strip()
            if stripped and stripped in _name_index:
                return _name_index[stripped]
        else:
            extended = f"{key}{suffix}"
            if extended in _name_index:
                return _name_index[extended]

    return None


def get_all() -> list[OrphanetDisease]:
    _ensure_loaded()
    return list(_diseases or [])


def get_by_orpha_id(orpha_id: str) -> OrphanetDisease | None:
    _ensure_loaded()
    if not _orpha_index:
        return None
    return _orpha_index.get(orpha_id)


_omim_to_orpha_index: dict[str, str] | None = None


def get_orpha_by_omim(omim_id: str) -> OrphanetDisease | None:
    if not omim_id:
        return None
    _ensure_loaded()
    global _omim_to_orpha_index
    if _omim_to_orpha_index is None:
        idx: dict[str, str] = {}
        for d in _diseases or []:
            for omim in d.omim_ids:
                idx.setdefault(str(omim).strip(), d.orpha_id)
        _omim_to_orpha_index = idx
    key = omim_id.split(":", 1)[1] if ":" in omim_id else omim_id
    orpha = _omim_to_orpha_index.get(key.strip())
    if not orpha:
        return None
    return _orpha_index.get(orpha) if _orpha_index else None


def _load_product6() -> tuple[
    dict[str, tuple[OrphanetGeneAssociation, ...]],
    dict[str, tuple[tuple[str, OrphanetGeneAssociation], ...]],
    dict[str, str],
]:
    if not PRODUCT6_PATH.exists():
        logger.warning(
            "Orphanet en_product6.xml not found at %s. Disease→gene seeding "
            "and the score_disease_match term will be 0. Run "
            "scripts/download_orphanet.py to fetch.",
            PRODUCT6_PATH,
        )
        return {}, {}, {}

    disease_to_genes: dict[str, list[OrphanetGeneAssociation]] = {}
    gene_to_diseases: dict[str, list[tuple[str, OrphanetGeneAssociation]]] = {}
    disorder_names: dict[str, str] = {}

    for _, elem in ET.iterparse(PRODUCT6_PATH, events=("end",)):
        if elem.tag != "Disorder":
            continue

        orpha_code = (elem.findtext("OrphaCode") or "").strip()
        if not orpha_code:
            elem.clear()
            continue
        orpha_id = f"ORPHA:{orpha_code}"

        name_elem = elem.find("Name")
        if name_elem is not None and (name_elem.text or "").strip():
            disorder_names[orpha_id] = name_elem.text.strip()

        disease_genes: list[OrphanetGeneAssociation] = []
        for assoc in elem.findall("./DisorderGeneAssociationList/DisorderGeneAssociation"):
            gene_elem = assoc.find("Gene")
            if gene_elem is None:
                continue
            symbol_elem = gene_elem.find("Symbol")
            if symbol_elem is None or not (symbol_elem.text or "").strip():
                continue
            symbol = symbol_elem.text.strip()

            hgnc_id: str | None = None
            for ref in gene_elem.findall("./ExternalReferenceList/ExternalReference"):
                if (ref.findtext("Source") or "").strip() == "HGNC":
                    hgnc_id = (ref.findtext("Reference") or "").strip() or None
                    break

            assoc_type = (
                assoc.findtext("./DisorderGeneAssociationType/Name") or ""
            ).strip()
            assoc_status = (
                assoc.findtext("./DisorderGeneAssociationStatus/Name") or ""
            ).strip()

            ga = OrphanetGeneAssociation(
                gene_symbol=symbol,
                hgnc_id=hgnc_id,
                association_type=assoc_type,
                validation_status=assoc_status,
            )
            disease_genes.append(ga)
            gene_to_diseases.setdefault(symbol, []).append((orpha_id, ga))

        if disease_genes:
            disease_to_genes[orpha_id] = disease_genes
        elem.clear()

    n_pairs = sum(len(v) for v in disease_to_genes.values())
    logger.info(
        "Orphanet gene-disease loaded: %d diseases, %d unique genes, %d associations from %s",
        len(disease_to_genes), len(gene_to_diseases), n_pairs, PRODUCT6_PATH.name,
    )

    return (
        {k: tuple(v) for k, v in disease_to_genes.items()},
        {k: tuple(v) for k, v in gene_to_diseases.items()},
        disorder_names,
    )


def _ensure_product6_loaded() -> None:
    global _disease_to_genes, _gene_to_diseases, _product6_disorder_names
    if _disease_to_genes is None:
        (_disease_to_genes,
         _gene_to_diseases,
         _product6_disorder_names) = _load_product6()

_SUBTYPE_MIN_LABEL_LEN = 5


def get_genes_for_disease(
    orpha_id: str,
    *,
    include_subtypes: bool = True,
) -> tuple[OrphanetGeneAssociation, ...]:
    _ensure_product6_loaded()
    if not _disease_to_genes:
        return ()

    direct = _disease_to_genes.get(orpha_id, ())
    if direct or not include_subtypes:
        return direct

    canonical = get_by_orpha_id(orpha_id)
    if canonical is None or len(canonical.label) < _SUBTYPE_MIN_LABEL_LEN:
        return ()

    label_lower = canonical.label.lower()
    matched: list[OrphanetGeneAssociation] = []
    seen_pairs: set[tuple[str, str]] = set()

    if _product6_disorder_names:
        for sub_orpha_id, sub_name in _product6_disorder_names.items():
            if sub_orpha_id == orpha_id:
                continue
            if label_lower not in sub_name.lower():
                continue
            for ga in _disease_to_genes.get(sub_orpha_id, ()):
                key = (ga.gene_symbol, ga.association_type)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                matched.append(ga)

    return tuple(matched)


def get_diseases_for_gene(gene_symbol: str) -> tuple[tuple[str, OrphanetGeneAssociation], ...]:
    _ensure_product6_loaded()
    if not _gene_to_diseases:
        return ()
    return _gene_to_diseases.get(gene_symbol, ())
