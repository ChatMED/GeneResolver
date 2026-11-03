"""ClinGen SVI BA1 exception list.

Source: Ghosh R, Harrison SM, Rehm HL, Plon SE, Biesecker LG; ClinGen Sequence Variant
Interpretation Working Group. "Updated recommendation for the benign stand-alone ACMG/AMP
criterion." Hum Mutat 2018;39(11):1525-1530. doi:10.1002/humu.23642 (Table 1).

These variants exceed the BA1 frequency threshold but must not be classified Benign by BA1
alone. Matching is build-independent: coding variants are matched on gene + protein change
(reference amino acid, position, alternate amino acid), and the ACAD9 5'UTR duplication on
gene + inserted sequence. An over-broad match only withholds BA1, which is the safe direction.
"""
from __future__ import annotations

import re

SOURCE = "ClinGen SVI BA1 exception list (Ghosh et al., Hum Mutat 2018, Table 1)"

# (gene, transcript, HGVS c., HGVS p. or None, GRCh37 position, alternative protein positions)
EXCEPTIONS: list[dict] = [
    {"gene": "ACAD9", "transcript": "NM_014049.4", "hgvs_c": "c.-44_-41dupTAAG", "hgvs_p": None,
     "grch37": "3:128598490", "dup_seq": "TAAG"},
    {"gene": "GJB2", "transcript": "NM_004004.5", "hgvs_c": "c.109G>A", "hgvs_p": "p.Val37Ile",
     "grch37": "13:20763612"},
    {"gene": "HFE", "transcript": "NM_000410.3", "hgvs_c": "c.187C>G", "hgvs_p": "p.His63Asp",
     "grch37": "6:26091179"},
    {"gene": "HFE", "transcript": "NM_000410.3", "hgvs_c": "c.845G>A", "hgvs_p": "p.Cys282Tyr",
     "grch37": "6:26093141"},
    {"gene": "MEFV", "transcript": "NM_000243.2", "hgvs_c": "c.1105C>T", "hgvs_p": "p.Pro369Ser",
     "grch37": "16:3299586"},
    {"gene": "MEFV", "transcript": "NM_000243.2", "hgvs_c": "c.1223G>A", "hgvs_p": "p.Arg408Gln",
     "grch37": "16:3299468"},
    {"gene": "PIBF1", "transcript": "NM_006346.2", "hgvs_c": "c.1214G>A", "hgvs_p": "p.Arg405Gln",
     "grch37": "13:73409497"},
    {"gene": "ACADS", "transcript": "NM_000017.3", "hgvs_c": "c.511C>T", "hgvs_p": "p.Arg171Trp",
     "grch37": "12:121175678"},
    # The paper's text also cites this variant as NM_001281724.2 c.1336G>C p.(Asp446His).
    {"gene": "BTD", "transcript": "NM_000060.4", "hgvs_c": "c.1330G>C", "hgvs_p": "p.Asp444His",
     "grch37": "3:15686693", "alt_positions": [446]},
]

_HGVS_P_RE = re.compile(r"p\.\(?([A-Za-z]{3})(\d+)([A-Za-z]{3})\)?")


def _protein_change(hgvs_p: str | None) -> tuple[str, int, str] | None:
    if not hgvs_p:
        return None
    m = _HGVS_P_RE.search(hgvs_p)
    if not m:
        return None
    return m.group(1).capitalize(), int(m.group(2)), m.group(3).capitalize()


def _inserted_sequence(variant_id: str) -> str | None:
    parts = variant_id.split(":")
    if len(parts) != 4:
        return None
    ref, alt = parts[2].upper(), parts[3].upper()
    if len(alt) <= len(ref) or not alt.startswith(ref):
        return None
    return alt[len(ref):]


def _is_rotation(seq: str, target: str) -> bool:
    return len(seq) == len(target) and seq in target + target


def match(gene: str | None, hgvs_p: str | None, variant_id: str) -> dict | None:
    """Return the matching exception entry, or None."""
    if not gene:
        return None
    gene = gene.upper()
    change = _protein_change(hgvs_p)
    inserted = _inserted_sequence(variant_id)
    for entry in EXCEPTIONS:
        if entry["gene"] != gene:
            continue
        if entry.get("dup_seq"):
            if inserted and _is_rotation(inserted, entry["dup_seq"]):
                return entry
            continue
        want = _protein_change(entry["hgvs_p"])
        if change and want and change[0] == want[0] and change[2] == want[2] \
                and change[1] in [want[1], *entry.get("alt_positions", [])]:
            return entry
    return None
