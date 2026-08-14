from __future__ import annotations

from dataclasses import dataclass, field

from generesolver.schemas.classification import CaseClass
from generesolver.schemas.trace import TestPower


@dataclass(frozen=True)
class BranchDescriptor:
    case_class: CaseClass | None
    label: str

    use_open_targets: bool = True
    use_literature: bool = True
    use_gwas: bool = False

    literature_terms: tuple[str, ...] = ()

    mechanism_focus: str = ""

    limitations: tuple[str, ...] = ()


CHROMOSOMAL = BranchDescriptor(
    case_class=CaseClass.CHROMOSOMAL,
    label="chromosomal",
    use_gwas=False,
    literature_terms=("chromosomal", "aneuploidy", "microdeletion", "contiguous gene syndrome"),
    mechanism_focus=(
        "Explain the finding as a chromosomal abnormality: aneuploidy or a contiguous-gene "
        "deletion/duplication. Identify which chromosomal region or syndrome the pattern of "
        "dysmorphism, developmental delay and multi-system STRUCTURAL anomalies fits, and say "
        "explicitly which genes in that region would account for which features. If the pattern "
        "is not regionally coherent, say so rather than naming a region."
    ),
    limitations=(
        "No CNV, cytoband or karyotype database was consulted — GeneResolver has none. This "
        "conclusion rests on the phenotype pattern and literature only, and cannot identify or "
        "size a specific deletion/duplication.",
    ),
)

MULTIFACTORIAL = BranchDescriptor(
    case_class=CaseClass.MULTIFACTORIAL,
    label="multifactorial",
    use_gwas=True,
    literature_terms=("GWAS", "polygenic", "heritability", "risk factor"),
    mechanism_focus=(
        "Explain the finding as a common multifactorial / complex trait: the joint effect of many "
        "common variants of small effect plus environmental exposure — NOT a single causal gene. "
        "Cite the GWAS-associated loci retrieved and what they implicate. Be explicit that no "
        "single variant is diagnostic here, and that a 'causal gene' is the wrong frame for this case."
    ),
    limitations=(
        "No polygenic risk score was computed — GeneResolver does not carry PRS weights. GWAS loci "
        "are reported as associations, not as a per-patient risk estimate.",
    ),
)

PHENOCOPY = BranchDescriptor(
    case_class=CaseClass.PHENOCOPY,
    label="phenocopy",
    use_gwas=False,
    literature_terms=("phenocopy", "teratogen", "congenital infection", "toxic", "nutritional deficiency"),
    mechanism_focus=(
        "Explain the finding as a PHENOCOPY — a genetic-looking presentation with a non-genetic "
        "cause: congenital infection, maternal teratogen/drug exposure, toxic insult, or severe "
        "nutritional deficiency. Name the specific acquired exposure or insult the pattern fits and "
        "the injury signature supporting it. State what would distinguish it from the genetic "
        "condition it mimics — that distinction is the entire clinical value of this branch."
    ),
    limitations=(
        "No toxicology, infectious-serology or exposure database was consulted. The acquired cause "
        "is inferred from the phenotype pattern and literature and MUST be confirmed serologically "
        "or by exposure history.",
    ),
)

UNCERTAIN = BranchDescriptor(
    case_class=None,
    label="uncertain",
    use_gwas=True,
    literature_terms=(),
    mechanism_focus=(
        "No etiology reached the confidence threshold. Lay out the competing hypotheses side by "
        "side with the evidence for and against EACH, and state plainly what is missing that would "
        "discriminate between them. Do NOT manufacture a single answer — the honest output here is "
        "a differential plus the test that would resolve it."
    ),
    limitations=(
        "No etiology class reached the confidence threshold; this case was not committed to a "
        "single mechanism, and was deliberately NOT routed to gene ranking.",
    ),
)


BY_CLASS: dict[CaseClass | None, BranchDescriptor] = {
    CaseClass.CHROMOSOMAL: CHROMOSOMAL,
    CaseClass.MULTIFACTORIAL: MULTIFACTORIAL,
    CaseClass.PHENOCOPY: PHENOCOPY,
    None: UNCERTAIN,
}
