import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
HPO_DIR = DATA_DIR / "hpo"
OUTPUTS_DIR = Path(os.environ.get("GENERESOLVER_OUTPUTS_DIR", ROOT / "outputs"))

CASE_BRANCHING_ENABLED = os.environ.get(
    "GENERESOLVER_CASE_BRANCHING", "1"
).strip().lower() in ("1", "true", "yes")

DEID_BACKEND = os.environ.get("GENERESOLVER_DEID_BACKEND", "openmed").strip().lower()

VEP_REST_URL = "https://rest.ensembl.org/vep/human/hgvs"
GNOMAD_GRAPHQL_URL = "https://gnomad.broadinstitute.org/api"
OPEN_TARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
CLINVAR_EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
UNIPROT_REST_URL = "https://rest.uniprot.org/uniprotkb"
REACTOME_REST_URL = "https://reactome.org/ContentService"
EUROPEPMC_REST_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"

LOCAL_MODE = os.environ.get("GENERESOLVER_LOCAL_MODE", "0").strip().lower() in ("1", "true", "yes")

PUBCASEFINDER_ENABLED = (
    os.environ.get("GENERESOLVER_PUBCASEFINDER", "0").strip().lower() in ("1", "true", "yes")
    and not LOCAL_MODE
)

# A ClinVar P/LP assertion lifts a variant's triage score to 0.95 (above the 0.75 ceiling).
# Set GENERESOLVER_CLINVAR_TRIAGE=0 for a ClinVar-blind run, e.g. on benchmarks of published variants.
CLINVAR_TRIAGE_ENABLED = os.environ.get(
    "GENERESOLVER_CLINVAR_TRIAGE", "1").strip().lower() in ("1", "true", "yes")

# Variant filter profile.
#   "original" (default): the filter used for every reported result. Rarity = overall gnomAD
#       v4.1 exome AF (VEP gnomADe_AF) < RARITY_AF_CEILING, or no AF; relevance = VEP IMPACT
#       HIGH/MODERATE or a consequence in the hard-coded list (which includes the LOW-impact
#       splice_region_variant).
#   "revised": same 1% ceiling, but measured on ONE source -- the highest gnomAD v4.1 exome AF
#       among the continental groups AFR/AMR/EAS/NFE/SAS (the same groups ACMG BA1 uses);
#       never VEP MAX_AF (mixes 1000 Genomes) or the slivar gnomAD v2.1.1 file. Frequency
#       "not_evaluated" is kept but counted and flagged separately from "absent".
#       splice_region_variant counts as relevant only with SpliceAI >= SPLICEAI_RELEVANCE_MIN.
# Revised results must be reported separately; they are not the published configuration.
VARIANT_FILTER_PROFILE = os.environ.get(
    "GENERESOLVER_VARIANT_FILTER_PROFILE", "original").strip().lower()
if VARIANT_FILTER_PROFILE not in ("original", "revised"):
    raise ValueError(f"GENERESOLVER_VARIANT_FILTER_PROFILE must be 'original' or 'revised', "
                     f"got {VARIANT_FILTER_PROFILE!r}")
# The 1% ceiling is the same in both profiles. Inheritance-specific ceilings (e.g. 0.1% for
# dominant models) are NOT implemented: they change candidate selection and need validation first.
RARITY_AF_CEILING = 0.01
SPLICEAI_RELEVANCE_MIN = 0.2

# Per-genotype read-quality filters (0 = off, the reported configuration). A missing GQ/DP
# passes. Keep off on simulated benchmarks: PhEval spiked causal records carry GT only, so
# filtering would remove only background calls and inflate performance.
MIN_GENOTYPE_QUALITY = float(os.environ.get("GENERESOLVER_MIN_GQ", "0") or 0)
MIN_READ_DEPTH = int(os.environ.get("GENERESOLVER_MIN_DP", "0") or 0)

GENE_CANDIDATE_CAP = int(os.environ.get("GENERESOLVER_GENE_CANDIDATE_CAP", "30") or 0)
VEP_LOCAL_DOCKER_IMAGE = os.environ.get(
    "GENERESOLVER_VEP_DOCKER_IMAGE", "ensemblorg/ensembl-vep:release_116.0"
)
VEP_LOCAL_CACHE_DIR = Path(os.environ.get("GENERESOLVER_VEP_CACHE_DIR") or DATA_DIR / "vep_cache")

SPECIFICITY_TIER_ENABLED = True

ALLOW_SYNTHESIZER_ON_EMPTY_HPO = False

STAGE_C_LLM_RANKING = True
SIMILAR_CASE_RERANK = False
SIMILAR_CASE_RERANK_K = 12
SIMILAR_CASE_RERANK_LAMBDA = 1.0
REASONING_LEAD_CONCLUSION = os.environ.get(
    "GENERESOLVER_REASONING_LEAD", "0").strip().lower() in ("1", "true", "yes")
REASONING_LEAD_DECISIVE = os.environ.get(
    "GENERESOLVER_REASONING_LEAD_DECISIVE", "0").strip().lower() in ("1", "true", "yes")
REASONING_LEAD_TOP_K = 10
STAGE_B_FALLBACK_GROUNDING = False
CONCLUDE_ANNOTATION_CAVEAT = False
CLUSTER_PLAYOFF_ENABLED = False
CLUSTER_PLAYOFF_MAGNITUDE_CAVEAT = False
LLM_NAMES_ONLY = os.environ.get(
    "GENERESOLVER_LLM_NAMES_ONLY", "0").strip().lower() in ("1", "true", "yes")

LLM_RANKED_ONLY = os.environ.get(
    "GENERESOLVER_LLM_RANKED_ONLY", "0").strip().lower() in ("1", "true", "yes")

DX_PIPELINE = os.environ.get(
    "GENERESOLVER_DX_PIPELINE", "0").strip().lower() in ("1", "true", "yes")

DX_AGENTIC = os.environ.get(
    "GENERESOLVER_DX_AGENTIC", "0").strip().lower() in ("1", "true", "yes")

SIMILAR_CASE_ENABLED = False
SIMILAR_CASE_TOP_CASES = 25
SIMILAR_CASE_TOP_DISEASES = 15
SIMILAR_CASE_MIN_SIM = 0.05
SIMILAR_CASE_DEDUP_SIM = 0.98

SIMILAR_CASE_MODE = os.environ.get("GENERESOLVER_SIMILAR_CASE_MODE", "cohort").strip().lower()
SIMILAR_CASE_EXCLUDE_DATASET = os.environ.get("GENERESOLVER_SIMILAR_CASE_EXCLUDE") or None

NER_BATCH_SIZE = 8
NER_AGGREGATION_STRATEGY = "average"
NER_MIN_CONFIDENCE = 0.70

HPO_CANDIDATE_TOP_K = 20
HPO_MIN_CONFIDENCE = 0.70

RANK_WEIGHT_PHENOTYPE = 0.30
RANK_WEIGHT_VARIANT = 0.26
RANK_WEIGHT_GENE_DISEASE = 0.17
RANK_WEIGHT_LITERATURE = 0.07
RANK_WEIGHT_GWAS = 0.03
RANK_WEIGHT_CONSTRAINT = 0.08
RANK_WEIGHT_PATHWAY = 0.09
PATHWAY_ANCHOR_TOP_K = 20

LR_RANKER_ENABLED = True

GENE_SCORING_POOL_K = 500
GENE_OUTPUT_TOP_K = 30

DISEASE_OUTPUT_TOP_K = 50

PHENOTYPE_DISEASE_SEED_TOP_K = 20
PHENOTYPE_DISEASE_SEED_MIN_OVERLAP = 3
PHENOTYPE_DISEASE_SEED_MAX_GENES = 15
PHENOTYPE_DISEASE_SEED_MAX_TOTAL_GENES = 30

OT_RECALL_SEED_MAX_GENES = 20
OT_RECALL_SEED_MIN_SCORE = 0.30
OT_RECALL_SEED_FETCH_SIZE = 50
OT_RECALL_SEED_BREADTH_SCORE = 0.50
OT_RECALL_SEED_MAX_DISEASE_BREADTH = 15

SHORTLIST_RECALL_SEED_TOP_K = 5
SHORTLIST_RECALL_SEED_MAX_GENES = 20

# False = tie-break mode: the LLM judge may nudge a gene by at most ±JUDGE_TIEBREAK_CAP,
# so it can only reorder genes whose deterministic scores differ by < 2 × cap. The
# delivered ranking then stays the evaluated deterministic ranking up to near-ties.
JUDGE_AUTHORITATIVE_RANK = False
JUDGE_TIEBREAK_CAP = 0.01
JUDGE_RANK_WEIGHT = 0.70
JUDGE_TOP_K = 30
JUDGE_MAX_CANDIDATES = 30
JUDGE_SEED_RESERVE = 10
JUDGE_AUTHORITATIVE_DISEASE_ORDER = False

DISEASE_MATCH_BOOST_MAX = 0.10
