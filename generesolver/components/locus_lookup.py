from __future__ import annotations

import re

from generesolver.integrations import clinvar_api
from generesolver.schemas.case import CaseState
from generesolver.schemas.report import ProvenanceItem


_RSID = re.compile(r"rs\d+", re.IGNORECASE)


def run(state: CaseState) -> CaseState:
    rsid = (state.input.rsid or "").strip()
    if not _RSID.fullmatch(rsid):
        raise ValueError(f"Invalid dbSNP rsID: {rsid!r}")
    records = clinvar_api.lookup_all_by_rsid(rsid)
    state.branch_result["locus_lookup"] = {
        "rsid": rsid,
        "clinvar_records": records,
        "source": "NCBI ClinVar E-utilities",
    }
    if not records:
        state.add_warning(f"No ClinVar records were retrieved for {rsid}; this does not establish absence from ClinVar.")
    state.provenance.append(ProvenanceItem(component="locus_lookup", tool_name="NCBI ClinVar E-utilities"))
    state.mark_done("locus_lookup")
    return state
