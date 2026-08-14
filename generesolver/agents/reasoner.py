from __future__ import annotations

import logging

from generesolver.schemas.case import CaseState
from generesolver.reasoner import hpo_differential

logger = logging.getLogger(__name__)


def run(state: CaseState) -> CaseState:
    present = list(state.hpo_profile.present_ids or [])
    absent = list(state.hpo_profile.absent_ids or [])
    if present:
        try:
            ranked = hpo_differential.rank_differential(present, absent or None)
            if ranked:
                state.ranked_diseases = ranked
                logger.info("reasoner agent: set %d ranked diseases (authoritative)",
                            len(ranked))
            else:
                logger.info("reasoner agent: no grounded diseases — keeping prior ranking")
            diag = getattr(state, "diagnostics", None)
            if isinstance(diag, dict):
                diag["reasoner_scoring"] = list(hpo_differential._LAST_SCORING)
                diag["reasoner_engines"] = dict(hpo_differential._LAST_ENGINES)
        except Exception as exc:
            logger.warning("reasoner agent failed, keeping prior ranking: %s", exc)
    else:
        logger.info("reasoner agent: no present HPO terms — skipped")
    state.mark_done("reasoner")
    return state
