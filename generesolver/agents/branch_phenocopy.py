from __future__ import annotations

from generesolver.branches import spine
from generesolver.branches.descriptors import PHENOCOPY
from generesolver.schemas.case import CaseState

STEP = "branch_phenocopy"


def run(state: CaseState) -> CaseState:
    return spine.run_branch(state, PHENOCOPY, STEP)
