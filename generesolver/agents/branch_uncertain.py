from __future__ import annotations

from generesolver.branches import spine
from generesolver.branches.descriptors import UNCERTAIN
from generesolver.schemas.case import CaseState

STEP = "branch_uncertain"


def run(state: CaseState) -> CaseState:
    return spine.run_branch(state, UNCERTAIN, STEP)
