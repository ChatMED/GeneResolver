from __future__ import annotations

from generesolver.branches import spine
from generesolver.branches.descriptors import MULTIFACTORIAL
from generesolver.schemas.case import CaseState

STEP = "branch_multifactorial"


def run(state: CaseState) -> CaseState:
    return spine.run_branch(state, MULTIFACTORIAL, STEP)
