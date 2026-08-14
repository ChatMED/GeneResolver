from __future__ import annotations

from generesolver.branches import spine
from generesolver.branches.descriptors import CHROMOSOMAL
from generesolver.schemas.case import CaseState

STEP = "branch_chromosomal"


def run(state: CaseState) -> CaseState:
    return spine.run_branch(state, CHROMOSOMAL, STEP)
