from __future__ import annotations

import json
import logging
from pathlib import Path

from generesolver import orchestrator
from generesolver.config import OUTPUTS_DIR
from generesolver.schemas.case import CaseInput, CaseState

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)


def run_case(inp: CaseInput, save_output: bool = True) -> CaseState:
    state = CaseState(case_id=inp.case_id, input=inp)

    state = orchestrator.run(state)
    logger.info("Task mode: %s", state.task_mode)

    if save_output:
        _save(state)

    return state


def _save(state: CaseState) -> None:
    out_dir = OUTPUTS_DIR / state.case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    case_state_path = out_dir / "case_state.json"
    case_state_path.write_text(
        state.model_dump_json(indent=2),
        encoding="utf-8",
    )
    logger.info("Case state saved to %s", case_state_path)
