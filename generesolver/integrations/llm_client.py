"""Single entry point for every LLM call in GeneResolver.

GeneResolver runs one model: MedGemma 27B IT served locally behind an OpenAI-compatible
endpoint (``GENERESOLVER_MEDGEMMA_BASE_URL``). There is no cloud or alternative backend.
``GENERESOLVER_LLM_BACKEND`` is accepted only as ``medgemma`` so that an old configuration
pointing elsewhere fails loudly instead of silently changing the model.
"""
from __future__ import annotations

import os

from generesolver.integrations import medgemma_client
from generesolver.integrations._llm_errors import LLMResponseError

__all__ = ["complete_json", "complete_structured", "complete_text",
           "LLMResponseError", "active_backend"]

_BACKEND = "medgemma"
_BACKEND_ENV = "GENERESOLVER_LLM_BACKEND"


def _check(backend: str | None = None) -> None:
    for value in (os.environ.get(_BACKEND_ENV), backend):
        if value and value.strip().lower() != _BACKEND:
            raise ValueError(f"Unsupported LLM backend {value!r}: GeneResolver only runs the local "
                             f"MedGemma backend. Unset {_BACKEND_ENV} or set it to 'medgemma'.")


def active_backend() -> str:
    _check()
    return _BACKEND


def complete_json(system: str, user: str, *, max_tokens: int, model: str | None = None,
                  guided_json: dict | None = None, temperature: float = 0.0,
                  backend: str | None = None) -> dict:
    _check(backend)
    return medgemma_client.complete_json(system, user, max_tokens=max_tokens, model=model,
                                         guided_json=guided_json, temperature=temperature)


def complete_structured(system: str, user: str, *, schema_model: type, max_tokens: int,
                        model: str | None = None, backend: str | None = None):
    _check(backend)
    return medgemma_client.complete_structured(system, user, schema_model=schema_model,
                                               max_tokens=max_tokens, model=model)


def complete_text(system: str, user: str, *, max_tokens: int, model: str | None = None,
                  temperature: float = 0.0, backend: str | None = None) -> str:
    _check(backend)
    return medgemma_client.complete_text(system, user, max_tokens=max_tokens, model=model,
                                         temperature=temperature)
