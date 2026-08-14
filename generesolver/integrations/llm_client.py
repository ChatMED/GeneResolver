from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_BACKEND_ENV = "GENERESOLVER_LLM_BACKEND"
_VALID = {"claude", "medgemma", "deepseek", "openai"}


def _selected_backend() -> str:
    raw = os.environ.get(_BACKEND_ENV, "medgemma").strip().lower()
    if raw not in _VALID:
        logger.warning(
            "%s = %r is not a recognised backend (valid: %s) — falling back to 'medgemma'.",
            _BACKEND_ENV, raw, sorted(_VALID),
        )
        return "medgemma"
    return raw

from generesolver.integrations._llm_errors import LLMResponseError

__all__ = ["complete_json", "complete_structured", "complete_text",
           "LLMResponseError", "active_backend"]


def active_backend() -> str:
    return _selected_backend()


def _resolve_backend(backend: str | None) -> str:
    if backend is None:
        return _selected_backend()
    normalized = backend.strip().lower()
    if normalized not in _VALID:
        logger.warning(
            "backend override %r is not recognised (valid: %s) — falling back to %s.",
            backend, sorted(_VALID), _BACKEND_ENV,
        )
        return _selected_backend()
    return normalized


def complete_json(system: str, user: str, *, max_tokens: int, model: str | None = None,
                  guided_json: dict | None = None, temperature: float = 0.0,
                  backend: str | None = None) -> dict:
    backend = _resolve_backend(backend)
    if backend == "medgemma":
        from generesolver.integrations import medgemma_client
        return medgemma_client.complete_json(system, user, max_tokens=max_tokens, model=model,
                                              guided_json=guided_json, temperature=temperature)
    if backend == "deepseek":
        from generesolver.integrations import deepseek_client
        return deepseek_client.complete_json(system, user, max_tokens=max_tokens, model=model,
                                             temperature=temperature)
    if backend == "openai":
        from generesolver.integrations import openai_client
        return openai_client.complete_json(system, user, max_tokens=max_tokens, model=model,
                                           temperature=temperature)
    from generesolver.integrations import anthropic_client
    return anthropic_client.complete_json(system, user, max_tokens=max_tokens, model=model,
                                          temperature=temperature)


def complete_structured(system: str, user: str, *, schema_model: type, max_tokens: int,
                        model: str | None = None, backend: str | None = None):
    backend = _resolve_backend(backend)
    if backend == "medgemma":
        from generesolver.integrations import medgemma_client
        return medgemma_client.complete_structured(system, user, schema_model=schema_model,
                                                    max_tokens=max_tokens, model=model)
    if backend == "deepseek":
        from generesolver.integrations import deepseek_client
        try:
            return deepseek_client.complete_structured(system, user, schema_model=schema_model,
                                                        max_tokens=max_tokens, model=model)
        except Exception as exc:
            try:
                raw = deepseek_client.complete_text(system, user, max_tokens=max_tokens, model=model)
                cleaned = deepseek_client._strip_fences(raw)
                parsed = deepseek_client._extract_json(cleaned)
                obj = parsed[0] if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) else parsed
                return schema_model.model_validate(obj)
            except Exception:
                raise
    if backend == "openai":
        from generesolver.integrations import openai_client
        return openai_client.complete_structured(system, user, schema_model=schema_model,
                                                  max_tokens=max_tokens, model=model)
    from generesolver.integrations import anthropic_client
    return anthropic_client.complete_structured(system, user, schema_model=schema_model,
                                                max_tokens=max_tokens, model=model)


def complete_text(system: str, user: str, *, max_tokens: int, model: str | None = None,
                  temperature: float = 0.0, backend: str | None = None) -> str:
    backend = _resolve_backend(backend)
    if backend == "medgemma":
        from generesolver.integrations import medgemma_client
        return medgemma_client.complete_text(system, user, max_tokens=max_tokens, model=model,
                                              temperature=temperature)
    if backend == "deepseek":
        from generesolver.integrations import deepseek_client
        return deepseek_client.complete_text(system, user, max_tokens=max_tokens, model=model,
                                             temperature=temperature)
    if backend == "openai":
        from generesolver.integrations import openai_client
        return openai_client.complete_text(system, user, max_tokens=max_tokens, model=model,
                                           temperature=temperature)
    from generesolver.integrations import anthropic_client
    return anthropic_client.complete_text(system, user, max_tokens=max_tokens, model=model,
                                          temperature=temperature)
