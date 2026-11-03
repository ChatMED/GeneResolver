from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

from openai import OpenAI

from generesolver.config import OUTPUTS_DIR
from generesolver.integrations._llm_errors import LLMResponseError

__all__ = ["complete_json", "complete_structured", "complete_text", "LLMResponseError"]

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "" #need medgemma url
DEFAULT_MODEL = "google/medgemma-27b-it"

_CACHE_DIR = OUTPUTS_DIR / "_llm_cache"
_CACHE_DISABLED_ENV = "GENERESOLVER_LLM_CACHE_DISABLE"
_BACKEND_TAG = "medgemma"


_client_singleton: OpenAI | None = None

_DEFAULT_TIMEOUT_S = 180.0
_DEFAULT_MAX_RETRIES = 1


def _get_client() -> OpenAI:
    global _client_singleton
    if _client_singleton is None:
        base_url = os.environ.get("GENERESOLVER_MEDGEMMA_BASE_URL", DEFAULT_BASE_URL)
        api_key = os.environ.get("GENERESOLVER_MEDGEMMA_API_KEY", "not-needed")
        timeout_s = float(os.environ.get("GENERESOLVER_MEDGEMMA_TIMEOUT_S", _DEFAULT_TIMEOUT_S))
        max_retries = int(os.environ.get("GENERESOLVER_MEDGEMMA_MAX_RETRIES", _DEFAULT_MAX_RETRIES))
        _client_singleton = OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=max_retries,
        )
    return _client_singleton


def _model_name() -> str:
    return os.environ.get("GENERESOLVER_MEDGEMMA_MODEL", DEFAULT_MODEL)


def _cache_disabled() -> bool:
    return os.environ.get(_CACHE_DISABLED_ENV, "").strip() in {"1", "true", "TRUE", "yes"}


def _cache_key(model: str, system: str, user: str, max_tokens: int,
               response_format: dict | None = None, extra_body: dict | None = None,
               temperature: float = 0.0) -> str:
    rf = (response_format or {}).get("type", "")
    eb = json.dumps(extra_body, sort_keys=True) if extra_body else ""
    temp_suffix = f"\n---\n{temperature}" if temperature else ""
    payload = (f"{_BACKEND_TAG}\n{model}\n---\n{system}\n---\n{user}\n---\n{max_tokens}\n---\n{rf}"
               f"\n---\n{eb}{temp_suffix}")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return _CACHE_DIR / key[:2] / f"{key}.json"


def _cache_read(key: str) -> str | None:
    if _cache_disabled():
        return None
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))["response"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        logger.warning("MedGemma cache read failed for %s: %s", key[:8], exc)
        return None


def _cache_write(key: str, response: str) -> None:
    if _cache_disabled():
        return
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"response": response, "backend": _BACKEND_TAG}, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("MedGemma cache write failed for %s: %s", key[:8], exc)
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def complete_text(
    system: str,
    user: str,
    *,
    max_tokens: int,
    model: str | None = None,
    response_format: dict | None = None,
    extra_body: dict | None = None,
    temperature: float = 0.0,
) -> str:
    model = model or _model_name()
    key = _cache_key(model, system, user, max_tokens, response_format, extra_body, temperature)

    cached = _cache_read(key)
    if cached is not None:
        return cached

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    extra = {}
    if response_format:
        extra["response_format"] = response_format
    if extra_body:
        extra["extra_body"] = extra_body
    resp = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **extra,
        frequency_penalty=0.4,
    )
    raw = resp.choices[0].message.content or ""
    _cache_write(key, raw)
    return raw


def complete_structured(
    system: str,
    user: str,
    *,
    schema_model: type,
    max_tokens: int,
    model: str | None = None,
):
    obj = complete_json(system, user, max_tokens=max_tokens, model=model,
                        guided_json=schema_model.model_json_schema())
    try:
        return schema_model.model_validate(obj)
    except Exception as exc:
        raise LLMResponseError(f"MedGemma structured output was not schema-valid: {exc}") from exc


def _json_error_context(text: str, pos: int, window: int = 300) -> str:
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    return text[start:end]


def complete_json(
    system: str,
    user: str,
    *,
    max_tokens: int,
    model: str | None = None,
    guided_json: dict | None = None,
    temperature: float = 0.0,
) -> dict:
    instruction = "Return exactly one valid JSON object without Markdown or prose."
    if guided_json is not None:
        instruction += " Match this JSON schema: " + json.dumps(guided_json)
    try:
        raw = complete_text((system + "\n" + instruction).strip(), user,
                            max_tokens=max_tokens, model=model, temperature=temperature)
    except Exception as exc:
        raise LLMResponseError(f"MedGemma request failed: {exc}") from exc
    cleaned = _strip_fences(raw)

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"MedGemma response was not valid JSON: {exc}",
                               raw_payload=cleaned) from exc
    if not isinstance(obj, dict):
        raise LLMResponseError("MedGemma returned non-object JSON", raw_payload=cleaned)
    if guided_json is not None:
        import jsonschema
        try:
            jsonschema.validate(obj, guided_json)
        except jsonschema.ValidationError as exc:
            raise LLMResponseError(f"MedGemma JSON failed schema validation: {exc.message}",
                                   raw_payload=cleaned) from exc
    return obj
