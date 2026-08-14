from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

from openai import BadRequestError, OpenAI

from generesolver.config import OUTPUTS_DIR
from generesolver.integrations._llm_errors import LLMResponseError

__all__ = ["complete_json", "complete_structured", "complete_text", "LLMResponseError"]

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o"

_CACHE_DIR = OUTPUTS_DIR / "_llm_cache"
_CACHE_DISABLED_ENV = "GENERESOLVER_LLM_CACHE_DISABLE"
_BACKEND_TAG = "openai"

_client_singleton: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client_singleton
    if _client_singleton is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Get a key from platform.openai.com/api-keys "
                "before using the openai backend.\n"
                "PowerShell: $env:OPENAI_API_KEY = '<key>'\n"
                "Bash:       export OPENAI_API_KEY=<key>"
            )
        base_url = os.environ.get("GENERESOLVER_OPENAI_BASE_URL", DEFAULT_BASE_URL)
        _client_singleton = OpenAI(base_url=base_url, api_key=api_key)
    return _client_singleton


def _model_name() -> str:
    return os.environ.get("GENERESOLVER_OPENAI_MODEL", DEFAULT_MODEL)


def _cache_disabled() -> bool:
    return os.environ.get(_CACHE_DISABLED_ENV, "").strip() in {"1", "true", "TRUE", "yes"}


def _cache_key(model: str, system: str, user: str, max_tokens: int,
               response_format: dict | None = None, temperature: float = 0.0) -> str:
    rf = (response_format or {}).get("type", "")
    temp_suffix = f"\n---\n{temperature}" if temperature else ""
    payload = f"{_BACKEND_TAG}\n{model}\n---\n{system}\n---\n{user}\n---\n{max_tokens}\n---\n{rf}{temp_suffix}"
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
        logger.warning("OpenAI cache read failed for %s: %s", key[:8], exc)
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
        logger.warning("OpenAI cache write failed for %s: %s", key[:8], exc)
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


def _extract_json(cleaned: str) -> dict | list:
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch in "{[":
            try:
                value, _end = decoder.raw_decode(cleaned, i)
            except json.JSONDecodeError:
                continue
            if isinstance(value, (dict, list)):
                return value
    raise json.JSONDecodeError("no JSON value found in response", cleaned, 0)


def complete_text(
    system: str,
    user: str,
    *,
    max_tokens: int,
    model: str | None = None,
    response_format: dict | None = None,
    temperature: float = 0.0,
) -> str:
    model = model or _model_name()
    key = _cache_key(model, system, user, max_tokens, response_format, temperature)

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
    try:
        resp = _get_client().chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            **extra,
        )
    except BadRequestError as exc:
        if "temperature" in str(exc).lower():
            resp = _get_client().chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                **extra,
            )
        else:
            raise
    raw = resp.choices[0].message.content or ""
    _cache_write(key, raw)
    return raw


def complete_json(
    system: str,
    user: str,
    *,
    max_tokens: int,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict:
    json_system = (
        system.rstrip()
        + "\n\nReturn only one valid JSON object. "
          "Do not include Markdown fences or explanatory text."
    )

    raw = complete_text(
        json_system,
        user,
        max_tokens=max_tokens,
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    cleaned = _strip_fences(raw)

    try:
        obj = _extract_json(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(
            f"OpenAI response was not valid JSON: {exc.msg} "
            f"(first 200 chars: {cleaned[:200]!r})",
            raw_payload=cleaned,
        ) from exc

    if not isinstance(obj, dict):
        raise LLMResponseError(
            "OpenAI returned valid JSON, but the top-level value was "
            f"{type(obj).__name__}; expected a JSON object. "
            f"(first 200 chars: {cleaned[:200]!r})",
            raw_payload=cleaned,
        )

    return obj


def complete_structured(system: str, user: str, *, schema_model: type, max_tokens: int,
                        model: str | None = None):
    obj = complete_json(system, user, max_tokens=max_tokens, model=model)
    try:
        return schema_model.model_validate(obj)
    except Exception as exc:
        raise LLMResponseError(f"OpenAI structured output was not schema-valid: {exc}") from exc
