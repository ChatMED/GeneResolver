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

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324"

_CACHE_DIR = OUTPUTS_DIR / "_llm_cache"
_CACHE_DISABLED_ENV = "GENERESOLVER_LLM_CACHE_DISABLE"
_BACKEND_TAG = "deepseek"

_client_singleton: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client_singleton
    if _client_singleton is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Get a key from "
                "openrouter.ai/settings/keys before using the deepseek backend.\n"
                "PowerShell: $env:OPENROUTER_API_KEY = '<key>'\n"
                "Bash:       export OPENROUTER_API_KEY=<key>"
            )
        base_url = os.environ.get("GENERESOLVER_DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
        _client_singleton = OpenAI(base_url=base_url, api_key=api_key)
    return _client_singleton


def _model_name() -> str:
    return os.environ.get("GENERESOLVER_DEEPSEEK_MODEL", DEFAULT_MODEL)


def _cache_disabled() -> bool:
    return os.environ.get(_CACHE_DISABLED_ENV, "").strip() in {"1", "true", "TRUE", "yes"}


def _cache_key(model: str, system: str, user: str, max_tokens: int,
               response_format: dict | None = None, temperature: float = 0.0,
               top_p: float | None = None, seed: int | None = None) -> str:
    rf = (response_format or {}).get("type", "")
    temp_suffix = f"\n---\n{temperature}" if temperature else ""
    sampling_suffix = f"\n---\ntop_p={top_p}\n---\nseed={seed}" if (top_p is not None or seed is not None) else ""
    payload = (f"{_BACKEND_TAG}\n{model}\n---\n{system}\n---\n{user}\n---\n{max_tokens}\n---\n"
              f"{rf}{temp_suffix}{sampling_suffix}")
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
        logger.warning("DeepSeek cache read failed for %s: %s", key[:8], exc)
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
        logger.warning("DeepSeek cache write failed for %s: %s", key[:8], exc)
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
    top_p: float | None = None,
    seed: int | None = None,
) -> str:
    model = model or _model_name()
    key = _cache_key(model, system, user, max_tokens, response_format, temperature, top_p, seed)

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
    if top_p is not None:
        extra["top_p"] = top_p
    if seed is not None:
        extra["seed"] = seed
    resp = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **extra,
        frequency_penalty=0.4,
    )
    if not resp.choices:
        raise LLMResponseError(
            f"DeepSeek response had no choices (finish_reason/prompt filtering?): {resp!r}"
        )
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
    top_p: float | None = None,
    seed: int | None = None,
) -> dict:
    raw = complete_text(system, user, max_tokens=max_tokens, model=model, temperature=temperature,
                        response_format={"type": "json_object"}, top_p=top_p, seed=seed)
    cleaned = _strip_fences(raw)
    try:
        return _extract_json(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(
            f"DeepSeek response was not valid JSON: {exc.msg} "
            f"(first 200 chars: {cleaned[:200]!r})",
            raw_payload=cleaned,
        ) from exc


def complete_structured(system: str, user: str, *, schema_model: type, max_tokens: int,
                        model: str | None = None):
    obj = complete_json(system, user, max_tokens=max_tokens, model=model)
    try:
        return schema_model.model_validate(obj)
    except Exception as exc:
        raise LLMResponseError(f"DeepSeek structured output was not schema-valid: {exc}") from exc
