from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

import anthropic

from generesolver.config import OUTPUTS_DIR
from generesolver.integrations._llm_errors import LLMResponseError

CLAUDE_MODEL = "claude-opus-5"  # default only; dataset scripts pass the model they used

__all__ = ["complete_json", "complete_structured", "complete_text", "LLMResponseError"]

logger = logging.getLogger(__name__)

_CACHE_DIR = OUTPUTS_DIR / "_llm_cache"
_CACHE_DISABLED_ENV = "GENERESOLVER_LLM_CACHE_DISABLE"


def _get_client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before running any agent that "
            "calls Claude (hpo_mapper, mechanism, report_generator).\n"
            "PowerShell: $env:ANTHROPIC_API_KEY = '<key>'\n"
            "Bash:       export ANTHROPIC_API_KEY=<key>"
        )
    return anthropic.Anthropic(api_key=key)


def _cache_disabled() -> bool:
    return os.environ.get(_CACHE_DISABLED_ENV, "").strip() in {"1", "true", "TRUE", "yes"}


def _cache_key(model: str, system: str, user: str, max_tokens: int,
               temperature: float = 0.0) -> str:
    temp_suffix = f"\n---\n{temperature}" if temperature else ""
    payload = f"{model}\n---\n{system}\n---\n{user}\n---\n{max_tokens}{temp_suffix}"
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
        logger.warning("LLM cache read failed for %s: %s", key[:8], exc)
        return None


def _cache_write(key: str, response: str) -> None:
    if _cache_disabled():
        return
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"response": response}, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("LLM cache write failed for %s: %s", key[:8], exc)
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
    temperature: float = 0.0,
) -> str:
    model = model or CLAUDE_MODEL
    key = _cache_key(model, system, user, max_tokens, temperature)

    cached = _cache_read(key)
    if cached is not None:
        return cached

    client = _get_client()
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        kwargs["system"] = system

    msg = client.messages.create(**kwargs)
    text_block = next((b for b in msg.content if b.type == "text"), None)
    if text_block is None:
        raise LLMResponseError(
            f"Claude response had no text block (stop_reason={msg.stop_reason!r}, "
            f"block types={[b.type for b in msg.content]!r})"
        )
    raw = text_block.text
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
    raw = complete_text(system, user, max_tokens=max_tokens, model=model, temperature=temperature)
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(
            f"Claude response was not valid JSON: {exc.msg} "
            f"(first 200 chars: {cleaned[:200]!r})"
        ) from exc


def complete_structured(system: str, user: str, *, schema_model: type, max_tokens: int,
                        model: str | None = None):
    obj = complete_json(system, user, max_tokens=max_tokens, model=model)
    try:
        return schema_model.model_validate(obj)
    except Exception as exc:
        raise LLMResponseError(f"Claude structured output was not schema-valid: {exc}") from exc
