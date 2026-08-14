from __future__ import annotations

import functools
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable

from generesolver.config import OUTPUTS_DIR

logger = logging.getLogger(__name__)


def _cache_dir(namespace: str) -> Path:
    return OUTPUTS_DIR / f"_{namespace}_cache"


def _key_for(namespace: str, args: tuple, kwargs: dict) -> str:
    payload = json.dumps(
        {"ns": namespace, "args": args, "kwargs": kwargs},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read(namespace: str, key: str) -> Any | None:
    path = _cache_dir(namespace) / key[:2] / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("disk-cache read failed for %s/%s: %s", namespace, key[:8], exc)
        return None


def _write(namespace: str, key: str, value: Any) -> None:
    path = _cache_dir(namespace) / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(value), encoding="utf-8")
    except (OSError, TypeError) as exc:
        logger.warning("disk-cache write failed for %s/%s: %s", namespace, key[:8], exc)


def disk_cached(namespace: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = _key_for(namespace, args, kwargs)
            hit = _read(namespace, key)
            if hit is not None:
                return hit
            result = fn(*args, **kwargs)
            _write(namespace, key, result)
            return result
        wrapper.__wrapped__ = fn
        return wrapper
    return decorator
