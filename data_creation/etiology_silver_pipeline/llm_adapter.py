from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any


def add_source_dir(source_dir: Path) -> None:
    source = str(source_dir.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)


def complete_json(
    *,
    client_module: str,
    system: str,
    payload: dict[str, Any],
    model: str | None = None,
    max_tokens: int = 1600,
) -> dict[str, Any]:
    client = importlib.import_module(client_module)
    kwargs: dict[str, Any] = {
        "system": system,
        "user": json.dumps(payload, ensure_ascii=False),
        "max_tokens": max_tokens,
    }
    if model:
        kwargs["model"] = model
    result = client.complete_json(**kwargs)
    if not isinstance(result, dict):
        raise TypeError(f"{client_module}.complete_json() did not return an object")
    return result
