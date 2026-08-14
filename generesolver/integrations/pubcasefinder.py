from __future__ import annotations

import logging
import random
import re
import threading
import time

import httpx

from generesolver.integrations._disk_cache import disk_cached

logger = logging.getLogger(__name__)

_API = "https://pubcasefinder.dbcls.jp/api/get_ranked_list"
_USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
)


def _headers() -> dict[str, str]:
    return {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "application/json,text/javascript,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
_TIMEOUT = 30.0
_RETRY_ATTEMPTS = 4
_RETRY_BACKOFF = 1.5
_MAX_BACKOFF = 30.0

_MIN_INTERVAL = 1.1
_throttle_lock = threading.Lock()
_last_request = [0.0]


def _throttle() -> None:
    with _throttle_lock:
        wait = _MIN_INTERVAL - (time.monotonic() - _last_request[0])
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.monotonic()
_RETRY_STATUS = {429, 500, 502, 503, 504}
_DOWN_STATUS = {502, 503, 504}
_dead_targets: set[str] = set()


@disk_cached("pubcasefinder")
def _raw_ranked(hpo_csv: str, target: str) -> list[dict]:
    params = {"target": target, "format": "json", "hpo_id": hpo_csv}
    last_exc: Exception | None = None
    with httpx.Client(timeout=_TIMEOUT) as client:
        for attempt in range(_RETRY_ATTEMPTS):
            _throttle()
            try:
                resp = client.get(_API, params=params, headers=_headers())
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                retriable = isinstance(exc, httpx.TransportError) or status in _RETRY_STATUS
                last_exc = exc
                if retriable and attempt < _RETRY_ATTEMPTS - 1:
                    delay = _RETRY_BACKOFF * (2 ** attempt)
                    if status == 429:
                        ra = exc.response.headers.get("Retry-After")
                        if ra:
                            try:
                                delay = max(delay, float(ra))
                            except ValueError:
                                pass
                    time.sleep(min(delay, _MAX_BACKOFF))
                    continue
                raise
    raise last_exc if last_exc else RuntimeError("pubcasefinder: unreachable")


def _disease_id(item: dict, target: str) -> str | None:
    if target == "omim":
        m = re.search(r"/entry/(\d+)", item.get("omim_url", "") or "")
        return f"OMIM:{m.group(1)}" if m else None
    oid = item.get("orpha_id")
    return f"ORPHA:{oid}" if oid else None


def ranked_diseases(hpo_ids, target: str = "omim", top_n: int = 20) -> list[dict]:
    ids = [h for h in (hpo_ids or []) if h]
    if not ids:
        return []
    if target in _dead_targets:
        return []
    hpo_csv = ",".join(sorted(ids))
    try:
        data = _raw_ranked(hpo_csv, target)
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in _DOWN_STATUS:
            _dead_targets.add(target)
            logger.warning(
                "PubCaseFinder target=%s is down (HTTP %s) - skipping it for the rest "
                "of this run; other targets unaffected.", target, status)
        else:
            logger.warning("PubCaseFinder failed (%s): %s", target, exc)
        return []

    out: list[dict] = []
    for item in (data or [])[:top_n]:
        if not isinstance(item, dict):
            continue
        did = _disease_id(item, target)
        if not did:
            continue
        genes = item.get("hgnc_gene_symbol") or []
        if isinstance(genes, str):
            genes = [genes]
        out.append({
            "disease_id": did,
            "name": (item.get("omim_disease_name_en")
                     or item.get("orpha_disease_name_en") or did),
            "genes": [g for g in genes if g],
            "score": float(item.get("score") or 0.0),
            "rank": int(item.get("rank") or 0),
            "source": "pubcasefinder",
        })
    if out:
        logger.info("PubCaseFinder: %d candidate disease(s) (target=%s)", len(out), target)
    return out


def is_available() -> bool:
    return True
