from __future__ import annotations

import logging
import time

import httpx

from generesolver.config import GNOMAD_GRAPHQL_URL

logger = logging.getLogger(__name__)

_GNOMAD_V4 = "gnomad_r4"
_GNOMAD_V2 = "gnomad_r2_1"
_REQUEST_INTERVAL = 0.5
_RETRY_ATTEMPTS = 3

_VARIANT_QUERY = """
query VariantQuery($variantId: String!, $dataset: DatasetId!) {
  variant(variantId: $variantId, dataset: $dataset) {
    variantId
    exome {
      ac
      an
      af
      populations { id ac an }
    }
    genome {
      ac
      an
      af
      populations { id ac an }
    }
  }
}
"""

_POPMAX_ELIGIBLE_POPULATIONS = {"afr", "amr", "asj", "eas", "fin", "mid", "nfe", "sas"}


def get_allele_frequency(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    genome_build: str = "GRCh38",
) -> dict:
    chrom_clean = chrom.replace("chr", "")
    variant_id = f"{chrom_clean}-{pos}-{ref}-{alt}"
    dataset = _GNOMAD_V4 if genome_build == "GRCh38" else _GNOMAD_V2

    result = _query_gnomad(variant_id, dataset)

    if result.get("af_status") != "observed" and dataset == _GNOMAD_V4:
        fallback = _query_gnomad(variant_id, _GNOMAD_V2)
        if fallback.get("af_status") == "observed":
            fallback["dataset"] = _GNOMAD_V2
            return fallback
        if result.get("af_status") == "absent" and fallback.get("af_status") == "absent":
            return {"af_status": "absent", "dataset": f"{_GNOMAD_V4}+{_GNOMAD_V2}"}
        return {}

    return result if result.get("af_status") in ("observed", "absent") else {}


def _is_not_found(payload: dict) -> bool:
    errors = payload.get("errors") or []
    return any("not found" in str(e.get("message", "")).lower() for e in errors)


def _query_gnomad(variant_id: str, dataset: str) -> dict:
    with httpx.Client(timeout=30.0) as client:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = client.post(
                    GNOMAD_GRAPHQL_URL,
                    json={"query": _VARIANT_QUERY, "variables": {"variantId": variant_id, "dataset": dataset}},
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code == 400:
                    logger.debug("gnomAD 400 for %s (dataset=%s): %s", variant_id, dataset, resp.text[:200])
                    try:
                        return {"af_status": "absent"} if _is_not_found(resp.json()) else {}
                    except ValueError:
                        return {}
                resp.raise_for_status()
                data = resp.json()
                variant = (data.get("data") or {}).get("variant")
                if not variant:
                    return {"af_status": "absent"} if _is_not_found(data) or not data.get("errors") else {}
                time.sleep(_REQUEST_INTERVAL)
                return {**_parse_gnomad_result(variant, dataset), "af_status": "observed"}
            except Exception as exc:
                if attempt < _RETRY_ATTEMPTS - 1:
                    time.sleep(1.0)
                else:
                    logger.debug("gnomAD query failed for %s: %s", variant_id, exc)
                    return {}
    return {}


def _popmax(populations: list[dict] | None) -> tuple[float | None, int | None, str | None]:
    best_af, best_an, best_pop = None, None, None
    for pop in populations or []:
        pid = pop.get("id")
        an = pop.get("an")
        ac = pop.get("ac")
        if pid not in _POPMAX_ELIGIBLE_POPULATIONS or not an:
            continue
        af = ac / an if ac is not None else None
        if af is not None and (best_af is None or af > best_af):
            best_af, best_an, best_pop = af, an, pid
    return best_af, best_an, best_pop


def _parse_gnomad_result(variant: dict, dataset: str) -> dict:
    exome = variant.get("exome") or {}
    genome = variant.get("genome") or {}

    source_data = exome if exome.get("af") is not None else genome
    source_name = "exome" if exome.get("af") is not None else "genome"

    af = source_data.get("af")
    ac = source_data.get("ac")
    an = source_data.get("an")
    af_popmax, af_popmax_allele_number, af_popmax_population = _popmax(source_data.get("populations"))

    return {
        "af": af,
        "af_popmax": af_popmax,
        "af_popmax_allele_number": af_popmax_allele_number,
        "af_popmax_population": af_popmax_population,
        "ac": ac,
        "an": an,
        "source": source_name,
        "dataset": dataset,
        "variant_id": variant.get("variantId"),
    }
