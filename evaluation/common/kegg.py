from __future__ import annotations

import httpx

KEGG_REST = "https://rest.kegg.jp"


def _find(client: httpx.Client, db: str, keyword: str) -> list[tuple[str, str]]:
    keyword = keyword.strip()
    if not keyword:
        return []
    response = client.get(f"{KEGG_REST}/find/{db}/{keyword}")
    if response.status_code == 404:
        return []
    response.raise_for_status()
    entries: list[tuple[str, str]] = []
    for line in response.text.splitlines():
        if not line.strip():
            continue
        entry_id, _, name = line.partition("\t")
        entries.append((entry_id.strip(), name.strip()))
    return entries


def find_compound(client: httpx.Client, keyword: str) -> list[tuple[str, str]]:
    return _find(client, "compound", keyword)


def find_disease(client: httpx.Client, keyword: str) -> list[tuple[str, str]]:
    return _find(client, "disease", keyword)


def get_entry(client: httpx.Client, entry_id: str) -> dict[str, str]:
    response = client.get(f"{KEGG_REST}/get/{entry_id}")
    if response.status_code == 404:
        return {}
    response.raise_for_status()
    fields: dict[str, list[str]] = {}
    current = None
    for line in response.text.splitlines():
        if line.startswith("///"):
            break
        if line and not line[0].isspace():
            parts = line.split(None, 1)
            current = parts[0]
            fields.setdefault(current, [])
            if len(parts) > 1:
                fields[current].append(parts[1].strip())
        elif current:
            fields[current].append(line.strip())
    return {k: " ".join(v).strip() for k, v in fields.items()}
