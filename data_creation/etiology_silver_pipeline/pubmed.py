from __future__ import annotations

import time
from xml.etree import ElementTree as ET

import httpx

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ID_CONVERTER = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
BIOC = (
    "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/"
    "pmcoa.cgi/BioC_json/{pmcid}/unicode"
)
EUROPEPMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"


def search(client: httpx.Client, name: str, synonyms: list[str], retmax: int) -> tuple[str, list[str]]:
    terms = [name] + [x for x in synonyms if x and x.lower() != name.lower()]
    disease = " OR ".join(f'"{term}"[Title/Abstract]' for term in terms[:4])
    query = (
        f"({disease}) AND "
        '(case reports[Publication Type] OR "case report"[Title/Abstract] '
        'OR "case presentation"[Title/Abstract]) AND humans[MeSH Terms]'
    )
    response = client.get(
        f"{EUTILS}/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": retmax,
            "sort": "pub date",
        },
    )
    response.raise_for_status()
    return query, response.json().get("esearchresult", {}).get("idlist", [])


def summary(client: httpx.Client, pmid: str) -> dict:
    response = client.get(
        f"{EUTILS}/esummary.fcgi",
        params={"db": "pubmed", "id": pmid, "retmode": "json"},
    )
    response.raise_for_status()
    return response.json().get("result", {}).get(pmid, {})


_ABSTRACT_TRAILER_PREFIXES = ("DOI:", "PMID:", "PMCID:", "PMCID :", "Copyright", "©", "Comment in:", "Erratum")


def abstract(client: httpx.Client, pmid: str) -> str:
    response = client.get(
        f"{EUTILS}/efetch.fcgi",
        params={"db": "pubmed", "id": pmid, "rettype": "abstract", "retmode": "text"},
    )
    response.raise_for_status()
    paragraphs = [p.strip() for p in response.text.strip().split("\n\n") if p.strip()]
    start = 0
    for i, p in enumerate(paragraphs):
        if p.startswith("(") or p.startswith("Author information") or len(p) < 200:
            start = i + 1
            continue
        break
    body_paragraphs = []
    for p in paragraphs[start:]:
        if p.startswith(_ABSTRACT_TRAILER_PREFIXES):
            break
        body_paragraphs.append(p)
    return "\n\n".join(body_paragraphs).strip()


def pmcid(client: httpx.Client, pmid: str) -> str | None:
    response = client.get(ID_CONVERTER, params={"ids": pmid, "format": "json"})
    response.raise_for_status()
    rows = response.json().get("records", [])
    return rows[0].get("pmcid") if rows else None


def pmcid_batch(client: httpx.Client, pmids: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for i in range(0, len(pmids), 200):
        chunk = pmids[i:i + 200]
        response = client.get(ID_CONVERTER, params={"ids": ",".join(chunk), "format": "json"})
        response.raise_for_status()
        for row in response.json().get("records", []):
            pmid = str(row.get("pmid") or row.get("requested-id") or "")
            if pmid:
                out[pmid] = row.get("pmcid")
    return out


def fulltext(client: httpx.Client, pmc_id: str) -> list | None:
    for attempt in range(3):
        try:
            response = client.get(
                BIOC.format(pmcid=pmc_id), headers={"Connection": "close"}
            )
            if response.status_code != 200 or not response.text.strip():
                return None
            return response.json()
        except (httpx.RemoteProtocolError, ValueError):
            if attempt == 2:
                return None
            time.sleep(1)
    return None


def passages(collections: list) -> list[dict]:
    output: list[dict] = []
    for collection in collections:
        for document in collection.get("documents", []):
            for passage in document.get("passages", []):
                text = passage.get("text", "").strip()
                if not text:
                    continue
                infons = passage.get("infons") or {}
                output.append(
                    {
                        "passage_id": f"P{len(output) + 1:04d}",
                        "section": infons.get("section") or infons.get("type") or "",
                        "offset": passage.get("offset"),
                        "text": text,
                    }
                )
    return output


def article_text(rows: list[dict]) -> str:
    return "\n\n".join(
        f"[{row['passage_id']}] {row['section']}: {row['text']}" for row in rows
    )


def europepmc_fulltext(client: httpx.Client, pmc_id: str) -> str | None:
    response = client.get(EUROPEPMC_FULLTEXT.format(pmcid=pmc_id))
    if response.status_code != 200 or not response.text.strip():
        return None
    return response.text


def europepmc_passages(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    parent_map = {child: parent for parent in root.iter() for child in parent}

    def section_title(p_elem) -> str:
        node = parent_map.get(p_elem)
        while node is not None:
            if node.tag == "sec":
                title_elem = node.find("title")
                return "".join(title_elem.itertext()).strip() if title_elem is not None else ""
            if node.tag == "abstract":
                return "abstract"
            node = parent_map.get(node)
        return ""

    output: list[dict] = []
    for p in root.iter("p"):
        text = "".join(p.itertext()).strip()
        if not text:
            continue
        output.append(
            {
                "passage_id": f"P{len(output) + 1:04d}",
                "section": section_title(p),
                "offset": None,
                "text": text,
            }
        )
    return output


def get_article_passages(client: httpx.Client, pmc_id: str) -> list[dict] | None:
    bioc = fulltext(client, pmc_id)
    if bioc:
        rows = passages(bioc)
        if rows:
            return rows
    xml_text = europepmc_fulltext(client, pmc_id)
    if not xml_text:
        return None
    try:
        rows = europepmc_passages(xml_text)
    except ET.ParseError:
        return None
    return rows or None
