from __future__ import annotations

import csv
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from generesolver.config import OUTPUTS_DIR
from generesolver.schemas.case import CaseState

logger = logging.getLogger(__name__)

_FHIR_ID_INVALID = re.compile(r"[^A-Za-z0-9\-.]+")


def _fhir_id(raw: str) -> str:
    cleaned = _FHIR_ID_INVALID.sub("-", raw).strip("-")
    return (cleaned or "case")[:64]


def run(state: CaseState) -> CaseState:
    case_dir: Path = OUTPUTS_DIR / state.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    _export_json(state, case_dir / "report.json")
    if state.ranked_genes:
        _export_csv(state, case_dir / "gene_ranking.csv")
    _export_fhir(state, case_dir / "diagnostic_report.fhir.json")
    _export_phenopacket(state, case_dir / "phenopacket.json")

    logger.info("Exports written to %s", case_dir)
    state.mark_done("exporter")
    return state

def _export_json(state: CaseState, path: Path) -> None:
    data = state.model_dump()
    path.write_text(
        json.dumps(data, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("JSON export → %s", path)

_CSV_FIELDS = [
    "rank", "gene", "final_score", "confidence",
    "score_phenotype", "score_variant", "score_gene_disease",
    "score_literature", "score_gwas",
    "best_variant_id", "known_disease_associations", "prioritization_reason",
]


def _export_csv(state: CaseState, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for gr in state.ranked_genes:
            writer.writerow({
                "rank": gr.rank,
                "gene": gr.gene,
                "final_score": round(gr.final_score, 4),
                "confidence": gr.confidence,
                "score_phenotype": round(gr.score_phenotype, 4),
                "score_variant": round(gr.score_variant, 4),
                "score_gene_disease": round(gr.score_gene_disease, 4),
                "score_literature": round(gr.score_literature, 4),
                "score_gwas": round(gr.score_gwas, 4),
                "best_variant_id": gr.best_variant_id or "",
                "known_disease_associations": "; ".join(gr.known_disease_associations),
                "prioritization_reason": gr.prioritization_reason,
            })
    logger.info("CSV export → %s", path)


def _coding(system: str, code: str | None, display: str) -> dict:
    if code:
        return {"system": system, "code": code, "display": display}
    return {"display": display}


def _patient(state: CaseState) -> dict:
    patient: dict = {
        "resourceType": "Patient",
        "id": f"patient-{_fhir_id(state.case_id)}",
        "meta": {"security": [{
            "system": "http://terminology.hl7.org/CodeSystem/v3-ActReason",
            "code": "ANONYED",
            "display": "anonymized",
        }]},
    }
    sex = (state.sex or state.input.sex or "").lower()
    if sex in ("male", "female"):
        patient["gender"] = sex
    elif sex:
        patient["gender"] = "unknown"

    age = state.age_years if state.age_years is not None else state.input.age_years
    if age is not None:
        patient["extension"] = [{
            "url": "https://generesolver.io/fhir/StructureDefinition/age-at-presentation",
            "valueQuantity": {
                "value": age, "unit": "a",
                "system": "http://unitsofmeasure.org", "code": "a",
            },
        }]
    return patient


def _conditions(state: CaseState, subject: dict) -> list[dict]:
    out: list[dict] = []
    report = state.doctor_report
    candidates = report.possible_conditions if report else []

    for i, c in enumerate(candidates[:8]):
        name = c.get("name") or ""
        if not name:
            continue
        disease_id = str(c.get("disease_id") or "")
        system, code = "", None
        if disease_id.upper().startswith("OMIM"):
            system, code = "http://www.omim.org", disease_id.split(":")[-1]
        elif disease_id.upper().startswith("ORPHA"):
            system, code = "http://www.orpha.net", disease_id.split(":")[-1]
        elif disease_id.upper().startswith("MONDO"):
            system, code = "http://purl.obolibrary.org/obo/mondo.owl", disease_id.replace(":", "_")

        out.append({
            "resourceType": "Condition",
            "id": f"cond-{i}",
            "verificationStatus": {"coding": [_coding(
                "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                "provisional", "Provisional")]},
            "category": [{"coding": [_coding(
                "http://terminology.hl7.org/CodeSystem/condition-category",
                "encounter-diagnosis", "Encounter Diagnosis")]}],
            "code": {"coding": [_coding(system, code, name)], "text": name},
            "subject": subject,
            "note": [{"text": (
                f"Likelihood: {c.get('likelihood', 'unknown')}. "
                f"Supporting: {'; '.join(str(s) for s in (c.get('supporting') or [])) or 'n/a'}. "
                f"Against: {'; '.join(str(s) for s in (c.get('against') or [])) or 'n/a'}."
            )}],
        })
    return out


def _service_requests(state: CaseState, subject: dict) -> list[dict]:
    priority = {"decisive": "urgent", "supportive": "routine", "exclusionary": "routine"}
    out: list[dict] = []

    for i, t in enumerate(state.next_steps.tests):
        out.append({
            "resourceType": "ServiceRequest",
            "id": f"svcreq-{i}",
            "status": "draft",
            "intent": "proposal",
            "priority": priority.get(t.power.value, "routine"),
            "code": {
                "coding": [_coding("http://loinc.org", t.loinc_code, t.test)],
                "text": t.test,
            },
            "subject": subject,
            "reasonCode": [{"text": t.rationale}] if t.rationale else [],
            "note": [{"text": (
                f"Power: {t.power.value}. "
                f"A positive result would establish: {'; '.join(t.confirms) or 'n/a'}. "
                f"A negative result would rule out: {'; '.join(t.refutes) or 'n/a'}."
            )}],
        })
    return out


def _biochem_observations(state: CaseState, subject: dict) -> list[dict]:
    interp = {"increased": ("H", "High"), "decreased": ("L", "Low"), "normal": ("N", "Normal")}
    out: list[dict] = []

    for i, m in enumerate(state.biochem_panel.measurements):
        if m.direction.value not in interp:
            continue
        code, display = interp[m.direction.value]
        obs: dict = {
            "resourceType": "Observation",
            "id": f"lab-{i}",
            "status": "final",
            "category": [{"coding": [_coding(
                "http://terminology.hl7.org/CodeSystem/observation-category",
                "laboratory", "Laboratory")]}],
            "code": {"coding": [_coding("http://loinc.org", None, m.analyte)], "text": m.analyte},
            "subject": subject,
            "interpretation": [{"coding": [_coding(
                "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                code, display)]}],
        }
        if m.value is not None:
            obs["valueQuantity"] = {"value": m.value, "unit": m.unit or ""}
        if m.ref_low is not None and m.ref_high is not None:
            obs["referenceRange"] = [{
                "low": {"value": m.ref_low, "unit": m.unit or ""},
                "high": {"value": m.ref_high, "unit": m.unit or ""},
            }]
        if m.specimen:
            obs["note"] = [{"text": f"Specimen: {m.specimen}"}]
        out.append(obs)
    return out


def _gene_observations(state: CaseState, subject: dict) -> list[dict]:
    out: list[dict] = []
    for gr in state.ranked_genes[:5]:
        obs: dict = {
            "resourceType": "Observation",
            "id": f"obs-{gr.gene}",
            "status": "preliminary",
            "subject": subject,
            "code": {"coding": [_coding(
                "http://loinc.org", "81252-9", "Discrete genetic variant")]},
            "component": [
                {
                    "code": {"coding": [_coding(
                        "http://loinc.org", "48018-6", "Gene studied")]},
                    "valueCodeableConcept": {"coding": [_coding(
                        "http://www.genenames.org",
                        getattr(gr, "hgnc_id", None), gr.gene)]},
                },
                {
                    "code": {"coding": [_coding(
                        "http://loinc.org", "93044-6", "Level of evidence")]},
                    "valueString": f"{gr.confidence} (GeneResolver score {gr.final_score:.3f})",
                },
            ],
        }
        if gr.best_variant_id:
            obs["component"].append({
                "code": {"coding": [_coding("http://loinc.org", "47999-8", "Variant")]},
                "valueString": gr.best_variant_id,
            })
        out.append(obs)
    return out


def _export_fhir(state: CaseState, path: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    report = state.doctor_report

    urns: dict[str, str] = {}

    def _urn_for(resource_type: str, resource_id: str) -> str:
        key = f"{resource_type}/{resource_id}"
        if key not in urns:
            urns[key] = f"urn:uuid:{uuid.uuid4()}"
        return urns[key]

    patient = _patient(state)
    subject = {"reference": _urn_for("Patient", patient["id"])}

    conditions = _conditions(state, subject)
    service_requests = _service_requests(state, subject)
    observations = _gene_observations(state, subject) + _biochem_observations(state, subject)
    for r in [patient, *conditions, *service_requests, *observations]:
        _urn_for(r["resourceType"], r["id"])

    if report and report.case_summary:
        conclusion = report.case_summary
    elif state.ranked_genes:
        top = state.ranked_genes[0]
        conclusion = (f"Top candidate gene: {top.gene} (score={top.final_score:.2f}, "
                      f"confidence={top.confidence}). Expert review required.")
    else:
        conclusion = "No candidate genes identified. Expert review required."

    extensions = [
        {
            "url": "https://generesolver.io/fhir/StructureDefinition/decision-support-disclaimer",
            "valueString": (report.disclaimer if report
                            else "This is a decision-support report and requires clinical review."),
        },
        {
            "url": "https://generesolver.io/fhir/StructureDefinition/confidence-level",
            "valueString": state.confidence_level,
        },
    ]
    if report and report.classification:
        extensions.append({
            "url": "https://generesolver.io/fhir/StructureDefinition/etiology-classification",
            "valueString": (f"{report.classification}"
                            + (f" — {report.classification_reasoning}"
                               if report.classification_reasoning else "")),
        })
    if report and report.sources_consulted:
        extensions.append({
            "url": "https://generesolver.io/fhir/StructureDefinition/sources-consulted",
            "valueString": ", ".join(report.sources_consulted),
        })
    for limitation in (report.investigation_limitations if report else []):
        extensions.append({
            "url": "https://generesolver.io/fhir/StructureDefinition/investigation-limitation",
            "valueString": limitation,
        })

    diagnostic_report_id = f"generesolver-{_fhir_id(state.case_id)}"
    diagnostic_report = {
        "resourceType": "DiagnosticReport",
        "id": diagnostic_report_id,
        "meta": {"profile": [
            "http://hl7.org/fhir/uv/genomics-reporting/StructureDefinition/genomics-report"
        ]},
        "status": "partial",
        "category": [{"coding": [_coding(
            "http://terminology.hl7.org/CodeSystem/v2-0074", "GE", "Genetics")]}],
        "code": {"coding": [_coding(
            "http://loinc.org", "81247-9", "Master HL7 genetic variant reporting panel")]},
        "subject": subject,
        "effectiveDateTime": now,
        "issued": now,
        "result": [{"reference": _urn_for("Observation", o["id"])} for o in observations],
        "conclusion": conclusion,
        "conclusionCode": [
            {"coding": c["code"]["coding"], "text": c["code"].get("text", "")}
            for c in conditions
        ],
        "extension": extensions,
    }

    bundle = {
        "resourceType": "Bundle",
        "id": f"generesolver-bundle-{_fhir_id(state.case_id)}",
        "type": "collection",
        "timestamp": now,
        "entry": [
            {"fullUrl": _urn_for(r["resourceType"], r["id"]), "resource": r}
            for r in [patient, diagnostic_report, *conditions, *service_requests, *observations]
        ],
    }

    path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "FHIR export → %s (%d conditions, %d service requests, %d observations)",
        path, len(conditions), len(service_requests), len(observations),
    )


def _export_phenopacket(state: CaseState, path: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()

    sex_map = {"male": "MALE", "female": "FEMALE"}
    sex = sex_map.get((state.input.sex or "").lower(), "UNKNOWN_SEX")
    subject = {"id": state.case_id, "sex": sex}

    phenotypic_features: list[dict] = []
    for term in state.hpo_profile.terms:
        phenotypic_features.append({
            "type": {"id": term.hpo_id, "label": term.label},
            "excluded": term.status.value == "absent",
        })

    genomic_interps: list[dict] = []
    for comparison in state.acmg_assessments[:10]:
        acmg = comparison.computed_acmg_assessment
        genomic_interps.append({
            "subjectOrBiosampleId": state.case_id,
            "interpretationStatus": _acmg_to_ppack_status(acmg.classification),
            "variantInterpretation": {
                "acmgPathogenicityClassification": acmg.classification.upper(),
                "variationDescriptor": {
                    "id": acmg.variant_id,
                    "geneContext": {"symbol": acmg.gene},
                    "label": acmg.variant_id,
                },
            },
        })

    interpretations: list[dict] = []
    if state.ranked_genes:
        top = state.ranked_genes[0]
        best_disease = ""
        for be in state.biological_evidence:
            if be.gene == top.gene and be.disease_associations:
                best_disease = be.disease_associations[0]
                break
        interpretations.append({
            "id": f"interpretation-{state.case_id}",
            "progressStatus": "COMPLETED",
            "diagnosis": {
                "disease": {
                    "id": "UNKNOWN",
                    "label": best_disease or f"Suspected {top.gene}-related disorder",
                },
                "genomicInterpretations": genomic_interps,
            },
        })

    variants: list[dict] = []
    for v in state.annotated_variants[:10]:
        var: dict = {
            "variationDescriptor": {
                "id": v.variant_id,
                "label": v.hgvs_p or v.variant_id,
                "geneContext": {"symbol": v.gene or ""},
                "vcfRecord": {
                    "genomeAssembly": (
                        state.vcf_context.genome_build
                        if state.vcf_context else "unknown"
                    ),
                    "chrom": v.variant_id.split(":")[0] if ":" in v.variant_id else "",
                    "pos": v.variant_id.split(":")[1] if v.variant_id.count(":") >= 1 else "",
                    "ref": v.variant_id.split(":")[2] if v.variant_id.count(":") >= 2 else "",
                    "alt": v.variant_id.split(":")[3] if v.variant_id.count(":") >= 3 else "",
                },
            }
        }
        variants.append(var)

    phenopacket = {
        "id": state.case_id,
        "subject": subject,
        "phenotypicFeatures": phenotypic_features,
        "interpretations": interpretations,
        "metaData": {
            "created": now,
            "createdBy": "GeneResolver v2",
            "resources": [
                {
                    "id": "hp",
                    "name": "human phenotype ontology",
                    "url": "http://purl.obolibrary.org/obo/hp.owl",
                    "version": "unknown",
                    "namespacePrefix": "HP",
                    "iriPrefix": "http://purl.obolibrary.org/obo/HP_",
                }
            ],
            "phenopacketSchemaVersion": "2.0",
        },
    }

    path.write_text(
        json.dumps(phenopacket, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Phenopacket export → %s", path)


def _acmg_to_ppack_status(classification: str) -> str:
    mapping = {
        "pathogenic": "CAUSATIVE",
        "likely_pathogenic": "CONTRIBUTORY",
        "vus": "UNCERTAIN",
        "likely_benign": "BENIGN",
        "benign": "BENIGN",
        "not_classifiable": "UNKNOWN",
    }
    return mapping.get(classification, "UNKNOWN")
