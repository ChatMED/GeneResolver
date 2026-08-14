from __future__ import annotations

import logging

from generesolver import config
from generesolver.integrations import deid_roberta, privacy_filter, stanford_deid
from generesolver.schemas.case import CaseState
from generesolver.schemas.clinical import EntityType
from generesolver.schemas.report import ProvenanceItem

logger = logging.getLogger(__name__)

_BACKENDS = {
    "deid_roberta": (deid_roberta, "obi/deid_roberta_i2b2"),
    "stanford": (stanford_deid, "StanfordAIMI/stanford-deidentifier-base"),
    "openmed":  (privacy_filter, "OpenMed/privacy-filter-multilingual"),
}


def _select_backend() -> tuple[object, str]:
    name = getattr(config, "DEID_BACKEND", "deid_roberta")
    order = [name] + [k for k in _BACKENDS if k != name]

    for key in order:
        module, model_id = _BACKENDS[key]
        if module.is_available():
            if key != name:
                logger.warning(
                    "De-id backend '%s' unavailable; fell back to '%s'.", name, key
                )
            return module, model_id

    module, model_id = _BACKENDS[name]
    return module, model_id

PROTECTED_TERMS = [
    "Angelman syndrome",
    "Williams syndrome",
    "Noonan syndrome",
    "Down syndrome",
    "Turner syndrome",
    "Marfan syndrome",
    "Rett syndrome",
    "Dravet syndrome",
    "West syndrome",
    "Lennox-Gastaut syndrome",
    "Guillain-Barre syndrome",
    "Guillain-Barré syndrome",
    "Ehlers-Danlos syndrome",
    "Gorlin syndrome",

    "Prader-Willi syndrome",
    "Patau syndrome",
    "Edwards syndrome",
    "DiGeorge syndrome",
    "22q11.2 deletion syndrome",
    "Fragile X syndrome",
    "Klinefelter syndrome",
    "Smith-Magenis syndrome",
    "Smith-Lemli-Opitz syndrome",
    "Cornelia de Lange syndrome",
    "Rubinstein-Taybi syndrome",
    "Sotos syndrome",
    "Kabuki syndrome",
    "CHARGE syndrome",
    "Aicardi syndrome",
    "Joubert syndrome",
    "Dandy-Walker syndrome",
    "Walker-Warburg syndrome",
    "Miller-Dieker syndrome",
    "Sturge-Weber syndrome",
    "Tuberous sclerosis complex",
    "Neurofibromatosis type 1",
    "Neurofibromatosis type 2",
    "Von Hippel-Lindau syndrome",
    "Beckwith-Wiedemann syndrome",
    "Silver-Russell syndrome",
    "Russell-Silver syndrome",
    "Alport syndrome",
    "Bardet-Biedl syndrome",
    "Laurence-Moon syndrome",
    "Meckel-Gruber syndrome",
    "Ellis-van Creveld syndrome",
    "McCune-Albright syndrome",
    "Waardenburg syndrome",
    "Treacher Collins syndrome",
    "Pierre Robin sequence",
    "Goldenhar syndrome",
    "Moebius syndrome",
    "Möbius syndrome",
    "Crouzon syndrome",
    "Apert syndrome",
    "Pfeiffer syndrome",
    "Saethre-Chotzen syndrome",
    "Netherton syndrome",
    "Proteus syndrome",
    "Cowden syndrome",
    "Peutz-Jeghers syndrome",
    "Lynch syndrome",
    "Li-Fraumeni syndrome",
    "Wiskott-Aldrich syndrome",
    "Fanconi anemia",
    "Bloom syndrome",
    "Ataxia-telangiectasia",
    "Nijmegen breakage syndrome",
    "Cockayne syndrome",
    "Werner syndrome",
    "Rothmund-Thomson syndrome",

    "Landau-Kleffner syndrome",
    "Ohtahara syndrome",
    "Doose syndrome",
    "Rasmussen encephalitis",
    "Rasmussen syndrome",
    "MELAS syndrome",
    "MERRF syndrome",
    "Leigh syndrome",
    "Kearns-Sayre syndrome",
    "Charcot-Marie-Tooth disease",
    "Friedreich ataxia",
    "Huntington disease",
    "Wilson disease",
    "Alexander disease",
    "Canavan disease",
    "Krabbe disease",
    "Niemann-Pick disease",
    "Tay-Sachs disease",
    "Sandhoff disease",
    "Gaucher disease",
    "Fabry disease",
    "Pompe disease",
    "Metachromatic leukodystrophy",
    "Adrenoleukodystrophy",
    "Pelizaeus-Merzbacher disease",
    "Hereditary spastic paraplegia",
    "Duchenne muscular dystrophy",
    "Becker muscular dystrophy",
    "Myotonic dystrophy",
    "Spinal muscular atrophy",

    "Miller Fisher syndrome",
    "Bickerstaff brainstem encephalitis",
    "Lambert-Eaton myasthenic syndrome",
    "Isaacs syndrome",
    "Stiff person syndrome",
    "Opsoclonus-myoclonus syndrome",
    "Susac syndrome",
    "Tolosa-Hunt syndrome",
    "Brown-Sequard syndrome",
    "Brown-Séquard syndrome",
    "Wallenberg syndrome",
    "Horner syndrome",
    "Parsonage-Turner syndrome",

    "Lesch-Nyhan syndrome",
    "Zellweger syndrome",
    "Refsum disease",
    "Maple syrup urine disease",
    "Phenylketonuria",
    "Homocystinuria",
    "Menkes disease",
    "Hartnup disease",
    "Hurler syndrome",
    "Hunter syndrome",
    "Sanfilippo syndrome",
    "Morquio syndrome",
    "Maroteaux-Lamy syndrome",
    "Sly syndrome",
    "Lowe syndrome",
    "Alagille syndrome",
    "Birt-Hogg-Dube syndrome",
    "Birt-Hogg-Dubé syndrome",

    "Loeys-Dietz syndrome",
    "Vascular Ehlers-Danlos syndrome",
    "MASS syndrome",
    "Stickler syndrome",
    "Osteogenesis imperfecta",
    "Achondroplasia",
    "Thanatophoric dysplasia",
    "Holt-Oram syndrome",
    "Long QT syndrome",
    "Brugada syndrome",
    "Wolff-Parkinson-White syndrome",

    "Klippel-Feil syndrome",
    "Klippel-Trenaunay syndrome",
    "Poland syndrome",
    "VACTERL association",
    "VATER association",
    "Mowat-Wilson syndrome",
    "Bohring-Opitz syndrome",
    "Coffin-Siris syndrome",
    "Coffin-Lowry syndrome",
    "Pitt-Hopkins syndrome",
    "Koolen-de Vries syndrome",
    "Kleefstra syndrome",
    "KBG syndrome",
    "Wiedemann-Steiner syndrome",
    "Nicolaides-Baraitser syndrome",
    "Temple-Baraitser syndrome",
    "Ohdo syndrome",
    "Kabuki syndrome",

    "Usher syndrome",
    "Alstrom syndrome",
    "Alström syndrome",
    "Senior-Loken syndrome",
    "Senior-Løken syndrome",
    "Joubert syndrome with oculorenal defect",
    "Bardet-Biedl syndrome",
    "Branchio-oto-renal syndrome",
    "BOR syndrome",
    "Pendred syndrome",

    "Addison disease",
    "Graves disease",
    "Hashimoto thyroiditis",
    "Cushing syndrome",
    "Sjogren syndrome",
    "Sjögren syndrome",
    "Behcet disease",
    "Behçet disease",
    "Kawasaki disease",
    "Takayasu arteritis",
    "Goodpasture syndrome",
    "Churg-Strauss syndrome",
    "Erdheim-Chester disease",
    "Rosai-Dorfman disease",
    "Castleman disease",
]


def protect_terms(text: str, terms: list[str]) -> tuple[str, dict[str, str]]:
    mapping: dict[str, str] = {}
    protected = text
    for i, term in enumerate(terms):
        placeholder = f"__MEDICAL_TERM_{i}__"
        mapping[placeholder] = term
        protected = protected.replace(term, placeholder)
    return protected, mapping


def restore_terms(text: str, mapping: dict[str, str]) -> str:
    restored = text
    for placeholder, term in mapping.items():
        restored = restored.replace(placeholder, term)
    return restored


def run(state: CaseState) -> CaseState:
    if not state.input.clinical_note:
        state.clinical_note_deidentified = ""
        state.deidentification_applied = True
        state.mark_done("deidentifier")
        return state

    backend, model_id = _select_backend()

    protected, mapping = protect_terms(state.input.clinical_note, PROTECTED_TERMS)

    deidentified, applied = backend.deidentify_text(protected)

    deidentified = restore_terms(deidentified, mapping)

    state.clinical_note_deidentified = deidentified
    state.deidentification_applied = applied
    if applied and state.family_history:
        fh_protected, fh_map = protect_terms(state.family_history, PROTECTED_TERMS)
        fh_clean, fh_applied = backend.deidentify_text(fh_protected)
        if fh_applied:
            state.family_history = restore_terms(fh_clean, fh_map)
        else:
            state.family_history = None
            state.add_warning(
                "Family-history text could not be de-identified and was dropped."
            )

    if applied:
        for ent in state.clinical_entities:
            if ent.entity_type == EntityType.FAMILY_HISTORY and ent.text:
                prot, emap = protect_terms(ent.text, PROTECTED_TERMS)
                clean, ok = backend.deidentify_text(prot)
                ent.text = restore_terms(clean, emap) if ok else "[REDACTED]"

    if not applied:
        state.add_warning(
            "Privacy filter could not run — clinical note may contain PII. "
            "External API calls (Claude HPO mapper) will use the original text. "
            "De-identify the clinical note manually before running in production."
        )
        state.add_safety_flag(
            "PII filter failed — patient identifiers may have been sent to external APIs."
        )
    else:
        logger.info(
            "Clinical note deidentified (%d eponym phrase(s) protected).",
            sum(1 for p in mapping if p in protected),
        )

    state.provenance.append(ProvenanceItem(
        component="deidentifier",
        tool_name=f"{model_id} + eponym protection",
        tool_version=f"{len(PROTECTED_TERMS)} protected medical eponyms",
    ))
    state.mark_done("deidentifier")
    return state
