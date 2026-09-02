#!/usr/bin/env python3
"""Generate the checklist data package deterministically from dataset.csv."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reproducibility import (  # noqa: E402
    CONFIG,
    DESCRIPTOR_COLUMNS,
    EXPERIMENTAL_CONTEXT_COLUMNS,
    METADATA_COLUMNS,
    Y1_SOURCE,
    Y2_SOURCE,
    add_record_ids,
    build_model_input,
    make_group_splits,
    processed_descriptors,
)

DATA_DIR = ROOT / "data"


def normalize_doi(value: str) -> str:
    """Extract and canonicalize a DOI embedded in a DOI or publisher URL."""
    match = re.search(r"10\.\d{4,9}/[^\s?#]+", str(value), flags=re.I)
    if not match:
        return ""
    suffix = match.group(0).rstrip("./")
    return "https://doi.org/" + suffix


def descriptor_documentation(variable: str) -> tuple[str, str, str, str]:
    """Return conservative documentation derived from column semantics.

    The repository does not contain the manuscript methods/source table. Entries
    whose precise convention cannot be proven are explicitly flagged.
    """
    element = variable.rsplit("_of_", 1)[-1] if "_of_" in variable else ""
    if variable.startswith("Atomic_Weight"):
        return (f"Standard atomic weight assigned to {element}.", "u", "Element-property lookup used by the original study", "Numeric; missing values replaced by 0. Exact lookup source TO BE VERIFIED FROM MANUSCRIPT.")
    if variable.startswith("Covalent_Radius"):
        return (f"Covalent radius assigned to {element}.", "pm", "Element-property lookup used by the original study", "Numeric; missing values replaced by 0. Radius convention/source TO BE VERIFIED FROM MANUSCRIPT.")
    if variable.startswith("VanDer_Waals_Radius"):
        return (f"van der Waals radius assigned to {element}.", "pm", "Element-property lookup used by the original study", "Numeric; missing values replaced by 0. Radius convention/source TO BE VERIFIED FROM MANUSCRIPT.")
    if variable.startswith("Atomic_Volume"):
        return (f"Atomic volume assigned to {element}.", "cm^3 mol^-1", "Element-property lookup used by the original study", "Numeric; missing values replaced by 0. Unit/source TO BE VERIFIED FROM MANUSCRIPT.")
    if variable.startswith("Electron_Affinity"):
        return (f"Electron affinity assigned to {element}.", "kJ mol^-1", "Element-property lookup used by the original study", "Numeric; missing values replaced by 0. Sign convention/source TO BE VERIFIED FROM MANUSCRIPT.")
    if variable.startswith("d-electron_number"):
        return (f"d-electron count assigned to {element}.", "dimensionless", "Element-property lookup used by the original study", "Numeric; missing values replaced by 0. Electronic-state convention TO BE VERIFIED FROM MANUSCRIPT.")
    if variable == "p-electron_number_of_N":
        return ("p-electron count assigned to the non-metal/site element N.", "dimensionless", "Element-property lookup used by the original study", "Numeric; missing values replaced by 0. Electronic-state convention TO BE VERIFIED FROM MANUSCRIPT.")
    if variable.startswith("Number_of_valence_electrons"):
        return (f"Valence-electron count assigned to {element}.", "dimensionless", "Element-property lookup used by the original study", "Numeric; missing values replaced by 0. Valence convention TO BE VERIFIED FROM MANUSCRIPT.")
    if variable.startswith("First_ionization_energy"):
        return (f"First ionization energy assigned to {element}.", "kJ mol^-1", "Element-property lookup used by the original study", "Numeric; missing values replaced by 0. Lookup source TO BE VERIFIED FROM MANUSCRIPT.")
    if variable.startswith("Electrical_Conductivity"):
        return (f"Electrical conductivity assigned to {element}.", "S m^-1", "Element-property lookup used by the original study", "Numeric; missing values replaced by 0. Temperature/source TO BE VERIFIED FROM MANUSCRIPT.")
    if variable.startswith("Thermal_Conductivity"):
        return (f"Thermal conductivity assigned to {element}.", "W m^-1 K^-1", "Element-property lookup used by the original study", "Numeric; missing values replaced by 0. Temperature/source TO BE VERIFIED FROM MANUSCRIPT.")
    if variable.startswith("Pauling_electronegativity"):
        return (f"Pauling electronegativity assigned to {element}.", "dimensionless", "Element-property lookup used by the original study", "Numeric; missing values replaced by 0.")
    if variable.startswith("d-band_center"):
        return (f"d-band-center value assigned to {element}.", "eV", "Element-property lookup used by the original study", "Numeric; missing values replaced by 0. Energy reference/source TO BE VERIFIED FROM MANUSCRIPT.")
    formulas = {
        "Difference_in_the_d-electron_number_of_M1_and_M2": ("d-electron_number_of_M1 - d-electron_number_of_M2", "dimensionless"),
        "Average_d-electron_number_of_M1_and_M2": ("Arithmetic mean of the M1 and M2 d-electron counts", "dimensionless"),
        "Difference_in_the_valence-electron_number_of_M1_and_M2": ("Number_of_valence_electrons_of_M1 - Number_of_valence_electrons_of_M2", "dimensionless"),
        "Average_valence-electron_number_of_M1_and_M2": ("Arithmetic mean of the M1 and M2 valence-electron counts", "dimensionless"),
        "Difference_in_the_First_ionization_energy_of_metals": ("First_ionization_energy_of_M1 - First_ionization_energy_of_M2", "kJ mol^-1"),
        "Difference_in_the_electronegativity_of_M1_and_M2": ("Pauling_electronegativity_of_M1 - Pauling_electronegativity_of_M2", "dimensionless"),
        "Average_electronegativity_of_M1_and_M2": ("Arithmetic mean of M1 and M2 Pauling electronegativities", "dimensionless"),
        "Average_electronegativity_of_M1_and_N": ("Arithmetic mean of M1 and N Pauling electronegativities", "dimensionless"),
        "Average_electronegativity_of_M2_and_N": ("Arithmetic mean of M2 and N Pauling electronegativities", "dimensionless"),
        "Average_electronegativity_of_active_site_(M1-M2-N)": ("Arithmetic mean of M1, M2 and N Pauling electronegativities", "dimensionless"),
        "Correction_of_d_electron_number_of_M1_with_electronegativity_of_M2": ("d-electron_number_of_M1 multiplied by Pauling_electronegativity_of_M2", "dimensionless"),
        "Correction_of_d_electron_number_of_M1_with_electronegativity_of_N": ("d-electron_number_of_M1 multiplied by Pauling_electronegativity_of_N", "dimensionless"),
        "Correction_of_d_electron_number_of_M1_with_average_Electronegativity_of_M2_and_N": ("d-electron_number_of_M1 multiplied by Average_electronegativity_of_M2_and_N", "dimensionless"),
        "Correction_of_d_electron_number_of_M1_with_sum_atomic_radius_of_M1_and_M2": ("Composite d-electron/radius correction; exact formula TO BE VERIFIED FROM MANUSCRIPT", "TO BE VERIFIED FROM MANUSCRIPT"),
    }
    definition, unit = formulas.get(variable, ("TO BE VERIFIED FROM MANUSCRIPT", "TO BE VERIFIED FROM MANUSCRIPT"))
    return (definition, unit, "Derived descriptor supplied in dataset.csv", "Numeric; missing values replaced by 0. Derivation checked against stored values where unambiguous.")


def build_dictionary() -> pd.DataFrame:
    rows = [
        ("record_id", "identifier", "Stable row identifier assigned in original dataset row order.", "dimensionless", "Generated as R0001-R0200", "No transformation"),
        ("DOI", "identifier", "Canonical Digital Object Identifier URL of the literature source for this extracted record.", "dimensionless", "DOI extracted from the source URL in dataset.csv", "Canonicalized as https://doi.org/<identifier>; never used as a model feature"),
        ("DOI_raw", "metadata", "Source identifier/URL exactly as stored in dataset.csv.", "dimensionless", "dataset.csv", "Preserved unchanged"),
        ("data_source", "metadata", "Original study-specific source/group identifier used for grouped cross-validation.", "dimensionless", "dataset.csv", "String; used only as CV grouping variable"),
        ("Compounds(M1-M2-N)", "metadata", "Normalized M1-M2-N elemental composition label.", "dimensionless", "dataset.csv", "Parsed only to create internal M1/M2 one-hot columns"),
        ("Original_Compounds", "metadata", "Material label transcribed from the literature source.", "dimensionless", "dataset.csv", "Preserved as text; never used as a model feature"),
        ("Y1", "target", "Reported charge-discharge overpotential for the Li-O2 battery record.", "V", "Overpotential(V) in dataset.csv", "Numeric conversion; no imputation required. Precise voltage-gap convention TO BE VERIFIED FROM MANUSCRIPT."),
        ("Y2", "target", "Reported cycling performance: number of completed cycles for the literature record under its reported testing conditions.", "cycles", "Cycling_performance in dataset.csv", "Numeric conversion; no missing values in this release. Precise cycle endpoint criterion TO BE VERIFIED FROM MANUSCRIPT."),
    ]
    context_definitions = {
        "Current_density(mA/g)": ("First reported current-density context field.", "mA g^-1"),
        "Limited_specific_capacity(mAh/g)": ("First reported limited-specific-capacity context field.", "mAh g^-1"),
        "Current_density(mA/g).1": ("Second reported current-density context field.", "mA g^-1"),
        "Limited_specific_capacity(mAh/g).1": ("Second reported limited-specific-capacity context field.", "mAh g^-1"),
    }
    for variable, (definition, unit) in context_definitions.items():
        rows.append((variable, "experimental_context", definition, unit, "dataset.csv", "Preserved in raw/processed release but excluded from the 47-descriptor model. Relationship between the two repeated pairs TO BE VERIFIED FROM MANUSCRIPT."))
    for variable in DESCRIPTOR_COLUMNS:
        definition, unit, source, processing = descriptor_documentation(variable)
        rows.append((variable, "descriptor", definition, unit, source, processing))
    return pd.DataFrame(rows, columns=["variable", "role", "definition", "unit", "source_or_derivation", "processing"])


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(ROOT / CONFIG["raw_source"], dtype=str, keep_default_na=False)
    raw = add_record_ids(original)
    raw.insert(raw.columns.get_loc(Y1_SOURCE) + 1, "Y1_raw", raw[Y1_SOURCE])
    raw.insert(len(raw.columns), "Y2_raw", raw[Y2_SOURCE])
    raw.to_csv(DATA_DIR / "raw_dataset.csv", index=False, encoding="utf-8-sig")

    numeric_descriptors = processed_descriptors(original)
    processed = add_record_ids(original[METADATA_COLUMNS + EXPERIMENTAL_CONTEXT_COLUMNS].copy())
    processed.insert(processed.columns.get_loc("DOI") + 1, "DOI_raw", processed["DOI"])
    processed["DOI"] = processed["DOI_raw"].map(normalize_doi)
    processed["Y1"] = pd.to_numeric(original[Y1_SOURCE], errors="coerce")
    processed["Y2"] = pd.to_numeric(original[Y2_SOURCE], errors="coerce")
    processed = pd.concat([processed, numeric_descriptors], axis=1)
    processed.to_csv(DATA_DIR / "processed_dataset.csv", index=False, encoding="utf-8-sig")

    model_x = build_model_input(original)
    model_matrix = pd.concat([processed[["record_id", "DOI", "data_source"]], model_x, processed[["Y1", "Y2"]]], axis=1)
    model_matrix.to_csv(DATA_DIR / "model_input_matrix.csv", index=False, encoding="utf-8-sig")

    mapping = processed[["record_id", "DOI", "DOI_raw", "data_source", "Original_Compounds", "Compounds(M1-M2-N)"]].copy()
    mapping["source_reference"] = mapping["DOI"]
    mapping["source_title"] = "NOT AVAILABLE IN REPOSITORY"
    mapping.to_csv(DATA_DIR / "source_mapping.csv", index=False, encoding="utf-8-sig")

    split_rows = []
    splits = make_group_splits(processed[CONFIG["cross_validation"]["group_column"]])
    for fold, (train_idx, test_idx) in enumerate(splits):
        for assignment, indices in (("train", train_idx), ("test", test_idx)):
            part = processed.iloc[indices][["record_id", "DOI", "data_source"]].copy()
            part.insert(3, "fold", fold)
            part.insert(4, "assignment", assignment)
            split_rows.append(part)
    pd.concat(split_rows, ignore_index=True).to_csv(DATA_DIR / "split_assignments.csv", index=False, encoding="utf-8-sig")
    build_dictionary().to_csv(DATA_DIR / "data_dictionary.csv", index=False, encoding="utf-8-sig")

    doi_pattern = re.compile(r"^https?://(?:dx\.)?doi\.org/10\.\d{4,9}/\S+$", re.I)
    manifest = {
        "source_sha256": hashlib.sha256((ROOT / CONFIG["raw_source"]).read_bytes()).hexdigest(),
        "n_records": len(processed),
        "n_unique_record_id": processed["record_id"].nunique(),
        "n_unique_doi": processed["DOI"].nunique(),
        "n_valid_doi_rows": int(processed["DOI"].map(lambda x: bool(doi_pattern.match(str(x)))).sum()),
        "n_descriptors": len(DESCRIPTOR_COLUMNS),
        "n_model_input_columns_excluding_targets_and_metadata": model_x.shape[1],
        "n_internal_one_hot_columns": model_x.shape[1] - len(DESCRIPTOR_COLUMNS),
        "cv": CONFIG["cross_validation"],
    }
    (DATA_DIR / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
