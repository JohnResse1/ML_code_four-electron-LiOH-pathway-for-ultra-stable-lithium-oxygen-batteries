#!/usr/bin/env python3
"""Fail-fast consistency checks for the Nature checklist data release."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reproducibility import CONFIG, DESCRIPTOR_COLUMNS, build_model_input, make_group_splits  # noqa: E402


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    data = ROOT / "data"
    raw = pd.read_csv(data / "raw_dataset.csv")
    processed = pd.read_csv(data / "processed_dataset.csv")
    matrix = pd.read_csv(data / "model_input_matrix.csv")
    mapping = pd.read_csv(data / "source_mapping.csv")
    assignments = pd.read_csv(data / "split_assignments.csv")
    dictionary = pd.read_csv(data / "data_dictionary.csv")
    manifest = json.loads((data / "release_manifest.json").read_text())

    id_sets = [set(x["record_id"]) for x in (raw, processed, matrix, mapping)]
    require(all(ids == id_sets[0] for ids in id_sets), "record_id mismatch among released tables")
    require(raw["record_id"].is_unique and len(raw) == 200, "record_id values are not 200 unique rows")
    require(assignments.groupby(["fold", "record_id"]).size().eq(1).all(), "Each record must have one assignment per fold")
    require(assignments.groupby("record_id").size().eq(CONFIG["cross_validation"]["n_splits"]).all(), "Missing fold assignment")
    require(assignments[assignments.assignment == "test"].groupby("record_id").size().eq(1).all(), "Each record must be test exactly once")
    require(assignments.groupby(["fold", "data_source"])["assignment"].nunique().eq(1).all(), "A source group crosses train/test within a fold")

    regenerated = []
    splits = make_group_splits(processed[CONFIG["cross_validation"]["group_column"]])
    for fold, (train_idx, test_idx) in enumerate(splits):
        for label, idx in (("train", train_idx), ("test", test_idx)):
            regenerated.extend((fold, rid, label) for rid in processed.iloc[idx].record_id)
    saved = set(assignments[["fold", "record_id", "assignment"]].itertuples(index=False, name=None))
    require(saved == set(regenerated), "Saved split cannot be reproduced from configured seed")

    require(len(DESCRIPTOR_COLUMNS) == 47 and set(DESCRIPTOR_COLUMNS) <= set(processed), "Descriptor list is not exactly 47 released columns")
    x = build_model_input(pd.read_csv(ROOT / "dataset.csv"))
    forbidden = {"DOI", "data_source", "Compounds(M1-M2-N)", "Original_Compounds", "record_id", "Y1", "Y2", "Overpotential(V)", "Cycling_performance"}
    forbidden.update(CONFIG["experimental_context_columns"])
    require(not (forbidden & set(x.columns)), f"Forbidden model inputs: {forbidden & set(x.columns)}")
    require(dictionary[dictionary.role == "descriptor"].variable.nunique() == 47, "Dictionary does not document 47 descriptors")
    require({"Y1", "Y2"} <= set(dictionary.variable), "Targets absent from dictionary")
    require(processed["Y1"].notna().all(), "Y1 contains missing values")
    require(processed["Y2"].notna().all(), "Y2 contains unexpected missing values")

    doi_re = re.compile(r"^https?://(?:dx\.)?doi\.org/10\.\d{4,9}/\S+$", re.I)
    doi_token_re = re.compile(r"10\.\d{4,9}/[^\s?#]+", re.I)
    require(raw.DOI.map(lambda value: bool(doi_token_re.search(str(value)))).all(), "Raw source URL without an embedded DOI found")
    require(mapping.DOI.map(lambda value: bool(doi_re.match(str(value)))).all(), "Canonical DOI mapping is invalid")
    source_hash = hashlib.sha256((ROOT / CONFIG["raw_source"]).read_bytes()).hexdigest()
    require(source_hash == manifest["source_sha256"], "dataset.csv changed after release generation")

    report = {
        "N records": len(raw),
        "N unique record_id": raw.record_id.nunique(),
        "N literature DOI": raw.DOI.nunique(),
        "N descriptors": len(DESCRIPTOR_COLUMNS),
        "N internal one-hot columns": x.shape[1] - len(DESCRIPTOR_COLUMNS),
        "N total model-input columns": x.shape[1],
        "Y1": "charge-discharge overpotential [V]",
        "Y2": "cycling performance [cycles] (no missing values)",
        "CV strategy": CONFIG["cross_validation"]["strategy"],
        "N folds": CONFIG["cross_validation"]["n_splits"],
        "DOI included in X": "DOI" in x.columns,
        "Experimental context included in X": bool(set(CONFIG["experimental_context_columns"]) & set(x.columns)),
        "Missing record mapping": len(id_sets[0] - set(mapping.record_id)),
        "Missing split mapping": len(id_sets[0] - set(assignments.record_id)),
        "Split reproducible": True,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
