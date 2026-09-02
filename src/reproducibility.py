"""Canonical data and split construction for the checklist release."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "reproducibility.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    # JSON is a strict subset of YAML; this keeps the YAML configuration readable
    # without introducing an additional parser dependency.
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


CONFIG = load_config()
DESCRIPTOR_COLUMNS = CONFIG["descriptor_columns"]
METADATA_COLUMNS = CONFIG["metadata_columns"]
EXPERIMENTAL_CONTEXT_COLUMNS = CONFIG["experimental_context_columns"]
Y1_SOURCE = CONFIG["targets"]["Y1"]["source_column"]
Y2_SOURCE = CONFIG["targets"]["Y2"]["source_column"]


def add_record_ids(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    expected = [CONFIG["record_id_format"] % i for i in range(1, len(out) + 1)]
    if "record_id" in out:
        if out["record_id"].tolist() != expected:
            raise ValueError("Existing record_id values do not match the canonical row-order IDs")
        return out
    out.insert(0, "record_id", expected)
    return out


def extract_m1_m2(value: object) -> tuple[str, str]:
    parts = [part.strip() for part in str(value).replace(" ", "").split("-")]
    m1 = parts[0] if parts and parts[0] else "UNK"
    m2 = parts[1] if len(parts) > 1 and parts[1] else "UNK"
    return m1, m2


def processed_descriptors(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(DESCRIPTOR_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing configured descriptor columns: {missing}")
    return frame[DESCRIPTOR_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def build_model_input(frame: pd.DataFrame) -> pd.DataFrame:
    descriptors = processed_descriptors(frame).reset_index(drop=True)
    parsed = frame["Compounds(M1-M2-N)"].map(extract_m1_m2)
    m1 = pd.get_dummies(pd.Series([x[0] for x in parsed], name="M1"), prefix="m1", dtype=int)
    m2 = pd.get_dummies(pd.Series([x[1] for x in parsed], name="M2"), prefix="m2", dtype=int)
    return pd.concat([descriptors, m1, m2], axis=1)


def make_group_splits(groups, n_splits: int | None = None, seed: int | None = None):
    cv = CONFIG["cross_validation"]
    n_splits = cv["n_splits"] if n_splits is None else n_splits
    seed = cv["random_seed"] if seed is None else seed
    groups = np.asarray(groups).astype(str)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two groups are required")
    n_splits = min(n_splits, len(unique_groups))
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    validation_groups = np.array_split(shuffled, n_splits)
    indices = np.arange(len(groups))
    return [
        (indices[~np.isin(groups, fold_groups)], indices[np.isin(groups, fold_groups)])
        for fold_groups in validation_groups
    ]
