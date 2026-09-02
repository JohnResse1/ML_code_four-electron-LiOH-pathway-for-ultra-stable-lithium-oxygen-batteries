# Machine Learning-Driven Regulation of the Four-Electron LiOH Pathway for Ultra-Stable Lithium–Oxygen Batteries

This repository contains the machine-learning dataset, source-level provenance,
deterministic cross-validation assignments, analysis code and preserved outputs
for predicting Li–O2-battery overpotential in M1–M2–N catalyst records.

## Repository structure

```text
.
├── analysis.ipynb                 End-to-end analysis and figure notebook
├── dataset.csv                    Original repository dataset (unchanged)
├── config/reproducibility.yaml    Canonical targets, 47 descriptors and CV settings
├── data/
│   ├── raw_dataset.csv            Original values plus stable record_id and raw targets
│   ├── processed_dataset.csv      Numeric 47-descriptor table used for ML construction
│   ├── model_input_matrix.csv     47 descriptors plus internal M1/M2 one-hot columns
│   ├── source_mapping.csv         Row-to-literature DOI mapping
│   ├── data_dictionary.csv        Definitions, units, roles and processing rules
│   ├── split_assignments.csv      All train/test assignments for all five folds
│   └── release_manifest.json      Counts, source hash and release validation metadata
├── src/reproducibility.py         Single implementation of preprocessing and splitting
├── scripts/
│   ├── generate_data_release.py   Deterministically regenerates data/*
│   ├── validate_reproducibility.py Automated integrity checks
│   ├── reproduce_ml.py            Corrected eight-model reproduction
│   └── export_comprehensive_figure_data.py  Exports panel (a)–(f) CSVs
├── docs/figure_reproducibility.md Analysis-to-output traceability map
├── results/analysis/              Preserved legacy/publication outputs
└── results/checklist_reproduction Corrected 47-descriptor reproduction outputs
```

The exact source data for panels (a)–(f) of the comprehensive 3×2 overview are
provided under `results/checklist_reproduction/comprehensive_3x2_source_data/`.
They can be regenerated with `python scripts/export_comprehensive_figure_data.py`.

## Data availability and reproducibility

### Records and literature sources

The release contains 200 extracted records. Stable identifiers `R0001` through
`R0200` are assigned in the immutable row order of `dataset.csv`. Every record has
a literature identifier; 150 unique DOIs are represented, and repeated DOIs are
expected when a paper contributes multiple extracted records.

- Raw extracted values: `data/raw_dataset.csv`
- Machine-readable processed values: `data/processed_dataset.csv`
- Row-level DOI and material mapping: `data/source_mapping.csv`
- Definitions, units and provenance: `data/data_dictionary.csv`

Three original DOI fields were publisher/supplement URLs rather than canonical
`doi.org` URLs. `raw_dataset.csv` preserves these verbatim. The processed and
mapping tables additionally provide canonical DOI URLs extracted from them.

### Targets

- **Y1 — overpotential (V):** the reported charge–discharge overpotential for the
  Li–O2 battery record, stored originally as `Overpotential(V)`. The repository
  does not contain the manuscript definition of the exact voltage-gap convention;
  that convention is explicitly marked `TO BE VERIFIED FROM MANUSCRIPT` in the
  data dictionary rather than inferred.
- **Y2 — cycling performance (cycles):** the reported number of completed cycles
  under the source-specific testing conditions, stored originally as
  `Cycling_performance`. The exact end-of-life criterion must be verified against
  the manuscript/source extraction protocol.

The four current-density/limited-capacity columns are experimental-context
metadata. They are retained in raw and processed releases, but are not part of the
47 material descriptors and do not enter the corrected model.

### The 47 descriptors

The descriptor set is the 47 consecutive material-property and derived columns
from `Atomic_Weight_of_M1` through `d-band_center_of_M2`. It is declared explicitly
in `config/reproducibility.yaml` and imported by both the notebook and scripts.
Definitions, units, derivations and all cases needing manuscript verification are
listed in `data/data_dictionary.csv`.

The phrase “47 descriptors” refers to these scientific descriptor variables,
not the dimension after encoding. The model internally adds 34 composition
indicator columns (18 M1 and 16 M2), giving 81 model-input columns for the current
dataset. One-hot columns are exposed in `data/model_input_matrix.csv` but are not
renamed or counted as scientific descriptors.

### Preprocessing

The deterministic processing chain is:

1. Read `dataset.csv` and assign `record_id` from stable source row order.
2. Select only the explicit 47-column descriptor list.
3. Convert descriptor values with `pandas.to_numeric(errors='coerce')`.
4. Treat `none`, empty and other non-numeric tokens as missing, then fill with
   `0.0`, matching the original notebook's model preprocessing.
5. Parse M1 and M2 from `Compounds(M1-M2-N)` and append separate one-hot columns.
6. Apply `StandardScaler` only inside the ElasticNet, Lasso, Ridge and SVR
   pipelines; tree models receive the numeric matrix directly.

No DOI, source ID, record ID, compound-name string, target or experimental-context
column enters the corrected model matrix.

### Cross-validation assignments

The study uses group-aware cross-validation rather than a single fixed
train/validation/test partition. Groups are defined by `data_source`; all records
from the same source stay together within each fold. Unique groups are shuffled
with NumPy seed **42**, divided deterministically into **five folds**, and each
record is test once and train in the other four folds. There is no separately
labelled validation partition: hyperparameter search uses the same five folds.

All 1,000 row/fold assignments are provided in `data/split_assignments.csv`.
This exactly records the implemented historical strategy. Because tuning and
reported OOF evaluation use the same folds, these scores are not nested-CV
estimates; this methodological limitation is disclosed rather than hidden.

### Reproduce the release and machine-learning analysis

Python 3.12 and the pinned packages in `requirements.txt` are recommended.

```bash
python -m pip install -r requirements.txt
python scripts/generate_data_release.py
python scripts/validate_reproducibility.py
python scripts/reproduce_ml.py
```

The final command runs the complete corrected eight-model grid-search workflow
and writes new metrics, row-level predictions and the selected model under
`results/checklist_reproduction/`. It does not overwrite `results/analysis/`.
For figure-level inputs and notebook sections, see
`docs/figure_reproducibility.md`.

### Correction and legacy-result policy

The original notebook selected every column not present in an exclusion list.
This accidentally admitted `DOI`; numeric coercion converted it to an all-zero
pseudo-feature. It also admitted the four experimental-context columns. The
corrected implementation uses an explicit list of 47 descriptors and excludes all
metadata and targets.

Original saved results remain under `results/analysis/`. Corrected results and a
model-by-model comparison with the saved legacy metrics are written separately to
`results/checklist_reproduction/legacy_vs_corrected_metrics.csv`. The old summary
reports 85 features, whereas the old code applied to the current dataset generates
86; this provenance inconsistency is documented and not silently rewritten.

### Repository and archival identifier

- GitHub: https://github.com/JohnResse1/ML_code_four-electron-LiOH-pathway-for-ultra-stable-lithium-oxygen-batteries
- Zenodo DOI: **NOT YET AVAILABLE — must be added after depositing this corrected release.**

No Zenodo identifier is present in the repository, so one has not been invented.

## Model outputs

Eight regressors are evaluated: XGBoost, Gradient Boosting, Random Forest, Extra
Trees, SVR, ElasticNet, Lasso and Ridge. SHAP analysis is used for model-level
interpretation. Correlated derived descriptors mean that SHAP rankings should not
be interpreted as independent causal effects.

## License

This project is licensed under the MIT License; see `LICENSE`.
