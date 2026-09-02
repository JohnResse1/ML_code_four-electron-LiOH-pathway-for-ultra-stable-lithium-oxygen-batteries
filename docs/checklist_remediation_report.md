# Nature Communications data/code checklist remediation report

## Audit finding

`dataset.csv` is the only primary table in the original repository. It contains
200 rows and 57 columns: four source/material metadata fields, four experimental
context fields, Y1, 47 material descriptors, and Y2. Each row contains a source
URL with an embedded DOI; these represent 150 unique literature DOIs.

The original notebook did not use an explicit descriptor list. Its exclusion-list
logic admitted the DOI column and the four experimental-context fields. DOI was
coerced to an all-zero pseudo-feature. With the current dataset, that code creates
52 numeric candidate columns plus 34 M1/M2 indicator columns (86 total). The
committed legacy summary instead reports 85 total features, demonstrating that
the saved legacy artifacts and current dataset/notebook do not have fully matching
provenance.

## Corrected data model

- Identifiers/metadata: `record_id`, DOI, `data_source`, normalized and original
  compound labels.
- Experimental context, retained but excluded from X: two current-density and two
  limited-specific-capacity fields.
- Y1: `Overpotential(V)`, in V.
- Y2: `Cycling_performance`, in cycles.
- Scientific descriptors: the 47 explicit columns declared in
  `config/reproducibility.yaml`.
- Internal encoding: 34 data-dependent M1/M2 one-hot columns. These are model
  input dimensions, not additional scientific descriptors.

## Corrected reproduction result

The complete eight-model workflow was run with the released seed-42 five-fold
group assignments. It used 81 input columns (47 descriptors plus 34 indicators),
with no DOI or experimental-context feature. Random Forest ranked first with OOF
R2 = 0.838748. The legacy saved result ranked XGBoost first with OOF R2 = 0.884089.
All legacy files remain unchanged under `results/analysis/`; corrected metrics,
predictions and the fitted Random Forest are under
`results/checklist_reproduction/`.

The model-specific corrected-minus-legacy OOF changes are recorded in
`results/checklist_reproduction/legacy_vs_corrected_metrics.csv`. The largest
change among the eight models is XGBoost: -0.051923. The changes reflect exclusion
of four experimental-context predictors as well as removal of the constant DOI
pseudo-feature; a constant zero DOI alone cannot explain the change in tree-model
performance.

## Items requiring author/manuscript confirmation

The repository alone cannot establish the following details, so they are not
invented:

1. The exact voltage-gap convention represented by Y1.
2. The cycle-completion/end-of-life criterion represented by Y2.
3. Original reference tables, temperature/state conventions, and citation for the
   elemental property values.
4. The exact formula and unit of
   `Correction_of_d_electron_number_of_M1_with_sum_atomic_radius_of_M1_and_M2`.
5. The relationship between the two repeated current-density/capacity pairs.
6. The authoritative mapping from repository output filenames to final manuscript
   Fig. 1–2 and Supplementary Fig. 1–7 panel labels.
7. A Zenodo DOI for the corrected release; none is present in the repository.

All such fields are marked `TO BE VERIFIED FROM MANUSCRIPT` or otherwise disclosed
in `data/data_dictionary.csv`, `README.md`, and
`docs/figure_reproducibility.md`.
