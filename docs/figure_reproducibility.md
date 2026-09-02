# Figure reproducibility map

The repository does not contain the manuscript captions or an authoritative
manuscript-panel-to-file crosswalk. The mapping below therefore identifies every
analysis/figure-producing section and its exact inputs without inventing figure
numbers. The authors must verify which repository artifact corresponds to each
final typeset panel of Figs. 1–2 and Supplementary Figs. 1–7.

| Analysis / output artifact | Code | Released data input | Model or split dependency |
|---|---|---|---|
| Overpotential versus cycling scatter | `analysis.ipynb`, section 2 | `data/processed_dataset.csv`: Y1, Y2 | None |
| Pearson descriptor heat map | `analysis.ipynb`, section 3 | 47 columns in `data/processed_dataset.csv` and Y1 | None |
| Eight-model comparison and parity plots | `analysis.ipynb`, section 4; `scripts/reproduce_ml.py` | `data/model_input_matrix.csv` | Seed 42 assignments in `data/split_assignments.csv` |
| Ten-seed robustness box plot | `analysis.ipynb`, section 4 | Same 47 descriptors plus internal M1/M2 encoding | Group-aware five-fold splits regenerated for seeds 0–9 |
| SHAP summary | `analysis.ipynb`, section 5 | Same model input matrix | Full-data refit of the selected XGBoost configuration |
| Dataset overview, correlations, co-occurrence and t-SNE | `analysis.ipynb`, section 6 | `data/processed_dataset.csv`; composition metadata | t-SNE seed 42; no supervised split |
| Co/Ni category plots | `analysis.ipynb`, section 7 | Composition metadata and fitted-model predictions | Full-data refit; these are in-sample fitted predictions, not OOF estimates |

The exact plot-ready CSVs for all six panels of
`comprehensive_3x2_overview_curated` are in
`results/checklist_reproduction/comprehensive_3x2_source_data/`. Panel (e) has
two files because its top 15 features are selected by absolute Spearman rho but
the displayed heat map contains Pearson correlations.

## Traceability caveat

The files originally committed under `results/analysis/` are preserved as the
legacy publication outputs. Their saved summary says 85 model features. Applying
the old notebook feature-selection code to the current `dataset.csv` produces 86
columns (52 automatically selected numeric candidates plus 34 one-hot columns),
so the committed artifacts are not fully provenance-consistent with the current
input file. Corrected outputs are written to `results/checklist_reproduction/`
and never overwrite the legacy files.

## Reproduction commands

```bash
python scripts/generate_data_release.py
python scripts/validate_reproducibility.py
python scripts/reproduce_ml.py
```

To regenerate every exploratory plot, install `requirements.txt`, open
`analysis.ipynb`, and run all cells from a clean kernel. Its corrected outputs go
to `results/checklist_reproduction/notebook_outputs/`.
