# Machine-learning analysis for the Li–O₂ battery catalyst dataset

This directory contains the analysis code and data used for the machine-learning
results reported in the manuscript *Machine learning-driven regulation of the
four-electron LiOH pathway for lithium–oxygen batteries with prolonged
stability*.

## Contents

```text
analysis.ipynb       Main code
dataset.csv          200 catalyst records and their metadata/features
requirements.txt     Pinned Python dependencies
results/analysis/    Re-generated figures, tables, predictions and models
```

## Dataset and model inputs

The dataset contains 200 records. `DOI` is retained as literature metadata and
is not used as a model predictor. `data_source` is also excluded from the model
matrix, but is retained as the grouping variable for the group-wise
cross-validation. Records sharing the same `data_source` remain in the same
fold.

The workflow selects 51 numeric candidate columns after
excluding metadata and targets. These comprise the 47 material descriptors and
4 experimental-context variables present in the original input table. M1 and
M2 are additionally encoded using 34 one-hot composition indicators, giving 85
model-input columns in total.

Missing/non-numeric values are converted with `pandas.to_numeric`
(`errors='coerce'`) and filled with zero. The target is `Overpotential(V)` and
the grouping variable is `data_source`.

## Machine-learning workflow

Eight regression models are trained and tuned using grid search:

- XGBoost
- Gradient Boosting
- Random Forest
- ExtraTrees
- SVR
- ElasticNet
- Lasso
- Ridge

The notebook uses five group-wise folds with the fixed seed 42. It also evaluates
the selected XGBoost model under ten alternative random seeds, with five folds
per seed. SHAP analysis ranks features using mean absolute SHAP values.

The manuscript-reported model is XGBoost. The historical five-fold result is
approximately `R² = 0.8815 ± 0.0346`; exact floating-point values can vary
slightly with parallel tree-training and package/runtime details.

## Reproduction

Use the pinned `ML` environment or install the packages listed in
`requirements.txt`. From this directory, run:

```bash
conda activate ML
jupyter nbconvert --execute --to notebook --inplace analysis.ipynb
```

The notebook writes all generated outputs to `results/analysis/`, including the
model comparison, parity plots, repeated-validation results, SHAP tables/plots,
the final XGBoost model, and the Co/Ni screening outputs.

## License

This project is licensed under the MIT License; see `LICENSE`.
