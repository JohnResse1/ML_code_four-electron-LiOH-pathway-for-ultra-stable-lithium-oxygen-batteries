# Machine Learning-Driven Regulation of the Four-Electron LiOH Pathway for Ultra-Stable Lithium–Oxygen Batteries


## Overview

Lithium–oxygen (Li–O₂) batteries face critical challenges in cycling stability and overpotential. This work employs **machine learning** to predict the overpotential of electrocatalytic materials (M1–M2–N ternary compounds) for the four-electron LiOH pathway, accelerating the discovery of high-performance catalysts.

We systematically evaluated **8 regression models** (XGBoost, Random Forest, Extra Trees, Gradient Boosting, Ridge, Lasso, ElasticNet, and SVM) with GroupKFold cross-validation, and used **SHAP** for interpretability to identify key physicochemical descriptors governing overpotential.

## Repository Structure

```
MLcode/
├── analysis.ipynb          # Main analysis notebook (ML pipeline)
├── dataset.csv             # Experimental dataset of electrocatalytic materials
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── results/
    └── analysis/
        ├── model_comparison.csv              # 8-model performance comparison
        ├── all_models_oof_r2_summary.csv     # OOF R² summary
        ├── all_models_parity_predictions.csv # Train/test predictions for parity plots
        ├── repeat_validation_folds.csv       # 10-seed repeated validation
        ├── repeat_validation_summary.csv     # Repeated validation summary
        ├── repeat_validation_boxplot.png/pdf # Stability boxplot
        ├── pearson_physical_vs_target.csv    # Pearson correlation (features vs target)
        ├── pearson_heatmap_publication.png/pdf
        ├── spearman_correlations.csv         # Spearman correlation
        ├── shap_total_importance.csv         # SHAP global feature importance
        ├── shap_physical_importance.csv      # SHAP — physical features
        ├── shap_element_importance.csv       # SHAP — element features
        ├── shap_beeswarm_publication.png/pdf # SHAP beeswarm summary plot
        ├── parity_plot_all_models_subplots.png/pdf
        ├── overpotential_vs_cycling_scatter.png
        ├── overpotential_vs_cycling.csv
        ├── comprehensive_3x2_overview_curated.png/pdf
        ├── tsne_embedding.csv
        ├── element_cooccurrence.csv
        ├── element_groups.csv
        ├── coni_category_data.csv           # Co/Ni category analysis data
        ├── coni_category_stats.csv
        ├── coni_category_mean_ranking.png/pdf
        ├── coni_category_rank_mean_95CI.png/pdf
        ├── coni_category_distribution_box.png/pdf
        ├── analysis_summary.json            # Analysis summary metrics
        └── final_model_xgboost.joblib       # Trained XGBoost model
```

## Dataset

`dataset.csv` contains **~200 experimentally reported electrocatalytic materials** for Li–O₂ batteries, each with:

- **Composition**: Ternary M1–M2–N compounds (e.g., Fe–Ce–O, Co–Ni–S)
- **Target variable**: Overpotential (V) — lower values indicate better catalytic activity
- **Auxiliary variable**: Cycling performance (where available)
- **Physicochemical features** (~60 descriptors):
  - Atomic weight, covalent radius, van der Waals radius, atomic volume
  - Electron affinity, d-electron number, valence electron number
  - First ionization energy, electrical/thermal conductivity
  - Pauling electronegativity, d-band center
  - Engineered features (difference, average, and correction terms)

## Requirements

The code is implemented in Python 3.12. Install dependencies with:

```bash
pip install -r requirements.txt
```

Key packages:
- **numpy**, **pandas**, **scipy** — numerical and data processing
- **scikit-learn** — ML models (Random Forest, Ridge, Lasso, ElasticNet, SVM, etc.)
- **xgboost** — XGBoost regressor (best-performing model)
- **matplotlib**, **seaborn** — publication-quality visualizations
- **shap** — model interpretability (SHAP values)
- **joblib** — model serialization
- **jupyter** — notebook environment

## Usage

1. Clone the repository and install dependencies.
2. Launch Jupyter:
   ```bash
   jupyter notebook analysis.ipynb
   ```
3. Run cells sequentially. The notebook will:
   - Load and engineer features from `dataset.csv`
   - Perform Pearson/Spearman correlation analysis
   - Train 8 models with GroupKFold hyperparameter tuning
   - Generate parity plots for all models
   - Run 10-seed repeated validation for the best model
   - Compute SHAP values for interpretability
   - Generate publication-ready figures (saved as PNG/PDF)
   - Export all analysis results to `results/analysis/`

## Key Results

- **Best model**: XGBoost achieved the highest cross-validated R²
- **Feature importance**: SHAP analysis identified d-band center, electronegativity corrections, and atomic radii as key descriptors
- **Co–Ni synergy**: Materials with Co–Ni combinations showed favorable overpotential trends
- **Repeatability**: The best model demonstrated stable performance across 10 random seeds


## License

This project is licensed under the MIT License. See the LICENSE file for details.
