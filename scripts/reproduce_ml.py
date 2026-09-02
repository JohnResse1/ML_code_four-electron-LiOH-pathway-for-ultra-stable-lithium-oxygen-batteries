#!/usr/bin/env python3
"""Run the corrected eight-model workflow without overwriting legacy results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reproducibility import (  # noqa: E402
    CONFIG,
    DESCRIPTOR_COLUMNS,
    add_record_ids,
    build_model_input,
    make_group_splits,
)

OUT = ROOT / "results" / "checklist_reproduction"
SEED = CONFIG["cross_validation"]["random_seed"]


def specifications():
    return {
        "elasticnet": (
            Pipeline([("scaler", StandardScaler()), ("reg", ElasticNet(max_iter=20000, random_state=SEED))]),
            {"reg__alpha": [0.01, 0.1, 1.0], "reg__l1_ratio": [0.2, 0.5, 0.8]},
        ),
        "extratrees": (
            ExtraTreesRegressor(random_state=SEED, n_jobs=1),
            {"n_estimators": [300, 500], "max_depth": [None, 10], "min_samples_split": [2, 5]},
        ),
        "gradientboosting": (
            GradientBoostingRegressor(random_state=SEED),
            {"n_estimators": [200, 400], "learning_rate": [0.03, 0.05], "max_depth": [2, 3]},
        ),
        "lasso": (
            Pipeline([("scaler", StandardScaler()), ("reg", Lasso(max_iter=20000, random_state=SEED))]),
            {"reg__alpha": [0.001, 0.01, 0.1]},
        ),
        "rf": (
            RandomForestRegressor(random_state=SEED, n_jobs=1),
            {"n_estimators": [300, 500], "max_depth": [None, 10], "min_samples_split": [2, 5]},
        ),
        "ridge": (
            Pipeline([("scaler", StandardScaler()), ("reg", Ridge(random_state=SEED))]),
            {"reg__alpha": [0.1, 1.0, 10.0]},
        ),
        "svm": (
            Pipeline([("scaler", StandardScaler()), ("reg", SVR(kernel="rbf"))]),
            {"reg__C": [1, 10], "reg__gamma": ["scale", 0.05], "reg__epsilon": [0.05, 0.1]},
        ),
        "xgboost": (
            xgb.XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0, objective="reg:squarederror"),
            {"n_estimators": [200, 400], "max_depth": [3, 5], "learning_rate": [0.03, 0.05], "subsample": [0.8, 1.0]},
        ),
    }


def evaluate(estimator, x, y, splits, records, model_name):
    fold_scores, predictions = [], []
    oof = np.empty(len(y), dtype=float)
    for fold, (train_idx, test_idx) in enumerate(splits):
        fitted = clone(estimator).fit(x.iloc[train_idx], y[train_idx])
        for assignment, idx in (("train", train_idx), ("test", test_idx)):
            pred = fitted.predict(x.iloc[idx])
            if assignment == "test":
                oof[idx] = pred
                fold_scores.append(r2_score(y[idx], pred))
            predictions.append(pd.DataFrame({
                "model": model_name,
                "fold": fold,
                "assignment": assignment,
                "record_id": records.iloc[idx]["record_id"].values,
                "DOI": records.iloc[idx]["DOI"].values,
                "y_true": y[idx],
                "y_pred": pred,
            }))
    return np.asarray(fold_scores), oof, pd.concat(predictions, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = add_record_ids(pd.read_csv(ROOT / "dataset.csv"))
    x = build_model_input(raw)
    y = pd.to_numeric(raw[CONFIG["targets"]["Y1"]["source_column"]], errors="raise").to_numpy()
    groups = raw[CONFIG["cross_validation"]["group_column"]].astype(str).to_numpy()
    splits = make_group_splits(groups)

    rows, all_predictions, fitted_models = [], [], {}
    for name, (estimator, grid) in specifications().items():
        print(f"Tuning {name} on {x.shape[1]} model-input columns ({len(DESCRIPTOR_COLUMNS)} descriptors)", flush=True)
        # Single-process execution avoids platform-dependent process/semaphore
        # limits; it does not change the fitted estimators or CV assignments.
        search = GridSearchCV(estimator, grid, cv=splits, scoring="r2", n_jobs=1, refit=True)
        search.fit(x, y)
        scores, oof, predictions = evaluate(search.best_estimator_, x, y, splits, raw, name)
        rows.append({
            "model": name,
            "oof_r2": r2_score(y, oof),
            "cv_mean_r2": scores.mean(),
            "cv_std_r2": scores.std(),
            "grid_best_score": search.best_score_,
            "best_params": json.dumps(search.best_params_, sort_keys=True),
            "n_descriptors": len(DESCRIPTOR_COLUMNS),
            "n_internal_one_hot": x.shape[1] - len(DESCRIPTOR_COLUMNS),
            "n_model_input_columns": x.shape[1],
        })
        all_predictions.append(predictions)
        fitted_models[name] = search.best_estimator_

    comparison = pd.DataFrame(rows).sort_values("oof_r2", ascending=False).reset_index(drop=True)
    comparison.to_csv(OUT / "model_comparison_corrected.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(OUT / "all_models_predictions_corrected.csv", index=False)
    best_name = comparison.loc[0, "model"]
    joblib.dump(fitted_models[best_name], OUT / f"final_model_{best_name}_47_descriptors.joblib")

    legacy_path = ROOT / "results" / "analysis" / "all_models_oof_r2_summary.csv"
    legacy = pd.read_csv(legacy_path).set_index("model") if legacy_path.exists() else pd.DataFrame()
    impact = comparison[["model", "oof_r2"]].rename(columns={"oof_r2": "corrected_oof_r2"})
    if not legacy.empty:
        impact["legacy_saved_oof_r2"] = impact["model"].map(legacy["oof_r2"])
        impact["delta_corrected_minus_legacy"] = impact["corrected_oof_r2"] - impact["legacy_saved_oof_r2"]
    impact.to_csv(OUT / "legacy_vs_corrected_metrics.csv", index=False)
    summary = {
        "best_model": best_name,
        "best_model_oof_r2": float(comparison.loc[0, "oof_r2"]),
        "n_records": len(raw),
        "n_descriptors": len(DESCRIPTOR_COLUMNS),
        "n_internal_one_hot_columns": x.shape[1] - len(DESCRIPTOR_COLUMNS),
        "n_model_input_columns": x.shape[1],
        "doi_in_model_input": "DOI" in x.columns,
        "experimental_context_in_model_input": bool(set(CONFIG["experimental_context_columns"]) & set(x.columns)),
        "legacy_results_preserved_at": "results/analysis",
    }
    (OUT / "analysis_summary_corrected.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
