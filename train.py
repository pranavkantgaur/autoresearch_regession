"""
Regression training script for autoresearch.

This is the ONLY file the agent is allowed to edit.
It contains the model type, hyperparameters, and training loop.
The agent should modify MODEL_TYPE and the hyperparameter dictionaries
to improve val_rmse.

Usage:
    python train.py
    DATASET_PATH=my_data.xlsx TARGET_COLUMN=price python train.py

Dataset notes (updated_dataset_4_madam_MB_dye.xlsx):
  - 56 samples, 7 features, target = 'Adsorption capacity (mg/g)'
  - Target skewness ~2.1 → log1p-transform applied before training;
    predictions are back-transformed (expm1) before test evaluation.
  - Small dataset: val_rmse uses Leave-One-Out cross-validation on the
    combined train+val pool (n=47). LOO maximises training data per fold
    and gives the most stable RMSE estimate for very small datasets.
    The agent should minimise this LOO-RMSE (val_rmse).
  - Final model is retrained on the full train+val pool and evaluated on
    the held-out test set in the original (non-log) space.
"""

import os
import sys
import time
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import LeaveOneOut, cross_val_score

from prepare import prepare_data, evaluate, TIME_BUDGET

# ---------------------------------------------------------------------------
# Configuration — agent modifies this section
# ---------------------------------------------------------------------------

# Choose model: "random_forest" | "gradient_boosting" | "xgboost" | "lightgbm"
#               "ridge" | "lasso"
MODEL_TYPE = "gradient_boosting"

# Hyperparameters for each model type
RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "min_samples_split": 4,
    "min_samples_leaf": 1,
    "max_features": None,  # use all features — best for this 7-feature dataset
    "n_jobs": -1,
    "random_state": 42,
}

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 350,
    "learning_rate": 0.02,
    "max_depth": 3,
    "min_samples_split": 4,
    "min_samples_leaf": 1,
    "subsample": 0.85,
    "max_features": None,  # use all features — best for this 7-feature dataset
    "random_state": 42,
}

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 1.0,  # use all features — best for 7-feature dataset
    "min_child_weight": 3,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 4,
    "num_leaves": 15,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "min_child_samples": 5,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

RIDGE_PARAMS = {
    "alpha": 10.0,
}

LASSO_PARAMS = {
    "alpha": 0.05,
    "max_iter": 5000,
}

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(model_type: str):
    """Return an unfitted scikit-learn–compatible regressor."""
    mt = model_type.lower()
    if mt == "random_forest":
        return RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    elif mt == "gradient_boosting":
        return GradientBoostingRegressor(**GRADIENT_BOOSTING_PARAMS)
    elif mt == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(**XGBOOST_PARAMS)
    elif mt == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**LIGHTGBM_PARAMS)
    elif mt == "ridge":
        return Ridge(**RIDGE_PARAMS)
    elif mt == "lasso":
        return Lasso(**LASSO_PARAMS)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    # Load and prepare data (prepare.py handles scaling; do not modify prepare.py)
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     scaler, feature_names, n_features) = prepare_data()

    # -----------------------------------------------------------------------
    # Target transform: log1p reduces right-skew (skewness ~2.1) and
    # prevents large-value samples from dominating RMSE.
    # -----------------------------------------------------------------------
    y_train_log = np.log1p(y_train)
    y_val_log   = np.log1p(y_val)

    # Combine train + val into one pool for cross-validation.
    # The test split is kept strictly separate throughout.
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train_log, y_val_log])

    # -----------------------------------------------------------------------
    # val_rmse: Leave-One-Out cross-validation on train+val pool (log space).
    # LOO uses n-1 samples for training each fold, giving the most
    # data-efficient and stable estimate for this very small dataset (n=47).
    # -----------------------------------------------------------------------
    loo = LeaveOneOut()
    model_cv = build_model(MODEL_TYPE)

    t0 = time.perf_counter()
    cv_rmse = -cross_val_score(
        model_cv, X_tv, y_tv, cv=loo,
        scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="R.*2 score is not well-defined")
        cv_r2 = cross_val_score(
            model_cv, X_tv, y_tv, cv=loo, scoring="r2", n_jobs=-1
        )
    train_seconds = time.perf_counter() - t0

    val_rmse = float(cv_rmse.mean())
    # LOO folds have 1 test sample each, so per-fold R² is undefined.
    # Report the mean of any non-NaN fold R² scores (or 0 if all are NaN).
    valid_r2 = cv_r2[~np.isnan(cv_r2)]
    val_r2   = float(valid_r2.mean()) if len(valid_r2) > 0 else 0.0
    val_mae  = float(
        -cross_val_score(model_cv, X_tv, y_tv, cv=LeaveOneOut(),
                         scoring="neg_mean_absolute_error", n_jobs=-1).mean()
    )

    # -----------------------------------------------------------------------
    # Final model: fit on full train+val, evaluate on held-out test.
    # Predictions are back-transformed to original mg/g space for reporting.
    # -----------------------------------------------------------------------
    final_model = build_model(MODEL_TYPE)
    final_model.fit(X_tv, y_tv)

    train_pred   = np.expm1(final_model.predict(X_tv))
    test_pred    = np.expm1(final_model.predict(X_test))
    y_tv_orig    = np.expm1(y_tv)

    train_metrics = evaluate(y_tv_orig, train_pred)
    test_metrics  = evaluate(y_test,    test_pred)

    # Print standardised summary (parsed by run_agent.py)
    print("---")
    print(f"val_rmse:         {val_rmse:.6f}")
    print(f"val_mae:          {val_mae:.6f}")
    print(f"val_r2:           {val_r2:.6f}")
    print(f"train_rmse:       {train_metrics['rmse']:.6f}")
    print(f"train_r2:         {train_metrics['r2']:.6f}")
    print(f"test_rmse:        {test_metrics['rmse']:.6f}")
    print(f"test_r2:          {test_metrics['r2']:.6f}")
    print(f"train_seconds:    {train_seconds:.1f}")
    print(f"model_type:       {MODEL_TYPE}")
    print(f"n_features:       {n_features}")


if __name__ == "__main__":
    main()
