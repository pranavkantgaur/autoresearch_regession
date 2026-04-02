"""
Regression training script for autoresearch.

This is the ONLY file the agent is allowed to edit.
It contains the model type, hyperparameters, and training loop.
The agent should modify MODEL_TYPE and the hyperparameter dictionaries
to improve val_rmse.

Usage:
    python train.py
    DATASET_PATH=my_data.xlsx TARGET_COLUMN=price python train.py
"""

import os
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso

from prepare import prepare_data, evaluate, TIME_BUDGET

# ---------------------------------------------------------------------------
# Configuration — agent modifies this section
# ---------------------------------------------------------------------------

# Choose model: "random_forest" | "gradient_boosting" | "xgboost" | "lightgbm"
#               "ridge" | "lasso"
MODEL_TYPE = "random_forest"

# Hyperparameters for each model type
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": 42,
}

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 4,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "subsample": 0.8,
    "max_features": "sqrt",
    "random_state": 42,
}

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": -1,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

RIDGE_PARAMS = {
    "alpha": 1.0,
}

LASSO_PARAMS = {
    "alpha": 0.01,
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
    # Load and prepare data
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     scaler, feature_names, n_features) = prepare_data()

    model = build_model(MODEL_TYPE)

    # Train within the time budget
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0

    # Evaluate
    val_metrics = evaluate(y_val, model.predict(X_val))
    train_metrics = evaluate(y_train, model.predict(X_train))
    test_metrics = evaluate(y_test, model.predict(X_test))

    # Print standardised summary (parsed by run_agent.py)
    print("---")
    print(f"val_rmse:         {val_metrics['rmse']:.6f}")
    print(f"val_mae:          {val_metrics['mae']:.6f}")
    print(f"val_r2:           {val_metrics['r2']:.6f}")
    print(f"train_rmse:       {train_metrics['rmse']:.6f}")
    print(f"train_r2:         {train_metrics['r2']:.6f}")
    print(f"test_rmse:        {test_metrics['rmse']:.6f}")
    print(f"test_r2:          {test_metrics['r2']:.6f}")
    print(f"train_seconds:    {train_seconds:.1f}")
    print(f"model_type:       {MODEL_TYPE}")
    print(f"n_features:       {n_features}")


if __name__ == "__main__":
    main()
