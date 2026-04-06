"""
Regression training script for autoresearch.

This is the ONLY file the agent is allowed to edit.
It contains the model type, hyperparameters, and training loop.
The agent should modify MODEL_TYPE and the hyperparameter dictionaries
to improve val_rmse.

Usage:
    python train.py
    DATASET_PATH=my_data.xlsx TARGET_COLUMN=price python train.py

Current configuration targets Dataset 5 (updated_dataset_5_madam_MB_dye.xlsx):
  - 48 samples, 6 features, target = 'QM mg/g Y'
  - Hyperparameters selected via 5-fold CV HPO (see analysis_report.tex
    Section "Dataset 5 — Model Selection and HPO").
  - For Dataset 4 (56 samples, 7 features, target = 'Adsorption capacity
    (mg/g)'), the HPO-selected GB config is: n_estimators=30,
    learning_rate=0.07, max_depth=4, subsample=0.8, max_features='sqrt'.

Evaluation methodology (mirrors the stability-analysis notebook):
  - N_TRIALS (default 10) random train/test splits of the combined
    train+val pool (80 % train, 20 % test), each with a different seed.
  - val_rmse = mean test RMSE over those trials  ← primary optimisation target
  - val_r2   = mean test R²  over those trials
  - The fixed held-out test set (from prepare_data) is used only for the
    final test_rmse / test_r2 fields; it is never touched during training.

  Additionally, Leave-One-Out CV is reported in the *same* original-space
  units (via prepare.evaluate_loo_cv), providing a deterministic but
  high-variance complementary estimate.
"""

import os
import sys
import time
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split

from prepare import prepare_data, evaluate, evaluate_loo_cv, TIME_BUDGET

# ---------------------------------------------------------------------------
# Configuration — agent modifies this section
# ---------------------------------------------------------------------------

# Choose model: "random_forest" | "gradient_boosting" | "xgboost" | "lightgbm"
#               "ridge" | "lasso"
MODEL_TYPE = "gradient_boosting"

# Number of random trials and their seeds (mirrors notebook: n_trials=10,
# seeds drawn from 0-999 without replacement).  Fixed here for reproducibility.
# Generated via: np.random.RandomState(42).choice(1000, size=10, replace=False)
N_TRIALS = 10
TRIAL_SEEDS = [521, 737, 740, 660, 411, 678, 626, 513, 859, 136]

# Whether to log1p-transform the target before fitting.
# Train/val metrics are always reported in the original space (expm1 applied
# when True).  For GradientBoosting the raw-space target performs better.
USE_LOG_TRANSFORM = False

# Hyperparameters for each model type (tuned for Dataset 5 via HPO)
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,          # unlimited depth; HPO-selected for Dataset 5
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": None,       # use all features
    "n_jobs": -1,
    "random_state": 42,         # overridden per-trial below
}

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 30,         # HPO-selected for Dataset 5
    "learning_rate": 0.1,       # HPO-selected for Dataset 5
    "max_depth": 5,             # HPO-selected for Dataset 5
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "subsample": 1.0,           # HPO-selected for Dataset 5
    "max_features": "sqrt",
    "random_state": 42,
}

XGBOOST_PARAMS = {
    "n_estimators": 50,
    "learning_rate": 0.1,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 4,
    "num_leaves": 15,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "min_child_samples": 3,
    "reg_alpha": 0.0,
    "reg_lambda": 0.3,
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

def build_model(model_type: str, seed: int = 42):
    """Return an unfitted scikit-learn–compatible regressor."""
    mt = model_type.lower()
    if mt == "random_forest":
        params = dict(RANDOM_FOREST_PARAMS); params["random_state"] = seed
        return RandomForestRegressor(**params)
    elif mt == "gradient_boosting":
        params = dict(GRADIENT_BOOSTING_PARAMS); params["random_state"] = seed
        return GradientBoostingRegressor(**params)
    elif mt == "xgboost":
        from xgboost import XGBRegressor
        params = dict(XGBOOST_PARAMS); params["random_state"] = seed
        return XGBRegressor(**params)
    elif mt == "lightgbm":
        from lightgbm import LGBMRegressor
        params = dict(LIGHTGBM_PARAMS); params["random_state"] = seed
        return LGBMRegressor(**params)
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

    # Combine train + val into one pool for the trial splits, exactly as in the
    # notebook (the notebook uses the full dataset; here we exclude the held-out
    # test split to avoid data leakage).
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])

    # -----------------------------------------------------------------------
    # val_rmse: mean test RMSE over N_TRIALS random 80/20 splits of X_tv / y_tv.
    # Training uses log1p-transformed target; metrics are computed in the
    # original mg/g space (expm1 back-transform) to match the notebook.
    # -----------------------------------------------------------------------
    trial_test_rmse  = []
    trial_test_r2    = []
    trial_test_mae   = []
    trial_train_rmse = []
    trial_train_r2   = []

    t0 = time.perf_counter()
    for seed in TRIAL_SEEDS:
        Xt, Xts, yt, yts = train_test_split(
            X_tv, y_tv, test_size=0.2, random_state=int(seed)
        )
        model = build_model(MODEL_TYPE, seed=int(seed))
        yt_fit = np.log1p(yt) if USE_LOG_TRANSFORM else yt
        model.fit(Xt, yt_fit)

        # Back-transform to original space for evaluation
        yt_pred  = np.expm1(model.predict(Xt))  if USE_LOG_TRANSFORM else model.predict(Xt)
        yts_pred = np.expm1(model.predict(Xts)) if USE_LOG_TRANSFORM else model.predict(Xts)

        tr_metrics  = evaluate(yt,  yt_pred)
        tst_metrics = evaluate(yts, yts_pred)

        trial_train_rmse.append(tr_metrics["rmse"])
        trial_train_r2.append(tr_metrics["r2"])
        trial_test_rmse.append(tst_metrics["rmse"])
        trial_test_r2.append(tst_metrics["r2"])
        trial_test_mae.append(tst_metrics["mae"])

    train_seconds = time.perf_counter() - t0

    val_rmse  = float(np.mean(trial_test_rmse))
    val_r2    = float(np.mean(trial_test_r2))
    val_mae   = float(np.mean(trial_test_mae))

    # -----------------------------------------------------------------------
    # Final model: fit on full train+val pool, evaluate on fixed held-out test.
    # (matches the notebook's intent: train on all available data, report on
    # a truly unseen partition)
    # -----------------------------------------------------------------------
    final_model = build_model(MODEL_TYPE)
    y_tv_fit = np.log1p(y_tv) if USE_LOG_TRANSFORM else y_tv
    final_model.fit(X_tv, y_tv_fit)

    train_pred = np.expm1(final_model.predict(X_tv)) if USE_LOG_TRANSFORM else final_model.predict(X_tv)
    test_pred  = np.expm1(final_model.predict(X_test)) if USE_LOG_TRANSFORM else final_model.predict(X_test)

    train_metrics = evaluate(y_tv,    train_pred)
    test_metrics  = evaluate(y_test,  test_pred)

    # -----------------------------------------------------------------------
    # LOO CV — always in original space (via prepare.evaluate_loo_cv)
    # -----------------------------------------------------------------------
    loo_metrics = evaluate_loo_cv(
        X_tv, y_tv,
        build_model_fn=lambda seed: build_model(MODEL_TYPE, seed=seed),
        use_log_transform=USE_LOG_TRANSFORM,
    )

    # Print standardised summary (parsed by run_agent.py)
    print("---")
    print(f"val_rmse:         {val_rmse:.6f}")
    print(f"val_mae:          {val_mae:.6f}")
    print(f"val_r2:           {val_r2:.6f}")
    print(f"loo_rmse:         {loo_metrics['loo_rmse']:.6f}")
    print(f"loo_r2:           {loo_metrics['loo_r2']:.6f}")
    print(f"train_rmse:       {train_metrics['rmse']:.6f}")
    print(f"train_r2:         {train_metrics['r2']:.6f}")
    print(f"test_rmse:        {test_metrics['rmse']:.6f}")
    print(f"test_r2:          {test_metrics['r2']:.6f}")
    print(f"train_seconds:    {train_seconds:.1f}")
    print(f"model_type:       {MODEL_TYPE}")
    print(f"n_features:       {n_features}")


if __name__ == "__main__":
    main()

