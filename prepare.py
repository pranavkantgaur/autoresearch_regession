"""
Fixed utilities for regression autoresearch.
Contains data loading from Excel/CSV, preprocessing, and evaluation.

This file is NOT modified by the agent. It provides stable, reproducible
data splits and evaluation metrics.

Usage:
    python prepare.py                  # generate/verify sample dataset
    python prepare.py --dataset path/to/your_data.xlsx --target price
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300          # wall-clock training budget in seconds (5 minutes)
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Default dataset settings (override via env vars or CLI)
DEFAULT_DATA_FILE = os.environ.get("DATASET_PATH", "data/sample_dataset.xlsx")
DEFAULT_TARGET_COL = os.environ.get("TARGET_COLUMN", "target")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataframe(filepath: str) -> pd.DataFrame:
    """Load a dataset from an Excel (.xlsx/.xls) or CSV file."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath, engine="openpyxl")
    elif ext == ".csv":
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .xlsx, .xls, or .csv")
    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def prepare_data(
    filepath: str = None,
    target_col: str = None,
    scale_features: bool = True,
):
    """
    Load, preprocess, and split the dataset into train/val/test splits.

    Returns:
        X_train, X_val, X_test : np.ndarray  — feature matrices
        y_train, y_val, y_test : np.ndarray  — target vectors
        scaler                  : StandardScaler | None
        feature_names           : list[str]
        n_features              : int
    """
    if filepath is None:
        filepath = DEFAULT_DATA_FILE
    if target_col is None:
        target_col = DEFAULT_TARGET_COL

    df = load_dataframe(filepath)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col].values.astype(np.float64)

    # Encode categoricals, fill missing values
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(X.mean(numeric_only=True))
    feature_names = X.columns.tolist()
    X = X.values.astype(np.float64)

    # Train / val / test split  (stratify not applicable for regression)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, random_state=RANDOM_STATE
    )

    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    n_features = X_train.shape[1]
    print(
        f"Dataset: {len(df)} samples | "
        f"{n_features} features | "
        f"train={len(X_train)} val={len(X_val)} test={len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, n_features


# ---------------------------------------------------------------------------
# Evaluation  (ground-truth metric — do not modify)
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics.

    Returns a dict with keys: rmse, mae, r2
    The primary optimisation target is **val_rmse** (lower is better).
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Dataset generation helper (run once to create sample data)
# ---------------------------------------------------------------------------

def generate_sample_dataset(output_path: str = "data/sample_dataset.xlsx"):
    """
    Generate a sample regression dataset and save it as an Excel file.

    Uses sklearn's Diabetes dataset (10 clinical features, 442 samples) as
    baseline and augments it with make_regression to reach 2 000 samples,
    mimicking a typical Kaggle regression challenge.

    The Diabetes dataset is a well-known benchmark available from sklearn
    without any network download (data is bundled with the package).
    It is structurally similar to Kaggle regression datasets:
      - 10 numeric input features (age, sex, BMI, blood pressure, 6 serum
        measurements)
      - 1 continuous target (disease progression score one year after baseline)
    """
    from sklearn.datasets import load_diabetes, make_regression

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- real part: 442 samples from the UCI Diabetes dataset ---------------
    diab = load_diabetes(as_frame=True)
    df_real = diab.frame.rename(columns={"target": "target"})
    feature_names = [c for c in df_real.columns if c != "target"]

    # --- synthetic part: 1558 extra samples to reach 2000 total ---------------
    X_syn, y_syn = make_regression(
        n_samples=1558,
        n_features=len(feature_names),
        noise=50.0,
        random_state=RANDOM_STATE,
    )
    # Align synthetic data to the real feature scale (approximate)
    scaler_mean = df_real[feature_names].mean().values
    scaler_std = df_real[feature_names].std().values
    X_syn = X_syn * scaler_std + scaler_mean
    y_syn = np.clip(
        y_syn * df_real["target"].std() / y_syn.std() + df_real["target"].mean(),
        df_real["target"].min(), df_real["target"].max()
    )
    df_syn = pd.DataFrame(X_syn, columns=feature_names)
    df_syn["target"] = y_syn

    df = pd.concat([df_real, df_syn], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    df.to_excel(output_path, index=False, engine="openpyxl")
    print(f"Sample dataset saved to {output_path}  ({len(df)} rows, {len(feature_names)} features)")
    print(f"Features : {feature_names}")
    print(f"Target   : 'target'  (disease progression score, UCI Diabetes dataset)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare regression autoresearch data")
    parser.add_argument("--dataset", default=None, help="Path to your Excel/CSV dataset")
    parser.add_argument("--target", default=None, help="Name of the target column")
    parser.add_argument("--generate-sample", action="store_true",
                        help="Generate the built-in UCI Diabetes sample dataset")
    args = parser.parse_args()

    if args.generate_sample or not os.path.exists(DEFAULT_DATA_FILE):
        print("Generating sample dataset …")
        generate_sample_dataset()

    dataset = args.dataset or DEFAULT_DATA_FILE
    target = args.target or DEFAULT_TARGET_COL

    print(f"\nVerifying dataset: {dataset}  (target='{target}')")
    prepare_data(filepath=dataset, target_col=target)
    print("Dataset OK — ready for training.")
