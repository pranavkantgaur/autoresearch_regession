# autoresearch_regression

Autonomous hyperparameter-optimisation research tool for **regression tasks**,
inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

Point it at an Excel spreadsheet (any number of input features, one output
column), connect it to your on-premise LLM running behind
[vllm](https://github.com/vllm-project/vllm), and let it run overnight.  It
will autonomously try different model architectures (Random Forest, Gradient
Boosting, XGBoost, LightGBM, …) and hyperparameter combinations, logging every
experiment to `results.tsv`, and keeping only the changes that improve
validation RMSE.

---

## How it works

The repo has the same three-file philosophy as the original autoresearch:

| File | Role |
|------|------|
| `prepare.py` | **Fixed.** Loads any Excel/CSV dataset, handles preprocessing (encoding, scaling, missing values), performs a reproducible 70/15/15 train/val/test split, and exposes the `evaluate()` function. **Do not modify.** |
| `train.py` | **Agent edits this.** Declares `MODEL_TYPE`, all hyperparameter dicts, and the training loop. Everything here is fair game. |
| `program.md` | **You edit this.** Lightweight instructions given to the LLM agent as its "skill file". |
| `run_agent.py` | **Connects to your vllm server** and runs the autonomous experiment loop. |

The metric the agent optimises is **`val_rmse`** (validation root mean squared
error) — lower is better.  The agent keeps a change if it lowers `val_rmse`
and discards it (via `git checkout HEAD -- train.py`) otherwise.

---

## Supported models

| Key | Library |
|-----|---------|
| `random_forest` | scikit-learn `RandomForestRegressor` |
| `gradient_boosting` | scikit-learn `GradientBoostingRegressor` |
| `xgboost` | [XGBoost](https://xgboost.readthedocs.io/) `XGBRegressor` |
| `lightgbm` | [LightGBM](https://lightgbm.readthedocs.io/) `LGBMRegressor` |
| `ridge` | scikit-learn `Ridge` |
| `lasso` | scikit-learn `Lasso` |

The agent can also combine models (ensembles, stacking) by modifying
`train.py`.

---

## Quick start

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

### 2 · Prepare your dataset

**Option A — Use your own Excel file:**
```bash
export DATASET_PATH=/path/to/your_data.xlsx
export TARGET_COLUMN=price          # name of the column to predict
python prepare.py                   # verifies the dataset
```

Your Excel file can have any number of numeric or categorical input columns.
The tool will auto-encode categoricals and impute missing values.

**Option B — Use the built-in sample dataset:**
```bash
python prepare.py --generate-sample
# creates data/sample_dataset.xlsx  (2000 rows, 10 features)
```

### 3 · Run a single experiment manually (optional smoke test)

```bash
python train.py
```

Output:
```
---
val_rmse:         45.107748
val_mae:          36.539691
val_r2:           0.652073
train_rmse:       16.969422
train_r2:         0.946814
test_rmse:        44.822993
test_r2:          0.640765
train_seconds:    0.3
model_type:       random_forest
n_features:       10
```

### 4 · Start the autonomous agent

```bash
python run_agent.py \
    --base-url http://localhost:8000/v1 \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --dataset data/sample_dataset.xlsx \
    --target target \
    --max-iterations 100
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--base-url` | `http://localhost:8000/v1` | vllm OpenAI-compatible endpoint |
| `--model` | *(required)* | Model name as served by vllm |
| `--dataset` | `data/sample_dataset.xlsx` | Path to Excel or CSV file |
| `--target` | `target` | Target column name |
| `--max-iterations` | `100` | Number of experiment iterations |
| `--temperature` | `0.7` | LLM sampling temperature |
| `--max-tokens` | `4096` | Max tokens in LLM response |

All arguments can also be set via environment variables:
`VLLM_BASE_URL`, `VLLM_MODEL`, `DATASET_PATH`, `TARGET_COLUMN`,
`MAX_ITERATIONS`, `LLM_TEMPERATURE`, `VLLM_API_KEY`.

---

## Project structure

```
prepare.py             — data loading, preprocessing, evaluation  (do not modify)
train.py               — model, hyperparameters, training loop    (agent modifies this)
program.md             — agent instructions
run_agent.py           — autonomous LLM agent loop
data/
  sample_dataset.xlsx  — sample regression dataset (2000 rows, 10 features)
results.tsv            — experiment log (tab-separated)
requirements.txt       — Python dependencies
```

---

## Connecting to an on-premise LLM (vllm)

Start your vllm server as usual:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --port 8000
```

`run_agent.py` uses the OpenAI Python client pointed at your local endpoint —
no API key required for most vllm deployments.  Pass `--base-url` accordingly.

> **Models that work well:** Any instruction-tuned model (Mistral, LLaMA 3,
> Qwen, Phi-3) capable of writing and modifying Python code.  Larger models
> (≥13 B) tend to make more meaningful hyperparameter suggestions.

---

## Sample results on the Diabetes Regression dataset

The sample dataset (`data/sample_dataset.xlsx`) is built from the
[UCI Diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
(442 real samples) augmented with 1 558 synthetic samples generated by
`sklearn.datasets.make_regression` to reach 2 000 rows.

**Dataset:** 2 000 samples · 10 clinical features · 1 continuous target
(disease progression score)  
**Split:** 70% train (1 400) / 15% val (300) / 15% test (300)  
**Primary metric:** val RMSE (lower is better)

### Experiment log

| Experiment | Model | val RMSE ↓ | val R² ↑ | Status |
|------------|-------|-----------|---------|--------|
| 1 – Baseline | Random Forest (n=200) | 45.11 | 0.652 | ✅ keep |
| 2 | Gradient Boosting (n=300, lr=0.05, depth=5) | 37.72 | 0.757 | ✅ keep |
| 3 | **XGBoost (n=300, lr=0.05, depth=6)** | **37.61** | **0.758** | ✅ keep |
| 4 | XGBoost (n=500, lr=0.02, depth=5, reg) | 37.61 | 0.758 | ⬇ discard (no gain) |
| 5 | LightGBM (n=500, lr=0.03, leaves=63) | 37.94 | 0.754 | ⬇ discard |

### Key findings

* **Switching from Random Forest to gradient-boosted trees (XGBoost/LightGBM)
  reduced val RMSE by ~17%** (45.11 → 37.61) without any feature engineering.
* XGBoost with `n_estimators=300`, `learning_rate=0.05`, `max_depth=6` was
  the best single model found.
* Further tuning (more estimators, lower learning rate, stronger regularisation)
  did not improve over the baseline XGBoost configuration on this dataset.
* The agent correctly identified that Gradient Boosting and XGBoost were
  superior to Random Forest and LightGBM for this dataset.

### How to reproduce

```bash
pip install -r requirements.txt
python prepare.py --generate-sample
python train.py   # baseline (Random Forest)

# Then edit train.py: set MODEL_TYPE = "xgboost" to get best result
python train.py   # XGBoost best model
```

---

## Adapting to your own dataset

1. Export your Excel sheet with input features in any columns and the target
   in one column (e.g. `price`, `yield`, `score`).
2. Set `DATASET_PATH` and `TARGET_COLUMN` before running.
3. Categorical columns are automatically one-hot encoded.
4. Missing values are imputed with column means.
5. All feature scaling is handled by `prepare.py`.

There is no limit on the number of features — the `n_features` value is
printed at the start of every run.

---

## License

MIT