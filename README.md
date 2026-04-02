# autoresearch_regression

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/pranavkantgaur/autoresearch_regession)

Autonomous hyperparameter-optimisation research tool for **regression tasks**,
inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

Point it at an Excel spreadsheet (any number of input features, one output
column), and let it run overnight.  It autonomously tries different model
architectures (Random Forest, Gradient Boosting, XGBoost, LightGBM, …) and
hyperparameter combinations, logging every experiment to `results.tsv`, keeping
only changes that improve validation RMSE.

**Two LLM backends are supported — pick the one that suits you:**

| Backend | When to use | Setup |
|---------|-------------|-------|
| **GitHub Models** *(default)* | You have GitHub Copilot Pro / Free and want a zero-config setup — especially in Codespaces | `GITHUB_TOKEN` is auto-injected in Codespaces; add a PAT locally |
| **On-premise vllm** | You have your own GPU server | Pass `--base-url` |

---

## How it works

The repo has the same three-file philosophy as the original autoresearch:

| File | Role |
|------|------|
| `prepare.py` | **Fixed.** Loads any Excel/CSV dataset, handles preprocessing (encoding, scaling, missing values), performs a reproducible 70/15/15 train/val/test split, and exposes the `evaluate()` function. **Do not modify.** |
| `train.py` | **Agent edits this.** Declares `MODEL_TYPE`, all hyperparameter dicts, and the training loop. Everything here is fair game. |
| `program.md` | **You edit this.** Lightweight instructions given to the LLM agent as its "skill file". |
| `run_agent.py` | **Runs the autonomous loop.** Calls GitHub Models or vllm, modifies `train.py`, evaluates, keeps/reverts with git. |

The metric the agent optimises is **`val_rmse`** (validation root mean squared
error) — lower is better.

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

The agent can also combine models (ensembles, stacking) by modifying `train.py`.

---

## Option A — GitHub Copilot Pro in GitHub Codespaces *(recommended)*

### 1 · Open in Codespaces

Click the badge at the top of this README, or go to
**Code → Codespaces → Create codespace on this branch**.

Python dependencies are installed automatically on container creation.
`GITHUB_TOKEN` is injected automatically — no extra setup required.

### 2 · Prepare your dataset

**Use your own Excel file** — upload it via the VS Code explorer, then:
```bash
export DATASET_PATH=data/my_data.xlsx
export TARGET_COLUMN=price        # name of the column to predict
python prepare.py                 # verifies the dataset
```

**Or use the built-in sample dataset:**
```bash
python prepare.py --generate-sample
# creates data/sample_dataset.xlsx  (2000 rows, 10 features)
```

### 3 · Run a single experiment manually (smoke test)

```bash
python train.py
```

### 4 · Start the autonomous agent

```bash
# Uses gpt-4o-mini via GitHub Models by default
python run_agent.py \
    --dataset data/sample_dataset.xlsx \
    --target target \
    --max-iterations 50
```

To use a more powerful model (uses more of your Copilot quota):
```bash
python run_agent.py --model gpt-4o --max-iterations 50
```

Available GitHub Models for code tasks:
- `gpt-4o-mini` *(default — fast, free tier, good results)*
- `gpt-4o` *(higher quality suggestions)*
- `Meta-Llama-3.1-70B-Instruct`
- `Meta-Llama-3.1-8B-Instruct` *(fastest)*

See the full list at <https://github.com/marketplace/models>.

---

## Option B — GitHub Copilot Pro locally (PAT)

If you want to run on your local machine without Codespaces:

1. Create a GitHub Personal Access Token with **`models:read`** scope at
   <https://github.com/settings/tokens>.
2. Export it:
   ```bash
   export GITHUB_TOKEN=ghp_...
   ```
3. Run the agent — it will auto-detect the token and use GitHub Models:
   ```bash
   python run_agent.py \
       --dataset data/sample_dataset.xlsx \
       --target target
   ```

---

## Option C — On-premise vllm

If you have your own GPU server running vllm:

```bash
python run_agent.py \
    --base-url http://YOUR_SERVER:8000/v1 \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --dataset data/sample_dataset.xlsx \
    --target target \
    --max-iterations 100
```

> **Models that work well:** Any instruction-tuned model (Mistral, LLaMA 3,
> Qwen, Phi-3) capable of writing and modifying Python code.  Larger models
> (≥13 B) tend to make more meaningful hyperparameter suggestions.

---

## All CLI arguments

```bash
python run_agent.py [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--base-url` | `https://models.inference.ai.azure.com` if `GITHUB_TOKEN` set, else `http://localhost:8000/v1` | LLM endpoint URL |
| `--model` | `gpt-4o-mini` (GitHub Models) | Model name |
| `--api-key` | `GITHUB_TOKEN` env var | API key (auto-resolved; rarely needed) |
| `--dataset` | `data/sample_dataset.xlsx` | Path to Excel or CSV file |
| `--target` | `target` | Target column name |
| `--max-iterations` | `100` | Number of experiment iterations |
| `--temperature` | `0.7` | LLM sampling temperature |
| `--max-tokens` | `4096` | Max tokens in LLM response |

All arguments can also be set via environment variables:
`GITHUB_TOKEN`, `GITHUB_MODELS_MODEL`, `VLLM_BASE_URL`, `VLLM_MODEL`,
`VLLM_API_KEY`, `DATASET_PATH`, `TARGET_COLUMN`, `MAX_ITERATIONS`,
`LLM_TEMPERATURE`.

---

## Project structure

```
prepare.py             — data loading, preprocessing, evaluation  (do not modify)
train.py               — model, hyperparameters, training loop    (agent modifies this)
program.md             — agent instructions
run_agent.py           — autonomous LLM agent loop (GitHub Models or vllm)
.devcontainer/
  devcontainer.json    — Codespaces / Dev Container configuration
data/
  sample_dataset.xlsx  — sample regression dataset (2000 rows, 10 features)
results.tsv            — experiment log (tab-separated)
requirements.txt       — Python dependencies
```

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
2. Upload the file to the repo (or mount it in your Codespace).
3. Set `DATASET_PATH` and `TARGET_COLUMN` before running.
4. Categorical columns are automatically one-hot encoded.
5. Missing values are imputed with column means.
6. All feature scaling is handled by `prepare.py`.

There is no limit on the number of features — the `n_features` value is
printed at the start of every run.

---

## License

MIT