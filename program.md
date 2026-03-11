# autoresearch_regression

This is an autonomous hyperparameter-optimisation research tool for regression,
inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
Give it an Excel dataset and an on-premise LLM, let it run overnight, and wake
up to a log of experiments and (hopefully) a better model.

## Setup

To start a new run, work with the LLM agent to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar11`).
   The branch `autoresearch/<tag>` must not already exist — this is a fresh run.

2. **Create the branch**: `git checkout -b autoresearch/<tag>` from master.

3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `prepare.py` — fixed data loading, preprocessing, and evaluation. **Do NOT modify.**
   - `train.py` — the file you iterate on. Contains model selection, hyperparameters,
     and training loop. **Everything here is fair game.**

4. **Prepare your dataset**:
   - Place your Excel file (`.xlsx`) anywhere accessible and set `DATASET_PATH`:
     ```
     export DATASET_PATH=/path/to/your_data.xlsx
     export TARGET_COLUMN=your_target_column_name
     ```
   - Or use the built-in sample dataset (California Housing):
     ```
     python prepare.py --generate-sample
     ```

5. **Verify the dataset**: Run `python prepare.py` and confirm it prints dataset stats.

6. **Initialise results.tsv**: Create the file with just the header row.
   The baseline will be recorded after the first run.

7. **Confirm and go**: Once setup looks good, kick off the experimentation loop.

---

## Experimentation

Each experiment:
1. Modifies `train.py` (model type, hyperparameters, feature engineering).
2. Runs `python train.py > run.log 2>&1`.
3. Reads the standardised output to extract `val_rmse`.
4. Logs the result in `results.tsv`.
5. Decides whether to keep or discard the change.

### What you CAN do (in `train.py`)
- Change `MODEL_TYPE` to any supported model:
  `random_forest`, `gradient_boosting`, `xgboost`, `lightgbm`, `ridge`, `lasso`
- Tune any hyperparameter in the `*_PARAMS` dicts.
- Add feature-engineering steps (log transforms, polynomial features, interactions)
  **before** passing data to `prepare_data()` — but only by post-processing the
  arrays returned by `prepare_data()`; do not modify `prepare.py`.
- Combine multiple models (stacking, averaging).
- Add early stopping, cross-validation, etc.

### What you CANNOT do
- Modify `prepare.py`. It is read-only.
- Change the train/val/test split logic in `prepare.py`.
- Modify the `evaluate()` function. `val_rmse` is the ground-truth metric.
- Install new packages. Only packages in `requirements.txt` are available.

### The goal
**Minimise `val_rmse`** — lower is better.

### Simplicity criterion
All else being equal, simpler is better.
- A tiny RMSE improvement that adds ugly complexity is not worth it.
- Removing something and getting equal or better results is a great outcome.

---

## Output format

When `train.py` finishes it prints a summary like:

```
---
val_rmse:         0.456789
val_mae:          0.321456
val_r2:           0.834567
train_rmse:       0.234567
train_r2:         0.912345
test_rmse:        0.467890
test_r2:          0.828901
train_seconds:    12.3
model_type:       random_forest
n_features:       8
```

Extract the key metric:
```
grep "^val_rmse:" run.log
```

If the grep output is empty, the run crashed. Check:
```
tail -n 50 run.log
```

---

## Logging results

Log each experiment to `results.tsv` (tab-separated):

```
commit  val_rmse  val_r2  train_seconds  status  description
```

- `commit`        — 7-char git hash
- `val_rmse`      — primary metric (6 decimal places); use 9999.000000 for crashes
- `val_r2`        — coefficient of determination; use 0.000000 for crashes
- `train_seconds` — wall-clock training time; use 0.0 for crashes
- `status`        — `keep`, `discard`, or `crash`
- `description`   — brief description of what this experiment tried

Example:

```
commit  val_rmse  val_r2  train_seconds  status  description
a1b2c3d 0.456789  0.834567  12.3  keep  baseline RandomForest n_estimators=200
b2c3d4e 0.423456  0.851234  45.6  keep  XGBoost lr=0.05 n_estimators=300
c3d4e5f 0.498765  0.802345  18.9  discard  GradientBoosting overfitting
d4e5f6g 9999.000000 0.000000  0.0  crash  LightGBM import error
```

---

## The experiment loop

Run on a dedicated branch (e.g. `autoresearch/mar11`).

LOOP FOREVER:

1. Look at the git state: current branch/commit.
2. Tune `train.py` with one experimental idea (one change at a time).
3. `git commit -am "experiment: <brief description>"`
4. Run: `python train.py > run.log 2>&1`
5. Read results: `grep "^val_rmse:\|^val_r2:\|^train_seconds:" run.log`
6. If empty → crashed. Read `tail -n 50 run.log`, fix, re-run. After 2 failed
   attempts on the same idea, revert and try something else.
7. Compare `val_rmse` to the best so far.
8. **Keep** if `val_rmse` improves (or roughly equal with simpler code).
   **Discard** (`git revert HEAD`) otherwise.
9. Log to `results.tsv`.
10. Go to step 2.

---

## Ideas to try (in rough priority order)

1. Switch model type (`xgboost`, `lightgbm`) — often the biggest gains.
2. Tune `n_estimators` and `learning_rate`.
3. Tune `max_depth` / `num_leaves`.
4. Tune regularisation (`reg_alpha`, `reg_lambda`, `subsample`).
5. Feature engineering: log-transform skewed features, polynomial interactions.
6. Ensemble: average predictions from two complementary models.
7. Cross-validation in train.py (fit on full train+val, report CV RMSE).
