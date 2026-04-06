"""
Autonomous regression research agent.

Supports two LLM backends:

  1. **GitHub Models** (default when running in GitHub Codespaces or whenever
     GITHUB_TOKEN is set).  Powered by your GitHub Copilot Pro / Free
     subscription — no additional API key required.

     The agent calls the GitHub Models OpenAI-compatible endpoint:
       https://models.inference.ai.azure.com

     Usage in Codespaces (GITHUB_TOKEN is injected automatically):
       python run_agent.py \\
           --dataset data/sample_dataset.xlsx \\
           --target target

     Usage locally (create a PAT with models:read scope):
       export GITHUB_TOKEN=ghp_...
       python run_agent.py --dataset data/sample_dataset.xlsx --target target

     Choose a different model (default: gpt-4o-mini):
       python run_agent.py --model gpt-4o ...
       # Or: export GITHUB_MODELS_MODEL=gpt-4o

  2. **On-premise vllm** (or any OpenAI-compatible server).
     Pass --base-url to override the endpoint and optionally --api-key.

       python run_agent.py \\
           --base-url http://localhost:8000/v1 \\
           --model mistralai/Mistral-7B-Instruct-v0.3 \\
           --api-key not-needed \\
           --dataset data/sample_dataset.xlsx \\
           --target target

Environment variables:
    GITHUB_TOKEN          — GitHub PAT with models:read scope (auto-set in Codespaces)
    GITHUB_MODELS_MODEL   — GitHub Models model name (default: gpt-4o-mini)
    VLLM_BASE_URL         — vllm/custom endpoint (overrides GitHub Models)
    VLLM_MODEL            — model name for vllm
    VLLM_API_KEY          — API key for vllm (default: not-needed)
    DATASET_PATH          — path to Excel/CSV dataset
    TARGET_COLUMN         — name of the target column
    MAX_ITERATIONS        — number of experiment iterations (default: 100)
    LLM_TEMPERATURE       — sampling temperature (default: 0.7)
    LLM_MAX_TOKENS        — max tokens in LLM response (default: 4096)
"""

import argparse
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# GitHub Models constants
# ---------------------------------------------------------------------------

GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com"
GITHUB_MODELS_DEFAULT_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Defaults  (resolved at import time so CLI can override)
# ---------------------------------------------------------------------------

_github_token = os.environ.get("GITHUB_TOKEN", "")
_use_github_models = bool(_github_token) and "VLLM_BASE_URL" not in os.environ

# Backend endpoint & key
DEFAULT_BASE_URL = os.environ.get(
    "VLLM_BASE_URL",
    GITHUB_MODELS_ENDPOINT if _use_github_models else "http://localhost:8000/v1",
)
DEFAULT_MODEL = os.environ.get(
    "VLLM_MODEL",
    os.environ.get("GITHUB_MODELS_MODEL", GITHUB_MODELS_DEFAULT_MODEL)
    if _use_github_models
    else "",
)
DEFAULT_API_KEY = (
    _github_token
    if _use_github_models
    else os.environ.get("VLLM_API_KEY", "not-needed")
)

DEFAULT_DATASET = os.environ.get("DATASET_PATH", "data/sample_dataset.xlsx")
DEFAULT_TARGET = os.environ.get("TARGET_COLUMN", "target")
DEFAULT_MAX_ITER = int(os.environ.get("MAX_ITERATIONS", "100"))
DEFAULT_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "4096"))

RESULTS_FILE = Path("results.tsv")
TRAIN_FILE = Path("train.py")
PROGRAM_FILE = Path("program.md")
LOG_FILE = Path("run.log")
TRAIN_TIMEOUT = 600  # seconds — hard kill after 10 min

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def extract_code_block(text: str) -> str:
    """Pull the first ```python … ``` (or bare ``` … ```) block from an LLM reply."""
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: assume the whole reply is code
    return text.strip()


def parse_metrics(log_text: str) -> dict:
    """
    Parse the standardised metric block from train.py stdout.

    Expected lines (after '---'):
        val_rmse:         0.456789
        val_mae:          0.321456
        val_r2:           0.834567
        train_seconds:    12.3
        model_type:       random_forest
    """
    metrics = {}
    for line in log_text.splitlines():
        m = re.match(r"^(val_rmse|val_mae|val_r2|train_rmse|train_r2|"
                     r"test_rmse|test_r2|loo_rmse|loo_r2|train_seconds|model_type|n_features)"
                     r"\s*:\s*(.+)$", line.strip())
        if m:
            key, val = m.group(1), m.group(2).strip()
            try:
                metrics[key] = float(val)
            except ValueError:
                metrics[key] = val
    return metrics


def git_short_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "0000000"


def git_commit(message: str):
    subprocess.run(["git", "add", "train.py"], check=False)
    subprocess.run(["git", "commit", "-m", message], check=False)


def git_revert():
    """Revert train.py to HEAD (discard current changes without a new commit)."""
    subprocess.run(["git", "checkout", "HEAD", "--", "train.py"], check=False)


def ensure_results_file():
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(
            "commit\tval_rmse\tval_r2\ttrain_seconds\tstatus\tdescription\n",
            encoding="utf-8"
        )


def append_result(commit: str, metrics: dict, status: str, description: str):
    val_rmse = metrics.get("val_rmse", 9999.0)
    val_r2 = metrics.get("val_r2", 0.0)
    train_secs = metrics.get("train_seconds", 0.0)
    line = (
        f"{commit}\t{val_rmse:.6f}\t{val_r2:.6f}\t"
        f"{train_secs:.1f}\t{status}\t{description}\n"
    )
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    print(f"  Logged: {line.strip()}")


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    program_text = read_file(PROGRAM_FILE)
    return (
        "You are an expert machine learning researcher specialising in "
        "regression hyperparameter optimisation.\n\n"
        + program_text
    )


def build_user_prompt(train_code: str, results_text: str, crash_log: str = "") -> str:
    prompt = textwrap.dedent(f"""
        Here is the current `train.py`:

        ```python
        {train_code}
        ```

        Current experiment history (`results.tsv`):
        ```
        {results_text if results_text else "(empty — this is the first run)"}
        ```
    """).strip()

    if crash_log:
        prompt += f"\n\nThe last run **crashed**. Here is the tail of the log:\n```\n{crash_log}\n```\nPlease fix the code."
    else:
        prompt += (
            "\n\nBased on the experiment history, propose and implement the next "
            "improvement to reduce `val_rmse`. "
            "Respond with the **complete** modified `train.py` file only "
            "(no explanations, just the Python code)."
        )

    return prompt


def ask_llm(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call the LLM and return its text response."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(dataset: str, target: str) -> tuple[str, dict]:
    """
    Run `python train.py` and return (log_text, metrics_dict).
    """
    env = os.environ.copy()
    env["DATASET_PATH"] = dataset
    env["TARGET_COLUMN"] = target

    LOG_FILE.write_text("", encoding="utf-8")

    try:
        result = subprocess.run(
            [sys.executable, "train.py"],
            capture_output=True,
            text=True,
            timeout=TRAIN_TIMEOUT,
            env=env,
        )
        log_text = result.stdout + "\n" + result.stderr
    except subprocess.TimeoutExpired:
        log_text = f"ERROR: training timed out after {TRAIN_TIMEOUT}s\n"

    LOG_FILE.write_text(log_text, encoding="utf-8")
    metrics = parse_metrics(log_text)
    return log_text, metrics


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_agent(
    base_url: str,
    model: str,
    api_key: str,
    dataset: str,
    target: str,
    max_iterations: int,
    temperature: float,
    max_tokens: int,
):
    backend = (
        "GitHub Models (Copilot Pro)"
        if base_url == GITHUB_MODELS_ENDPOINT
        else f"vllm / custom  ({base_url})"
    )
    print(f"\n{'='*60}")
    print("  autoresearch_regression — autonomous agent")
    print(f"  Backend  : {backend}")
    print(f"  LLM      : {model}")
    print(f"  Dataset  : {dataset}  (target='{target}')")
    print(f"  Max iters: {max_iterations}")
    print(f"{'='*60}\n")

    client = OpenAI(base_url=base_url, api_key=api_key)
    ensure_results_file()

    best_rmse = float("inf")
    crash_streak = 0
    system_prompt = build_system_prompt()

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'─'*60}")
        print(f"  Iteration {iteration}/{max_iterations}  (best val_rmse so far: {best_rmse:.6f})")
        print(f"{'─'*60}")

        train_code = read_file(TRAIN_FILE)
        results_text = read_file(RESULTS_FILE)
        crash_log = ""

        # Build prompt (include crash info if the previous run failed)
        if crash_streak > 0:
            crash_log = read_file(LOG_FILE)[-3000:]  # last 3000 chars of log

        user_prompt = build_user_prompt(train_code, results_text, crash_log)

        # Ask the LLM for the next experiment
        print("  Querying LLM …")
        try:
            llm_reply = ask_llm(
                client, model, system_prompt, user_prompt, temperature, max_tokens
            )
        except Exception as exc:
            print(f"  LLM error: {exc}")
            time.sleep(5)
            continue

        new_code = extract_code_block(llm_reply)

        # Basic sanity check: must contain "from prepare import" or "import prepare"
        if "prepare" not in new_code:
            print("  LLM reply did not contain valid train.py code — skipping.")
            crash_streak += 1
            if crash_streak >= 3:
                print("  Too many consecutive bad replies. Reverting to last good state.")
                git_revert()
                crash_streak = 0
            continue

        # Write new train.py
        TRAIN_FILE.write_text(new_code, encoding="utf-8")
        print("  train.py updated.")

        # Commit the change
        git_commit(f"experiment iteration {iteration}")
        commit = git_short_hash()

        # Run the experiment
        print("  Running experiment …")
        t_start = time.perf_counter()
        log_text, metrics = run_experiment(dataset, target)
        elapsed = time.perf_counter() - t_start

        val_rmse = metrics.get("val_rmse")
        model_type = metrics.get("model_type", "unknown")

        if val_rmse is None:
            # Crashed
            crash_streak += 1
            print(f"  CRASH (streak={crash_streak}). Tail of log:")
            print(textwrap.indent(log_text[-1000:], "    "))
            append_result(commit, {}, "crash", f"crash at iteration {iteration}")
            if crash_streak >= 3:
                print("  Reverting train.py after 3 consecutive crashes.")
                git_revert()
                crash_streak = 0
            continue

        crash_streak = 0
        print(f"  val_rmse={val_rmse:.6f}  val_r2={metrics.get('val_r2', 0):.4f}"
              f"  model={model_type}  elapsed={elapsed:.1f}s")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            status = "keep"
            print(f"  ✓ New best! val_rmse={best_rmse:.6f}")
        else:
            status = "discard"
            print(f"  ✗ No improvement. Reverting.")
            git_revert()

        desc = f"{model_type} iter={iteration}"
        append_result(commit, metrics, status, desc)

    print(f"\n{'='*60}")
    print(f"  Run complete. Best val_rmse: {best_rmse:.6f}")
    print(f"  Results saved to {RESULTS_FILE}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Autonomous regression research agent — "
            "uses GitHub Models (Copilot Pro) by default when GITHUB_TOKEN is set, "
            "or any OpenAI-compatible endpoint via --base-url."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL,
        help=(
            "LLM endpoint URL.  Defaults to GitHub Models when GITHUB_TOKEN is set, "
            "otherwise http://localhost:8000/v1 (vllm).  "
            "GitHub Models: https://models.inference.ai.azure.com"
        ),
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=(
            "Model name.  GitHub Models examples: gpt-4o, gpt-4o-mini, "
            "Meta-Llama-3.1-70B-Instruct.  "
            "vllm example: mistralai/Mistral-7B-Instruct-v0.3"
        ),
    )
    parser.add_argument(
        "--api-key", default=None,
        help=(
            "API key.  For GitHub Models this is your GITHUB_TOKEN (read from env "
            "automatically).  For vllm leave as 'not-needed'."
        ),
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET,
        help="Path to the Excel (.xlsx) or CSV dataset",
    )
    parser.add_argument(
        "--target", default=DEFAULT_TARGET,
        help="Name of the target column in the dataset",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=DEFAULT_MAX_ITER,
        help="Maximum number of experiment iterations",
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help="LLM sampling temperature",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help="Maximum tokens in LLM response",
    )
    args = parser.parse_args()

    if not args.model:
        parser.error(
            "No model specified.  Either:\n"
            "  • Set GITHUB_TOKEN to use GitHub Models automatically, or\n"
            "  • Pass --model (e.g. --model gpt-4o-mini), or\n"
            "  • Set VLLM_MODEL for an on-premise vllm server."
        )

    # Resolve api-key: CLI > env > fallback
    api_key = args.api_key or DEFAULT_API_KEY

    if args.base_url == GITHUB_MODELS_ENDPOINT and api_key in ("not-needed", ""):
        parser.error(
            "GitHub Models requires a GitHub token.\n"
            "  In Codespaces this is set automatically (GITHUB_TOKEN).\n"
            "  Locally: export GITHUB_TOKEN=<your PAT with models:read scope>"
        )

    run_agent(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        dataset=args.dataset,
        target=args.target,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
