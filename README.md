# Petabite

ESM-C fine-tuning for **PETase activity prediction** with an uncertainty-based
**active learning** loop.

## Install

```bash
uv venv
uv sync --extra dev
```

## Layout

- `src/petabite/` — data, model, trainer, active_learning, utils modules
- `run/conf/` — Hydra configs
- `run/pipeline/` — train / predict / active_learning entrypoints
- `data/sample_petase.csv` — tiny synthetic example (`sequence,label,id`)

## Usage

```bash
# Train (LoRA fine-tune ESM-C)
uv run python run/pipeline/train.py

# Predict on a CSV
uv run python run/pipeline/predict.py data.csv=path/to/seqs.csv

# Run one active-learning round
uv run python run/pipeline/active_learning.py
```

## Status

First-commit scope is **interfaces + stubs**. Registries, configs, and class
skeletons are wired and importable; ESM-C loading, training, and MC-dropout
raise `NotImplementedError` with TODO markers to be filled in.
