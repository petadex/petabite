# Petabite: ESM-C PETase Activity Prediction + Active Learning — Design

**Date:** 2026-06-16
**Status:** Approved (design), pending implementation plan

## Goal

Boilerplate PyTorch repository to fine-tune **ESM-C** for **PETase activity
prediction** with a **pool-based active learning** loop for wet-lab-in-the-loop
campaigns.

## Decisions (confirmed)

| Decision | Choice |
|---|---|
| Fine-tuning method | **LoRA / PEFT** (backbone frozen, adapters + head trained) |
| Task type | **Regression and classification**, config-driven |
| ESM-C source | **HuggingFace `transformers`** (EvolutionaryScale ESM-C checkpoints) |
| Active learning | **Uncertainty-based** (MC-dropout), with random baseline; pluggable |
| First-commit scope | **Interfaces + stubs** — registries, configs, class skeletons with `NotImplementedError`/TODO; structurally complete, not yet runnable |
| Input data format | CSV with `sequence`, `label`, optional `id` |

## Stack

- `uv` for dependency/venv management
- Hydra + OmegaConf for configuration
- `transformers` (ESM-C) + `peft` (LoRA) + PyTorch
- Factory/Registry pattern throughout; files kept 200–400 lines
- Reproducibility: global seed util, env recording, Hydra config + `pip freeze`
  saved per run (per `experiment-reproducibility` rule)

## Repository layout

```
petabite/
├── pyproject.toml              # uv-managed deps + tool config (ruff, mypy, pytest)
├── README.md                   # purpose, install, usage, AL loop overview
├── .gitignore                  # (exists) + outputs/, caches
├── src/petabite/
│   ├── __init__.py
│   ├── data_module/
│   │   ├── __init__.py         # DatasetFactory, register_dataset, __all__
│   │   ├── dataset/
│   │   │   ├── __init__.py     # registry
│   │   │   └── petase_dataset.py   # PETaseDataset(cfg): CSV -> samples
│   │   ├── tokenization.py     # ESM-C tokenizer wrapper
│   │   ├── splits.py           # train/val/test + AL labeled/pool splitting
│   │   └── utils.py            # CSV/FASTA IO, dataset hashing
│   ├── model_module/
│   │   ├── __init__.py         # ModelFactory, BackboneFactory, HeadFactory
│   │   ├── backbone/
│   │   │   ├── __init__.py     # registry
│   │   │   └── esmc_backbone.py    # HF ESM-C load + LoRA wrap
│   │   ├── heads/
│   │   │   ├── __init__.py     # registry
│   │   │   ├── regression_head.py
│   │   │   └── classification_head.py
│   │   ├── model.py            # ESMCActivityModel(cfg): backbone + head
│   │   └── uncertainty.py      # MC-dropout / ensemble variance estimators
│   ├── trainer_module/
│   │   ├── __init__.py
│   │   ├── trainer.py          # HF Trainer wrapper, seeding, checkpointing
│   │   └── metrics.py          # reg: RMSE/R²/Spearman; clf: AUROC/F1/Acc
│   ├── active_learning/
│   │   ├── __init__.py         # AcquisitionFactory, register_acquisition
│   │   ├── acquisition/
│   │   │   ├── __init__.py     # registry
│   │   │   ├── base.py         # AcquisitionFunction interface
│   │   │   ├── uncertainty.py  # MC-dropout variance acquisition
│   │   │   └── random_baseline.py
│   │   ├── loop.py             # ActiveLearningLoop: round controller
│   │   └── selection.py        # top-K batch selection, wet-lab query export
│   └── utils/
│       ├── __init__.py
│       ├── registry.py         # generic Registry base
│       ├── seed.py             # set_seed()
│       ├── env.py              # log_environment(), pip freeze dump
│       └── logging.py          # logger setup
├── run/
│   ├── conf/
│   │   ├── config.yaml         # top-level Hydra compose
│   │   ├── data/petase.yaml
│   │   ├── model/esmc_lora.yaml
│   │   ├── trainer/default.yaml
│   │   └── active_learning/uncertainty.yaml
│   └── pipeline/
│       ├── train.py            # train + eval entrypoint
│       ├── predict.py          # batch inference on a CSV
│       └── active_learning.py  # run one or more AL rounds
├── tests/
│   ├── test_dataset.py
│   ├── test_model_forward.py
│   ├── test_acquisition.py
│   └── test_al_loop.py
├── data/
│   └── sample_petase.csv       # tiny synthetic example (sequence,label,id)
└── outputs/                    # gitignored; Hydra timestamped runs
```

## Components and interfaces

Each unit has one purpose, a config-driven constructor, and is testable in
isolation.

- **PETaseDataset(cfg)** — reads CSV (`sequence,label[,id]`), exposes
  `__len__`/`__getitem__` yielding tokenized inputs + label. Depends on
  tokenization + utils.
- **ESMCBackbone(cfg)** — loads HF ESM-C, wraps with LoRA via `peft`, freezes
  base weights. Returns pooled/per-residue embeddings. Stub: `NotImplementedError`.
- **Heads** — `RegressionHead`, `ClassificationHead`; registry-selected by
  `cfg.model.task`. Define output dim + loss.
- **ESMCActivityModel(cfg)** — composes backbone + head; `forward` returns
  logits/values + loss when labels present.
- **Uncertainty estimators** — MC-dropout sampling and ensemble variance over
  predictions; consumed by acquisition.
- **Trainer** — wraps HF Trainer: seeded, computes task metrics, saves
  checkpoints + resolved Hydra config + `pip freeze`.
- **AcquisitionFunction (base)** — `score(model, pool) -> scores`; implemented by
  `UncertaintyAcquisition` and `RandomAcquisition`.
- **ActiveLearningLoop** — orchestrates: train on labeled set → score pool →
  select top-K → export wet-lab query CSV → (after labels return) re-ingest →
  next round. State (labeled/pool indices, round number) persisted to `outputs/`.

## Data flow

1. CSV → `PETaseDataset` → ESM-C tokenizer → batched tensors.
2. `ESMCActivityModel` = LoRA-wrapped ESM-C + task head; `cfg.model.task`
   (`regression|classification`) selects head + loss (MSE / CrossEntropy).
3. `Trainer` trains (adapters + head only), evaluates, checkpoints with full
   reproducibility artifacts.
4. **AL loop:** small labeled set + unlabeled pool → train → uncertainty score
   pool → select top-K → write `outputs/<run>/query_round_N.csv` for wet-lab →
   ingest returned labels → repeat.

## Error handling

- Specific exceptions (`FileNotFoundError` for missing data/checkpoints,
  `ValueError` for invalid `task`/config), logged via module loggers.
- Config validation at construction time (e.g. unknown registry key →
  explicit error listing valid keys).

## Testing

- `test_dataset.py` — CSV parsing, tokenization output shapes, label dtype.
- `test_model_forward.py` — forward pass shape + loss with a tiny stand-in
  backbone (no weight download).
- `test_acquisition.py` — each acquisition ranks a synthetic pool correctly;
  random is reproducible under seed.
- `test_al_loop.py` — one AL round on synthetic data updates labeled/pool state
  and emits a query file.

Because first-commit scope is **interfaces + stubs**, tests assert
interface/contract behavior; stubs raising `NotImplementedError` are marked
`pytest.mark.skip`/`xfail` with TODO pointers until implemented.

## Out of scope (YAGNI for now)

- Full-finetune and frozen-embedding training paths (LoRA only; config leaves
  room to add later).
- Diversity/BADGE/EI acquisition functions (interface allows adding; not built).
- Distributed/multi-GPU training, hyperparameter sweeps.
- Real ESM-C weight download wiring beyond the loader stub.
```
