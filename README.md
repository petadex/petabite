# Petabite

Masked-LM LoRA fine-tuning of ESM-C on the Logan-derived PETadex, for iGEM
Toronto 2026 (project PetaBite).

The corpus is
[`petadex/catalytic-orfs-90pid`](https://huggingface.co/datasets/petadex/catalytic-orfs-90pid),
the 90%-identity centroids of the plastic-degrading enzyme homologs mined from
the Logan planetary-scale SRA assembly. Continued pretraining on it produces
PlasticESM, a domain-adapted backbone. Only LoRA adapters train; the backbone
stays frozen.

This repo does MLM fine-tuning and nothing else. Downstream activity prediction
lives outside it.

## Setup

The environment is module-loaded plus a venv, not `uv`:

```bash
source activate_petabite.sh          # module loads + ~/petabite venv
export PYTHONNOUSERSITE=1
export HF_HOME=/project/6122149/purav/hf_cache
pip install -e .                     # required: makes `python -m petabite.*` work
```

Without the editable install, `python -m petabite.mlm_finetune` fails with
`ModuleNotFoundError: No module named 'petabite'`.

Do not set `PYTHONPATH` on this cluster. Setting it drops the module-provided
site-packages from `sys.path`, and `numpy`, `packaging` and `pyarrow` disappear.

`esm>=3.4.1` is required, not optional. The `biohub/ESMC-300M` checkpoint ships
a remote-code shim whose `from_pretrained` returns an `esm` model, and its
loader refuses anything older.

## Stage the assets before submitting a job

```bash
python scripts/prestage_assets.py    # run on a login node
```

Downloads the checkpoint and the corpus into `HF_HOME` and loads each once so
the remote modeling code is cached too. Jobs then run with `HF_HUB_OFFLINE=1`.

## Pre-tokenize the corpus

Assembling 18,172,960 rows and tokenizing them takes minutes at the start of
every job. Do it once instead:

```bash
# small set for smoke tests
python scripts/pretokenize_corpus.py --out data/tokenized-smoke --limit 8192 --max-shards 1

# the full corpus
python scripts/pretokenize_corpus.py --out /project/6122149/purav/petabite-tokenized/90pid
```

Point `data.tokenized_path` at the result and startup becomes instant.

## Train

```bash
sbatch scripts/smoke_lora.sbatch     # short end-to-end check on one L40S
python -m petabite.mlm_finetune      # full run, config/mlm.yaml
python -m petabite.mlm_finetune trainer.max_steps=100 data.max_train_samples=50000
```

Any config key can be overridden on the command line. To resume, set
`trainer.resume_from_checkpoint` to `true` for the latest checkpoint in
`output_dir`, or to an explicit path.

## Verify a checkpoint

Training saves `lora_adapters.safetensors` plus a `lora_manifest.json`, and
verifies the file against the live model before exiting. To re-check one later:

```bash
python scripts/verify_checkpoint.py outputs/esmc-mlm-lora-bs16-r32
```

It asks three separate questions: is the checkpoint complete, does it reload
into a fresh model exactly, and did the adapters actually move away from their
zero initialisation. The last one matters because an untrained adapter
round-trips perfectly and is still worthless.

## Weights and Biases

Runs log to the `petabite-esmc` project. Compute nodes on this cluster do have
outbound network, so runs stream live and `wandb.mode` is `online`. For an
isolated node, set it to `offline` and upload afterwards from a login node with
`wandb sync wandb/offline-run-*`.

## Throughput

```bash
sbatch scripts/benchmark.sbatch      # baseline against tuned, same GPU
```

## Layout

```
src/petabite/
  mlm_finetune.py              the entrypoint
  data_module/
    tokenization.py            builds the ESM-C tokenizer from tokenizer.json
    tokenized.py               loads a pre-tokenized corpus
  model_module/
    lora.py                    the LoRA mechanism, save and load
  trainer_module/
    lora_callback.py           puts adapters in every checkpoint, restores on resume
  utils/
config/mlm.yaml
scripts/                       staging, pre-tokenizing, SLURM jobs, verification
tests/
```

## A note on the LoRA mechanism

ESM-C keeps the projections Transformer Engine likes to fuse, so only
`attn.out_proj` is an `nn.Linear`. PEFT can reach that one and nothing else.
This repo used to run PEFT for `out_proj` alongside a second hand-rolled
mechanism for the fused modules, which meant two save paths: PEFT's
`get_peft_model_state_dict` keeps only keys containing both `lora_` and
`.default.`, so a checkpoint held the `out_proj` adapters and nothing else,
12.5% of what had been trained at r=32.

`model_module/lora.py` replaces both halves with one mechanism covering all four
projections, and owns its own save and load. Three guards:

- `apply_lora` raises if it adapts nothing. `esm` has already renamed the fused
  classes once, from `_PyTorchLayerNormLinear`/`_PyTorchLayerNormMLP` to
  `EsmcLayerNormLinear`/`EsmcLayerNormMLP` in 3.4.1, and the previous code
  responded by silently training only `out_proj` while reporting a healthy loss
  curve.
- Each wrapper is checked against the module it replaced at injection time.
  `lora_B` is zero-initialised, so a correct wrapper is bit-identical to its
  base, and a drifted re-implementation fails immediately.
- Every save is followed by comparing the file to the live model.

Installing `transformer-engine` would swap the fused modules for TE's own
classes, which this file does not wrap. `apply_lora` will raise rather than
quietly skip them.
