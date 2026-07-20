# Petabite

LoRA masked-LM fine-tuning of [ESM-C 300M](https://huggingface.co/biohub/ESMC-300M) on
the Logan-derived PETadex, producing **PlasticESM** — a protein language model
domain-adapted to plastic-degrading enzymes.

This repo does MLM fine-tuning and nothing else. Activity prediction, inference, and
active learning are out of scope.

## Install

```bash
uv venv
uv pip install -e ".[dev]"
```

Note: `.venv` has no `pip`; use `uv pip`. Loading `biohub/ESMC-300M` needs a
`transformers` build that ships `models/esmc`.

## Usage

```bash
./train-lora.sh                                    # uses config/mlm.yaml
python -m petabite.mlm_finetune
python -m petabite.mlm_finetune data.max_train_samples=100000   # smoke test
python -m petabite.mlm_finetune trainer.resume_from_checkpoint=true
```

Every setting is a Hydra override; see `config/mlm.yaml`.

## Layout

```
config/mlm.yaml                     all hyperparameters
src/petabite/
  mlm_finetune.py                   entrypoint
  model_module/fused_lora.py        LoRA for ESM-C's fused QKV/FFN modules
  trainer_module/                   fused-LoRA checkpoint callback
  utils/                            seeding, env capture, logging, registry
```

## Why `fused_lora.py` exists

ESM-C fuses LayerNorm into its QKV projection (`_PyTorchLayerNormLinear`) and its
SwiGLU FFN (`_PyTorchLayerNormMLP`). Neither subclasses `nn.Linear`, so **PEFT cannot
see them** — `target_modules: [out_proj]` is the only thing PEFT can reach, and that is
just 12.5% of the adapter budget at r=32.

`fused_lora.py` wraps both fused modules by hand. PEFT's state-dict filter also drops
these weights at save time (they carry no `.default.` adapter segment), so
`FusedLoRASaveCallback` writes them to `fused_lora.safetensors` beside PEFT's adapter
and restores them on resume.

### Checking a run is healthy

| Signal | Expected at r=32 |
|---|---|
| Injection log | `Injected fused LoRA: 30 QKV adapters, 30 FFN adapters` |
| `print_trainable_parameters()` | ≈ 14.7M (1.8M means only `out_proj` is training) |
| `fused_lora.safetensors` | ≈ 51.6 MB, beside a ≈ 7.4 MB `adapter_model.safetensors` |

Injection raises if it matches zero modules — installing `transformer-engine` renames
the fused classes, which would otherwise silently train `out_proj` alone.

## Development

```bash
pytest          # 26 tests, no network or model download required
ruff check .
mypy src/
```
