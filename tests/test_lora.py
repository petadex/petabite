"""Tests for the hand-rolled LoRA mechanism in model_module.lora.

The real ESM-C checkpoint is 300M parameters and has to be staged into HF_HOME,
so almost everything here runs against a toy module tree instead. The toy tree
reproduces the two fused ESM-C module classes exactly: same class names, same
attribute names, same forward. That is enough, because lora.py finds its targets
by class name and reads the base weights by attribute name.

Keeping the fake faithful is what makes the equivalence check meaningful. If the
wrappers in lora.py drift from the forward passes quoted in their docstrings,
these tests fail without any checkpoint being downloaded.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from petabite.model_module.lora import (
    LoraConfig,
    LoraLayerNormLinear,
    LoraLayerNormMLP,
    LoraLinear,
    apply_lora,
    assert_adapters_match,
    load_lora,
    save_lora,
    trainable_parameters,
)

# Small enough that a whole toy model builds in microseconds, large enough that
# a shape error cannot hide behind a square matrix.
HIDDEN = 12
FFN_HIDDEN = 20
N_BLOCKS = 3


# --------------------------------------------------------------------------
# Stand-ins for the two fused ESM-C modules
# --------------------------------------------------------------------------


class EsmcLayerNormLinear(nn.Module):
    """Stand-in for esm's fused LayerNorm + QKV projection.

    The class name has to match exactly: lora.py matches targets by
    ``type(child).__name__`` rather than by import, so a rename in `esm` shows
    up as a loud failure instead of a silent no-op. The forward is the one
    quoted in LoraLayerNormLinear's docstring.
    """

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.randn(d_out, d_in) * 0.1)
        self.layer_norm_weight = nn.Parameter(torch.randn(d_in) * 0.1 + 1.0)
        self.layer_norm_bias = nn.Parameter(torch.randn(d_in) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(
            x, (self.d_in,), self.layer_norm_weight, self.layer_norm_bias, self.eps
        )
        return F.linear(x, self.weight)


class EsmcLayerNormMLP(nn.Module):
    """Stand-in for esm's fused LayerNorm + SwiGLU FFN.

    ``fc1_weight`` is ``(2 * ffn_hidden, hidden)`` because SwiGLU splits the
    projection into a gate half and a value half. The forward is the one quoted
    in LoraLayerNormMLP's docstring.
    """

    def __init__(self, hidden: int, ffn_hidden: int) -> None:
        super().__init__()
        self.hidden_size = hidden
        self.eps = 1e-5
        self.fc1_weight = nn.Parameter(torch.randn(2 * ffn_hidden, hidden) * 0.1)
        self.fc2_weight = nn.Parameter(torch.randn(hidden, ffn_hidden) * 0.1)
        self.layer_norm_weight = nn.Parameter(torch.randn(hidden) * 0.1 + 1.0)
        self.layer_norm_bias = nn.Parameter(torch.randn(hidden) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(
            x,
            (self.hidden_size,),
            self.layer_norm_weight,
            self.layer_norm_bias,
            self.eps,
        )
        x = F.linear(x, self.fc1_weight)
        gate, value = x.chunk(2, dim=-1)
        x = F.silu(gate) * value
        return F.linear(x, self.fc2_weight)


class ToyAttention(nn.Module):
    """The two attention children lora.py looks for, under their real names."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.layernorm_qkv = EsmcLayerNormLinear(hidden, 3 * hidden)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)


class ToyBlock(nn.Module):
    def __init__(self, hidden: int, ffn_hidden: int) -> None:
        super().__init__()
        self.attn = ToyAttention(hidden)
        self.ffn = EsmcLayerNormMLP(hidden, ffn_hidden)


class ToyModel(nn.Module):
    """A stack of toy blocks with a forward that touches all three wrappers.

    The attention is not real attention: the value third of the QKV output is
    fed straight to out_proj. That is deliberate. These tests are about adapter
    injection and serialisation, and a real attention would only add arithmetic
    that the wrappers never see.
    """

    def __init__(self, n_blocks: int = N_BLOCKS, hidden: int = HIDDEN,
                 ffn_hidden: int = FFN_HIDDEN) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ToyBlock(hidden, ffn_hidden) for _ in range(n_blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            qkv = block.attn.layernorm_qkv(x)
            value = qkv.chunk(3, dim=-1)[2]
            x = x + block.attn.out_proj(value)
            x = x + block.ffn(x)
        return x


# Two tensors per LoraBranch (down.weight and up.weight); one branch for qkv,
# one for out_proj, two for the FFN, per block.
BRANCHES_PER_BLOCK = 4
EXPECTED_ADAPTER_TENSORS = N_BLOCKS * BRANCHES_PER_BLOCK * 2


@pytest.fixture
def toy_model() -> ToyModel:
    torch.manual_seed(0)
    return ToyModel()


@pytest.fixture
def cfg() -> LoraConfig:
    # Dropout off so a wrapper's output is deterministic in training mode too.
    return LoraConfig(r=4, alpha=8.0, dropout=0.0)


def _example_input(batch: int = 2, length: int = 5) -> torch.Tensor:
    return torch.randn(batch, length, HIDDEN)


# --------------------------------------------------------------------------
# Injection
# --------------------------------------------------------------------------


def test_apply_lora_reports_expected_counts(toy_model: ToyModel, cfg: LoraConfig) -> None:
    report = apply_lora(toy_model, cfg)

    assert report.qkv == N_BLOCKS
    assert report.out_proj == N_BLOCKS
    assert report.ffn == N_BLOCKS
    assert report.adapters == 3 * N_BLOCKS
    assert len(report.adapted_modules) == 3 * N_BLOCKS
    assert report.trainable_params > 0
    assert report.trainable_params < report.total_params


def test_apply_lora_replaces_modules_with_wrappers(
    toy_model: ToyModel, cfg: LoraConfig
) -> None:
    apply_lora(toy_model, cfg)

    for block in toy_model.blocks:
        assert isinstance(block.attn.layernorm_qkv, LoraLayerNormLinear)
        assert isinstance(block.attn.out_proj, LoraLinear)
        assert isinstance(block.ffn, LoraLayerNormMLP)
        # The base module must survive inside the wrapper, unwrapped.
        assert isinstance(block.attn.layernorm_qkv.base, EsmcLayerNormLinear)
        assert isinstance(block.ffn.base, EsmcLayerNormMLP)


def test_apply_lora_raises_when_nothing_matches(cfg: LoraConfig) -> None:
    """The regression test for the bug this module exists to fix.

    The previous implementation matched class names that `esm` had renamed, so
    it injected zero adapters and trained nothing while producing a perfectly
    healthy-looking loss curve. Adapting nothing must raise.
    """
    model = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))

    with pytest.raises(RuntimeError, match="adapted zero modules"):
        apply_lora(model, cfg)


def test_apply_lora_is_a_noop_at_initialisation(
    toy_model: ToyModel, cfg: LoraConfig
) -> None:
    """lora_B is zero-initialised, so the adapted model must match bit for bit.

    This is the property that makes training start from the pretrained weights
    exactly, and the property that the internal equivalence check relies on.
    """
    import copy

    x = _example_input()
    reference = copy.deepcopy(toy_model).eval()
    with torch.no_grad():
        expected = reference(x)

    apply_lora(toy_model, cfg)
    toy_model.eval()
    with torch.no_grad():
        actual = toy_model(x)

    assert torch.equal(actual, expected)


def test_only_lora_parameters_require_grad(
    toy_model: ToyModel, cfg: LoraConfig
) -> None:
    apply_lora(toy_model, cfg)

    trainable = [n for n, p in toy_model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in toy_model.named_parameters() if not p.requires_grad]

    assert len(trainable) == EXPECTED_ADAPTER_TENSORS
    assert all(".lora" in name for name in trainable)
    assert frozen, "the base weights should still be there, just frozen"
    assert not any(".lora" in name for name in frozen)


def test_dropping_a_target_drops_its_adapter(toy_model: ToyModel) -> None:
    """Each projection is a separate flag so an ablation can drop just one."""
    cfg = LoraConfig(r=4, dropout=0.0, targets=("qkv", "out_proj", "fc1"))
    apply_lora(toy_model, cfg)

    for block in toy_model.blocks:
        assert block.ffn.lora_fc1 is not None
        assert block.ffn.lora_fc2 is None

    # One fewer branch per block, so two fewer tensors per block.
    assert len(trainable_parameters(toy_model)) == EXPECTED_ADAPTER_TENSORS - 2 * N_BLOCKS


# --------------------------------------------------------------------------
# Save and load
# --------------------------------------------------------------------------


def _perturb_adapters(model: nn.Module) -> None:
    """Give every adapter a distinct non-zero value.

    Without this, every adapter tensor is either Kaiming-initialised or zero,
    and a round-trip that silently wrote zeros everywhere would still pass the
    equality checks below.
    """
    with torch.no_grad():
        for i, param in enumerate(p for p in model.parameters() if p.requires_grad):
            param.add_(torch.randn_like(param) + float(i))


def test_save_verify_load_round_trips_exactly(
    toy_model: ToyModel, cfg: LoraConfig, tmp_path: Path
) -> None:
    report = apply_lora(toy_model, cfg)
    _perturb_adapters(toy_model)

    save_lora(toy_model, tmp_path, cfg, report)

    # assert_adapters_match is the check that would have caught the original
    # bug: it compares the file against the live model, tensor by tensor.
    n_verified = assert_adapters_match(toy_model, tmp_path)
    assert n_verified == EXPECTED_ADAPTER_TENSORS

    torch.manual_seed(99)  # different base init, so only the load can match
    fresh = ToyModel()
    apply_lora(fresh, cfg)
    n_loaded = load_lora(fresh, tmp_path, strict=True)
    assert n_loaded == EXPECTED_ADAPTER_TENSORS

    original = trainable_parameters(toy_model)
    restored = trainable_parameters(fresh)
    assert set(original) == set(restored)
    for name, expected in original.items():
        assert torch.equal(restored[name], expected), f"{name} did not round-trip"


def test_saved_manifest_describes_the_checkpoint(
    toy_model: ToyModel, cfg: LoraConfig, tmp_path: Path
) -> None:
    import json

    report = apply_lora(toy_model, cfg)
    save_lora(toy_model, tmp_path, cfg, report)

    manifest = json.loads((tmp_path / "lora_manifest.json").read_text())
    assert manifest["config"]["r"] == cfg.r
    assert manifest["config"]["targets"] == list(cfg.targets)
    assert manifest["num_tensors"] == EXPECTED_ADAPTER_TENSORS
    assert manifest["injection"]["adapters"] == 3 * N_BLOCKS


def test_save_refuses_a_model_with_no_adapters(
    toy_model: ToyModel, cfg: LoraConfig, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="no trainable parameters"):
        save_lora(toy_model, tmp_path, cfg)


def test_load_lora_without_weights_raises(
    toy_model: ToyModel, cfg: LoraConfig, tmp_path: Path
) -> None:
    apply_lora(toy_model, cfg)

    with pytest.raises(FileNotFoundError):
        load_lora(toy_model, tmp_path / "nothing-here", strict=True)


def test_strict_load_rejects_a_rank_mismatch(
    toy_model: ToyModel, cfg: LoraConfig, tmp_path: Path
) -> None:
    """Same adapter names, different shapes: caught before the first forward."""
    apply_lora(toy_model, cfg)
    _perturb_adapters(toy_model)
    save_lora(toy_model, tmp_path, cfg)

    other = ToyModel()
    apply_lora(other, LoraConfig(r=cfg.r * 2, dropout=0.0))

    with pytest.raises(RuntimeError, match="shape mismatch"):
        load_lora(other, tmp_path, strict=True)


def test_strict_load_rejects_a_target_mismatch(
    toy_model: ToyModel, cfg: LoraConfig, tmp_path: Path
) -> None:
    """A checkpoint trained with more targets than the model has must not load."""
    apply_lora(toy_model, cfg)
    _perturb_adapters(toy_model)
    save_lora(toy_model, tmp_path, cfg)

    other = ToyModel()
    apply_lora(other, LoraConfig(r=cfg.r, dropout=0.0, targets=("qkv", "out_proj", "fc1")))

    with pytest.raises(RuntimeError, match="does not match the model"):
        load_lora(other, tmp_path, strict=True)


# --------------------------------------------------------------------------
# Tokenizer, which needs the real checkpoint
# --------------------------------------------------------------------------


def _esmc_is_cached() -> bool:
    """True when biohub/ESMC-300M has been staged into the HF cache.

    Compute nodes have no outbound network, so the checkpoint is pre-staged
    from a login node. Without it this test cannot run, and skipping is the
    honest outcome rather than a download attempt that would hang.
    """
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return (Path(hf_home) / "hub" / "models--biohub--ESMC-300M").is_dir()


@pytest.mark.skipif(
    not _esmc_is_cached(),
    reason="biohub/ESMC-300M is not staged in HF_HOME; pre-stage it from a login node",
)
def test_load_esmc_tokenizer_has_mask_and_pad_tokens() -> None:
    """AutoTokenizer cannot load this checkpoint, so the loader is hand-built.

    Masked-LM training needs both a mask token and a pad token, and a missing
    one used to surface deep inside the first batch.
    """
    from petabite.data_module.tokenization import load_esmc_tokenizer

    tokenizer = load_esmc_tokenizer("biohub/ESMC-300M", max_length=64)

    assert tokenizer.mask_token_id is not None
    assert tokenizer.pad_token_id is not None
    assert tokenizer.model_max_length == 64
    assert len(tokenizer) > 0
