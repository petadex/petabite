"""Tests for fused-LoRA injection, saving, loading, and the Trainer callback.

ESM-C's fused modules are matched by class *name*, so these fakes reproduce the
attribute surface that LoRALayerNormLinear / LoRALayerNormMLP read. That keeps the
tests offline — no biohub/ESMC-300M download.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from petabite.model_module.fused_lora import (
    FUSED_LORA_FILENAME,
    LoRALayerNormLinear,
    LoRALayerNormMLP,
    fused_lora_state_dict,
    inject_fused_lora,
    load_fused_lora,
    save_fused_lora,
)
from petabite.trainer_module import FusedLoRASaveCallback

func = torch.nn.functional

D_MODEL = 8
QKV_OUT = 3 * D_MODEL
FFN_HIDDEN = 16  # post-SwiGLU width; fc1 emits 2x this
N_BLOCKS = 2
RANK = 4

# Keys are relative to the fused modules' common ancestor, which for this fake is
# `blocks` — so block 0's QKV adapter is `0.attn.layernorm_qkv.…`, not `blocks.0.…`.
_QKV_A_KEY = "0.attn.layernorm_qkv.lora_A.weight"


class _PyTorchLayerNormLinear(nn.Module):
    """Stand-in for ESM-C's fused LayerNorm+QKV projection."""

    def __init__(self) -> None:
        super().__init__()
        self.d_in = D_MODEL
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.randn(QKV_OUT, D_MODEL))
        self.layer_norm_weight = nn.Parameter(torch.ones(D_MODEL))
        self.layer_norm_bias = nn.Parameter(torch.zeros(D_MODEL))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = func.layer_norm(
            x, (self.d_in,), self.layer_norm_weight, self.layer_norm_bias, self.eps
        )
        return func.linear(x_n, self.weight)


class _PyTorchLayerNormMLP(nn.Module):
    """Stand-in for ESM-C's fused LayerNorm+SwiGLU FFN."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = D_MODEL
        self.eps = 1e-5
        self.fc1_weight = nn.Parameter(torch.randn(2 * FFN_HIDDEN, D_MODEL))
        self.fc2_weight = nn.Parameter(torch.randn(D_MODEL, FFN_HIDDEN))
        self.layer_norm_weight = nn.Parameter(torch.ones(D_MODEL))
        self.layer_norm_bias = nn.Parameter(torch.zeros(D_MODEL))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = func.layer_norm(
            x, (self.hidden_size,), self.layer_norm_weight, self.layer_norm_bias, self.eps
        )
        x1, x2 = func.linear(x_n, self.fc1_weight).chunk(2, dim=-1)
        return func.linear(func.silu(x1) * x2, self.fc2_weight)


class _Attn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layernorm_qkv = _PyTorchLayerNormLinear()
        self.out_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attn()
        self.ffn = _PyTorchLayerNormMLP()


class _FakeESMC(nn.Module):
    """Two-block stand-in mirroring ESM-C's module nesting."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(N_BLOCKS))


class _PeftLike(nn.Module):
    """Mimics get_peft_model's ``base_model.model.<...>`` renaming."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = base


@pytest.fixture
def injected() -> _FakeESMC:
    torch.manual_seed(0)
    model = _FakeESMC()
    inject_fused_lora(model, r=RANK, alpha=8.0, dropout=0.0)
    return model


def test_injects_both_module_types():
    model = _FakeESMC()
    counts = inject_fused_lora(model, r=RANK, alpha=8.0, dropout=0.0)
    assert counts == {"qkv": N_BLOCKS, "ffn": N_BLOCKS}
    assert isinstance(model.blocks[0].attn.layernorm_qkv, LoRALayerNormLinear)
    assert isinstance(model.blocks[0].ffn, LoRALayerNormMLP)


def test_raises_when_no_fused_modules_found():
    """The transformer-engine case: class names differ, nothing matches."""

    class Plain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(D_MODEL, D_MODEL)

    with pytest.raises(RuntimeError, match="found no fused modules"):
        inject_fused_lora(Plain(), r=RANK, alpha=8.0, dropout=0.0)


def test_injection_is_identity_at_init(injected: _FakeESMC):
    """lora_B is zero-initialized, so the wrapper must reproduce the base forward.

    This also checks the wrappers re-implement the fused forward correctly — a bug in
    the LayerNorm or SwiGLU re-derivation would show up as a mismatch here.
    """
    injected.eval()
    x = torch.randn(2, 5, D_MODEL)
    block = injected.blocks[0]
    qkv = block.attn.layernorm_qkv
    assert torch.allclose(qkv(x), qkv.base(x), atol=1e-6)
    assert torch.allclose(block.ffn(x), block.ffn.base(x), atol=1e-6)


def test_state_dict_covers_all_adapters_and_excludes_base(injected: _FakeESMC):
    state = fused_lora_state_dict(injected)
    # 2 tensors per QKV adapter + 4 per FFN adapter, per block.
    assert len(state) == N_BLOCKS * (2 + 4)
    assert not any(".base." in k for k in state)
    expected_params = N_BLOCKS * (
        (RANK * D_MODEL + QKV_OUT * RANK)
        + (RANK * D_MODEL + 2 * FFN_HIDDEN * RANK + RANK * FFN_HIDDEN + D_MODEL * RANK)
    )
    assert sum(t.numel() for t in state.values()) == expected_params


def test_save_load_round_trip(injected: _FakeESMC, tmp_path):
    # Train-like perturbation: zero-init lora_B means a fresh model differs.
    with torch.no_grad():
        for name, param in injected.named_parameters():
            if "lora_" in name:
                param.add_(torch.randn_like(param))
    trained = fused_lora_state_dict(injected)

    path = save_fused_lora(injected, tmp_path)
    assert path.name == FUSED_LORA_FILENAME

    torch.manual_seed(99)
    fresh = _FakeESMC()
    inject_fused_lora(fresh, r=RANK, alpha=8.0, dropout=0.0)
    assert not torch.allclose(
        fused_lora_state_dict(fresh)[_QKV_A_KEY], trained[_QKV_A_KEY]
    )

    n = load_fused_lora(fresh, tmp_path)
    assert n == len(trained)
    for key, tensor in trained.items():
        assert torch.equal(fused_lora_state_dict(fresh)[key], tensor)


def test_keys_are_peft_prefix_independent(injected: _FakeESMC, tmp_path):
    """Weights saved from a PEFT-wrapped model load into a bare backbone."""
    with torch.no_grad():
        for name, param in injected.named_parameters():
            if "lora_" in name:
                param.add_(1.0)
    wrapped = _PeftLike(injected)
    save_fused_lora(wrapped, tmp_path)

    bare = _FakeESMC()
    inject_fused_lora(bare, r=RANK, alpha=8.0, dropout=0.0)
    assert load_fused_lora(bare, tmp_path) == N_BLOCKS * 6


def test_keys_survive_different_nesting_depth(injected: _FakeESMC, tmp_path):
    """Stage 1 trains ESMCForMaskedLM (blocks under `esmc.`), Stage 2 loads ESMCModel.

    Keys are anchored on the fused modules' common ancestor, so the extra parent on
    the save side must not matter.
    """

    class _ForMaskedLM(nn.Module):
        def __init__(self, backbone: nn.Module) -> None:
            super().__init__()
            self.esmc = backbone
            self.lm_head = nn.Linear(D_MODEL, 4)

    with torch.no_grad():
        for name, param in injected.named_parameters():
            if "lora_" in name:
                param.add_(0.5)
    save_fused_lora(_ForMaskedLM(injected), tmp_path)

    bare = _FakeESMC()
    inject_fused_lora(bare, r=RANK, alpha=8.0, dropout=0.0)
    assert load_fused_lora(bare, tmp_path) == N_BLOCKS * 6
    for key, tensor in fused_lora_state_dict(injected).items():
        assert torch.equal(fused_lora_state_dict(bare)[key], tensor)


def test_double_injection_is_rejected(injected: _FakeESMC):
    with pytest.raises(RuntimeError, match="already applied"):
        inject_fused_lora(injected, r=RANK, alpha=8.0, dropout=0.0)


def test_non_strict_load_ignores_foreign_keys(injected: _FakeESMC, tmp_path):
    """A stray key must not be written into a base weight."""
    from safetensors.torch import save_file

    state = fused_lora_state_dict(injected)
    state["0.attn.layernorm_qkv.base.weight"] = torch.zeros(QKV_OUT, D_MODEL)
    save_file(state, str(tmp_path / FUSED_LORA_FILENAME))

    before = injected.blocks[0].attn.layernorm_qkv.base.weight.clone()
    load_fused_lora(injected, tmp_path, strict=False)
    assert torch.equal(injected.blocks[0].attn.layernorm_qkv.base.weight, before)


def test_rank_mismatch_is_rejected(injected: _FakeESMC, tmp_path):
    save_fused_lora(injected, tmp_path)
    other = _FakeESMC()
    inject_fused_lora(other, r=RANK * 2, alpha=8.0, dropout=0.0)
    with pytest.raises(RuntimeError, match="Shape mismatch"):
        load_fused_lora(other, tmp_path)


def test_missing_checkpoint_raises(injected: _FakeESMC, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fused_lora(injected, tmp_path)


def test_save_without_injection_raises(tmp_path):
    with pytest.raises(RuntimeError, match="no fused-LoRA modules"):
        save_fused_lora(_FakeESMC(), tmp_path)


# --- FusedLoRASaveCallback -------------------------------------------------------


def _args(output_dir, should_save: bool = True) -> SimpleNamespace:
    return SimpleNamespace(output_dir=str(output_dir), should_save=should_save)


def _state(global_step: int = 0) -> SimpleNamespace:
    return SimpleNamespace(global_step=global_step)


def _perturbed_model() -> _FakeESMC:
    """An injected model whose adapters are no longer at their zero-init values."""
    torch.manual_seed(3)
    model = _FakeESMC()
    inject_fused_lora(model, r=RANK, alpha=8.0, dropout=0.0)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.add_(torch.randn_like(param))
    return model


def test_on_save_writes_into_checkpoint_dir(injected: _FakeESMC, tmp_path):
    FusedLoRASaveCallback().on_save(
        _args(tmp_path), _state(global_step=500), None, model=injected
    )
    assert (tmp_path / "checkpoint-500" / FUSED_LORA_FILENAME).is_file()


def test_on_save_skipped_on_non_main_process(injected: _FakeESMC, tmp_path):
    """Every rank runs callbacks; only the main process may write the file."""
    FusedLoRASaveCallback().on_save(
        _args(tmp_path, should_save=False), _state(global_step=500), None, model=injected
    )
    assert not (tmp_path / "checkpoint-500").exists()


def test_on_save_does_not_raise_on_failure(tmp_path):
    """A checkpoint-side failure must not kill a long training run."""
    FusedLoRASaveCallback().on_save(
        _args(tmp_path), _state(global_step=1), None, model=_FakeESMC()
    )
    assert not (tmp_path / "checkpoint-1").exists()


def test_fresh_run_loads_nothing(injected: _FakeESMC, tmp_path):
    before = fused_lora_state_dict(injected)
    FusedLoRASaveCallback(resume_from_checkpoint=None).on_train_begin(
        _args(tmp_path), _state(), None, model=injected
    )
    after = fused_lora_state_dict(injected)
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_resume_restores_fused_weights(tmp_path):
    trained = _perturbed_model()
    ckpt = tmp_path / "checkpoint-100"
    save_fused_lora(trained, ckpt)
    expected = fused_lora_state_dict(trained)

    fresh = _FakeESMC()
    inject_fused_lora(fresh, r=RANK, alpha=8.0, dropout=0.0)
    FusedLoRASaveCallback(resume_from_checkpoint=str(ckpt)).on_train_begin(
        _args(tmp_path), _state(global_step=100), None, model=fresh
    )

    restored = fused_lora_state_dict(fresh)
    for key, tensor in expected.items():
        assert torch.equal(restored[key], tensor)


def test_resume_true_picks_latest_checkpoint(tmp_path):
    """`True` must resolve the same checkpoint Trainer would pick."""
    older = _perturbed_model()
    save_fused_lora(older, tmp_path / "checkpoint-10")
    newer = _perturbed_model()
    with torch.no_grad():
        for name, param in newer.named_parameters():
            if "lora_" in name:
                param.mul_(-1.0)
    save_fused_lora(newer, tmp_path / "checkpoint-200")

    fresh = _FakeESMC()
    inject_fused_lora(fresh, r=RANK, alpha=8.0, dropout=0.0)
    FusedLoRASaveCallback(resume_from_checkpoint=True).on_train_begin(
        _args(tmp_path), _state(global_step=200), None, model=fresh
    )

    restored = fused_lora_state_dict(fresh)
    for key, tensor in fused_lora_state_dict(newer).items():
        assert torch.equal(restored[key], tensor)


def test_resume_without_fused_file_raises(injected: _FakeESMC, tmp_path):
    """A pre-fix checkpoint must refuse to resume, not silently zero the adapters."""
    ckpt = tmp_path / "checkpoint-42"
    ckpt.mkdir()
    (ckpt / "adapter_model.safetensors").write_bytes(b"")  # PEFT adapter only
    with pytest.raises(FileNotFoundError, match=FUSED_LORA_FILENAME):
        FusedLoRASaveCallback(resume_from_checkpoint=str(ckpt)).on_train_begin(
            _args(tmp_path), _state(global_step=42), None, model=injected
        )


def test_resume_without_fused_file_can_be_overridden(injected: _FakeESMC, tmp_path):
    ckpt = tmp_path / "checkpoint-42"
    ckpt.mkdir()
    callback = FusedLoRASaveCallback(
        resume_from_checkpoint=str(ckpt), require_on_resume=False
    )
    callback.on_train_begin(_args(tmp_path), _state(global_step=42), None, model=injected)
