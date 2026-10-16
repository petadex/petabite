"""LoRA adapters for the ESM-C backbone.

Why this module exists
----------------------
ESM-C does not expose four separate projections per block. It keeps the ones
Transformer Engine likes to fuse:

    attn.layernorm_qkv   EsmcLayerNormLinear   LayerNorm + QKV, weight (3d, d)
    attn.out_proj        nn.Linear             d -> d, no bias
    ffn                  EsmcLayerNormMLP      LayerNorm + SwiGLU, fc1 and fc2

Only ``out_proj`` is an ``nn.Linear``, so PEFT can only ever reach that one.
An earlier version of this repo worked around that by running PEFT for
``out_proj`` and a second, hand-rolled mechanism for the fused modules. Two
mechanisms meant two save paths, and PEFT's ``get_peft_model_state_dict`` keeps
only keys that contain both ``lora_`` and ``.default.``. The hand-rolled
adapters matched the first test and failed the second, so every checkpoint
silently contained the ``out_proj`` adapters and nothing else.

This module replaces both halves with one mechanism that covers all four
projections and owns its own save and load. There is no PEFT in the path, so
there is no filter to fall foul of.

Faithfulness
------------
Two of the three wrappers have to re-run the base computation, because LoRA
needs a tensor the base module computes internally and does not return: the
post-LayerNorm activation for QKV and fc1, and the post-SwiGLU activation for
fc2. Re-implementing a forward pass is exactly how this code broke before, so
the re-implementations are not trusted. ``apply_lora`` runs
:func:`check_wrapper_equivalence` on one wrapper of each kind at injection
time. Because ``lora_B`` is zero-initialised, a correct wrapper is numerically
identical to the module it replaced, so any drift raises immediately rather
than producing a plausible loss curve.

The same property is what makes the adapters a no-op at step 0, which is the
standard LoRA initialisation and means training starts from the pretrained
model exactly.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

# The ESM-C module class names this file knows how to wrap. They are matched by
# name rather than by import because `esm` renamed them once already: they were
# `_PyTorchLayerNormLinear` and `_PyTorchLayerNormMLP` before esm 3.4.1. Name
# matching keeps the dependency loose, and MISSING_TARGETS below turns a rename
# into a loud failure instead of a silent no-op.
QKV_CLASS = "EsmcLayerNormLinear"
MLP_CLASS = "EsmcLayerNormMLP"

# Written next to the weights so a checkpoint can be loaded without the config
# that produced it, and so a mismatched rank is caught at load rather than at
# the first forward pass.
MANIFEST_NAME = "lora_manifest.json"
WEIGHTS_NAME = "lora_adapters.safetensors"


@dataclass
class LoraConfig:
    """Which projections to adapt, and how.

    ``targets`` names the four projections in the order they run inside a
    block. Dropping one is a legitimate ablation, which is why each is a
    separate flag rather than a single boolean.
    """

    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.01
    targets: tuple[str, ...] = ("qkv", "out_proj", "fc1", "fc2")
    # Checked at injection. Disable only if profiling shows it matters, which it
    # will not: it is one forward pass on a 2x4xd tensor.
    verify: bool = True

    @property
    def scaling(self) -> float:
        return self.alpha / self.r


# --------------------------------------------------------------------------
# The low-rank branch, shared by all three wrappers
# --------------------------------------------------------------------------


class LoraBranch(nn.Module):
    """One ``B @ A`` low-rank branch, zero-initialised at ``B``.

    Kept as its own module so that every adapter in the model has the same two
    parameter names, ``down.weight`` and ``up.weight``. That makes the saved
    state dict readable and makes the parameter count trivially checkable.
    """

    def __init__(self, d_in: int, d_out: int, cfg: LoraConfig,
                 dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.down = nn.Linear(d_in, cfg.r, bias=False, dtype=dtype, device=device)
        self.up = nn.Linear(cfg.r, d_out, bias=False, dtype=dtype, device=device)
        self.dropout = nn.Dropout(cfg.dropout)
        self.scaling = cfg.scaling

        # Kaiming on the down projection, zeros on the up projection. The zeros
        # are what make the adapter a no-op at step 0.
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(self.dropout(x))) * self.scaling


def _branch_for(weight: torch.Tensor, d_in: int, d_out: int,
                cfg: LoraConfig) -> LoraBranch:
    """Build a branch matching the dtype and device of the weight it adapts.

    Creating the branch in the base weight's dtype is not cosmetic. The old
    code built ``nn.Linear`` at the default fp32 while the base could be bf16,
    which raises in ``F.linear`` as soon as the backbone is loaded in half
    precision.
    """
    return LoraBranch(d_in, d_out, cfg, weight.dtype, weight.device)


# --------------------------------------------------------------------------
# The three wrappers, in the order they run inside a transformer block
# --------------------------------------------------------------------------


class LoraLayerNormLinear(nn.Module):
    """Adapts ``EsmcLayerNormLinear``, the fused LayerNorm + QKV projection.

    The base forward, copied from ``esm.models.esmc.layers``, is::

        x = F.layer_norm(x, (d_in,), layer_norm_weight, layer_norm_bias, eps)
        return F.linear(x, weight)

    LoRA attaches to the projection, so it needs the normalised activation,
    which the base does not return. Hence the re-implementation, and hence the
    equivalence check in ``apply_lora``.
    """

    def __init__(self, base: nn.Module, cfg: LoraConfig) -> None:
        super().__init__()
        self.base = base
        d_in = base.d_in
        d_out = base.weight.shape[0]
        self.lora = _branch_for(base.weight, d_in, d_out, cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base
        normalised = F.layer_norm(
            x,
            (base.d_in,),
            base.layer_norm_weight,
            base.layer_norm_bias,
            base.eps,
        )
        return F.linear(normalised, base.weight) + self.lora(normalised)


class LoraLinear(nn.Module):
    """Adapts a plain ``nn.Linear``, which is what ``attn.out_proj`` is.

    This is the one wrapper that does not re-implement anything. The base takes
    the same input the LoRA branch does, so the base module is simply called.
    """

    def __init__(self, base: nn.Linear, cfg: LoraConfig) -> None:
        super().__init__()
        self.base = base
        self.lora = _branch_for(
            base.weight, base.in_features, base.out_features, cfg
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora(x)


class LoraLayerNormMLP(nn.Module):
    """Adapts ``EsmcLayerNormMLP``, the fused LayerNorm + SwiGLU FFN.

    The base forward, copied from ``esm.models.esmc.layers``, is::

        x = F.layer_norm(x, (hidden_size,), layer_norm_weight, layer_norm_bias, eps)
        x = F.linear(x, fc1_weight)
        x1, x2 = x.chunk(2, dim=-1)
        x = F.silu(x1) * x2
        return F.linear(x, fc2_weight)

    ``fc1_weight`` is ``(2 * ffn_hidden, hidden)`` because SwiGLU splits it into
    a gate and a value half. ``fc2_weight`` is ``(hidden, ffn_hidden)``, so the
    fc2 adapter sees the post-SwiGLU activation, which is half the width of the
    fc1 output.

    Both adapters are optional, so an ablation can keep one and drop the other.
    """

    def __init__(self, base: nn.Module, cfg: LoraConfig) -> None:
        super().__init__()
        self.base = base
        hidden = base.hidden_size
        fc1_out = base.fc1_weight.shape[0]      # 2 * ffn_hidden
        fc2_in = base.fc2_weight.shape[1]       # ffn_hidden

        self.lora_fc1 = (
            _branch_for(base.fc1_weight, hidden, fc1_out, cfg)
            if "fc1" in cfg.targets else None
        )
        self.lora_fc2 = (
            _branch_for(base.fc2_weight, fc2_in, hidden, cfg)
            if "fc2" in cfg.targets else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base
        normalised = F.layer_norm(
            x,
            (base.hidden_size,),
            base.layer_norm_weight,
            base.layer_norm_bias,
            base.eps,
        )

        projected = F.linear(normalised, base.fc1_weight)
        if self.lora_fc1 is not None:
            projected = projected + self.lora_fc1(normalised)

        gate, value = projected.chunk(2, dim=-1)
        activated = F.silu(gate) * value

        out = F.linear(activated, base.fc2_weight)
        if self.lora_fc2 is not None:
            out = out + self.lora_fc2(activated)
        return out


# --------------------------------------------------------------------------
# Injection
# --------------------------------------------------------------------------


@dataclass
class InjectionReport:
    """What ``apply_lora`` actually did. Returned so callers can assert on it."""

    qkv: int = 0
    out_proj: int = 0
    ffn: int = 0
    trainable_params: int = 0
    total_params: int = 0
    adapted_modules: list[str] = field(default_factory=list)

    @property
    def adapters(self) -> int:
        return self.qkv + self.out_proj + self.ffn

    def summary(self) -> str:
        pct = 100.0 * self.trainable_params / max(1, self.total_params)
        return (
            f"{self.adapters} adapted modules "
            f"(qkv={self.qkv}, out_proj={self.out_proj}, ffn={self.ffn}); "
            f"{self.trainable_params:,} trainable of {self.total_params:,} "
            f"({pct:.3f}%)"
        )


def apply_lora(model: nn.Module, cfg: LoraConfig) -> InjectionReport:
    """Wrap every target projection in ``model``, freeze everything else.

    Raises if nothing was adapted. That is the single most important line in
    this file: the previous implementation matched class names that `esm` had
    renamed, so it injected zero adapters, returned counts nobody inspected,
    and trained nothing. A run that adapts nothing must fail loudly, not
    produce a healthy-looking loss curve.
    """
    report = InjectionReport()
    _wrap_recursive(model, cfg, report, prefix="")

    if report.adapters == 0:
        raise RuntimeError(
            "LoRA injection adapted zero modules. Expected to find "
            f"{QKV_CLASS!r} and {MLP_CLASS!r} in the model tree. The `esm` "
            "package has renamed these classes before; check the installed "
            "version against the names at the top of this module."
        )

    freeze_base(model)
    if cfg.verify:
        check_wrapper_equivalence(model)

    report.total_params = sum(p.numel() for p in model.parameters())
    report.trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info("LoRA: %s", report.summary())
    return report


def _wrap_recursive(module: nn.Module, cfg: LoraConfig,
                    report: InjectionReport, prefix: str) -> None:
    """Replace target children in place, depth first.

    A replaced child is not descended into: the base module lives on inside the
    wrapper and must not be wrapped a second time.
    """
    for name, child in list(module.named_children()):
        path = f"{prefix}{name}"
        class_name = type(child).__name__

        if class_name == QKV_CLASS and "qkv" in cfg.targets:
            setattr(module, name, LoraLayerNormLinear(child, cfg))
            report.qkv += 1
            report.adapted_modules.append(path)

        elif class_name == MLP_CLASS and ("fc1" in cfg.targets or "fc2" in cfg.targets):
            setattr(module, name, LoraLayerNormMLP(child, cfg))
            report.ffn += 1
            report.adapted_modules.append(path)

        elif name == "out_proj" and isinstance(child, nn.Linear) and "out_proj" in cfg.targets:
            setattr(module, name, LoraLinear(child, cfg))
            report.out_proj += 1
            report.adapted_modules.append(path)

        else:
            _wrap_recursive(child, cfg, report, prefix=f"{path}.")


def freeze_base(model: nn.Module) -> None:
    """Train the LoRA branches and nothing else.

    Membership is decided by walking the module tree for ``LoraBranch``
    instances rather than by matching parameter names. A substring test would
    make the set of trained tensors depend on what things happen to be called,
    which is the class of mistake this whole module exists to avoid.
    """
    for param in model.parameters():
        param.requires_grad = False
    for _, branch in _lora_branches(model):
        for param in branch.parameters():
            param.requires_grad = True


def _lora_branches(model: nn.Module) -> list[tuple[str, LoraBranch]]:
    """Every LoRA branch in the model, with its dotted module path."""
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LoraBranch)
    ]


def trainable_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    """The adapter tensors, keyed by full parameter name.

    This is the entire checkpoint. It is defined as "the parameters inside LoRA
    branches", not "the parameters that require grad", so unfreezing something
    in the base model cannot quietly add 300M parameters to the adapter file,
    and a model that was never adapted yields nothing at all rather than
    everything.
    """
    tensors: dict[str, torch.Tensor] = {}
    for module_name, branch in _lora_branches(model):
        for param_name, param in branch.named_parameters():
            tensors[f"{module_name}.{param_name}"] = param.detach()
    return tensors


# --------------------------------------------------------------------------
# Equivalence check
# --------------------------------------------------------------------------


@torch.no_grad()
def check_wrapper_equivalence(model: nn.Module, atol: float = 0.0) -> None:
    """Assert each wrapper still reproduces the module it replaced.

    With ``lora_B`` at zero the low-rank branch contributes exactly zero, so a
    correct wrapper matches its base bit for bit and ``atol=0`` is the right
    default. Dropout is disabled for the duration, since an active dropout mask
    would scale the base path's input in training mode.

    Only the first wrapper of each kind is checked. They are built by the same
    code from the same class, so a fault in one is a fault in all, and checking
    120 of them would cost 40x more for no extra information.
    """
    was_training = model.training
    model.eval()
    try:
        for kind, wrapper_type in (
            ("qkv", LoraLayerNormLinear),
            ("out_proj", LoraLinear),
            ("ffn", LoraLayerNormMLP),
        ):
            found = next(
                (m for m in model.modules() if isinstance(m, wrapper_type)), None
            )
            if found is None:
                continue
            _check_one(kind, found, atol)
    finally:
        model.train(was_training)


@torch.no_grad()
def _check_one(kind: str, wrapper: nn.Module, atol: float) -> None:
    """Run one wrapper and its base on the same input and compare."""
    base = wrapper.base
    weight = base.weight if hasattr(base, "weight") else base.fc1_weight
    d_in = _input_width(base)

    x = torch.randn(2, 4, d_in, dtype=weight.dtype, device=weight.device)
    expected = base(x)
    actual = wrapper(x)

    if actual.shape != expected.shape:
        raise RuntimeError(
            f"LoRA wrapper for {kind} changed the output shape: "
            f"{tuple(expected.shape)} -> {tuple(actual.shape)}"
        )

    max_diff = (actual - expected).abs().max().item()
    if max_diff > atol:
        raise RuntimeError(
            f"LoRA wrapper for {kind} is not equivalent to the module it "
            f"replaced at initialisation: max |diff| = {max_diff:.3e} "
            f"(tolerance {atol:.3e}). The re-implemented forward pass has "
            f"drifted from {type(base).__name__} in the installed `esm`."
        )
    logger.debug("LoRA wrapper for %s matches its base exactly", kind)


def _input_width(base: nn.Module) -> int:
    """Input width of a wrapped base module, whichever kind it is."""
    if hasattr(base, "d_in"):
        return base.d_in
    if hasattr(base, "hidden_size"):
        return base.hidden_size
    if isinstance(base, nn.Linear):
        return base.in_features
    raise TypeError(f"cannot determine input width of {type(base).__name__}")


# --------------------------------------------------------------------------
# Save and load
# --------------------------------------------------------------------------


def save_lora(model: nn.Module, directory: str | Path,
              cfg: LoraConfig, report: InjectionReport | None = None) -> Path:
    """Write the adapters and a manifest describing them.

    The manifest exists so that :func:`load_lora` can refuse a mismatched
    checkpoint instead of discovering the problem as a shape error deep in a
    forward pass, and so that a checkpoint found on disk months later is
    self-describing.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    tensors = trainable_parameters(model)
    if not tensors:
        raise RuntimeError(
            "refusing to save: the model has no trainable parameters, so there "
            "are no adapters to write"
        )

    # safetensors needs contiguous tensors on CPU.
    tensors = {name: t.contiguous().cpu() for name, t in tensors.items()}
    save_file(tensors, directory / WEIGHTS_NAME)

    manifest = {
        "format": "petabite-lora-v1",
        "config": {
            "r": cfg.r,
            "alpha": cfg.alpha,
            "dropout": cfg.dropout,
            "targets": list(cfg.targets),
        },
        "tensors": {name: list(t.shape) for name, t in tensors.items()},
        "num_tensors": len(tensors),
        "num_parameters": sum(t.numel() for t in tensors.values()),
    }
    if report is not None:
        manifest["injection"] = {
            "qkv": report.qkv,
            "out_proj": report.out_proj,
            "ffn": report.ffn,
            "adapters": report.adapters,
        }
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

    logger.info(
        "Saved %d adapter tensors (%s parameters) to %s",
        manifest["num_tensors"], f"{manifest['num_parameters']:,}", directory,
    )
    return directory / WEIGHTS_NAME


def load_lora(model: nn.Module, directory: str | Path, strict: bool = True) -> int:
    """Load adapters into a model that has already had ``apply_lora`` run on it.

    Returns the number of tensors loaded. With ``strict`` the key sets must
    match exactly in both directions, which is what catches a checkpoint
    trained with a different ``targets`` list or rank.
    """
    directory = Path(directory)
    weights_path = directory / WEIGHTS_NAME
    if not weights_path.exists():
        raise FileNotFoundError(f"no adapter weights at {weights_path}")

    loaded = load_file(str(weights_path))
    current = trainable_parameters(model)

    missing = sorted(set(current) - set(loaded))
    unexpected = sorted(set(loaded) - set(current))
    if strict and (missing or unexpected):
        raise RuntimeError(
            "adapter checkpoint does not match the model:\n"
            f"  missing from checkpoint: {missing[:6]}{' ...' if len(missing) > 6 else ''}\n"
            f"  not present in model:    {unexpected[:6]}{' ...' if len(unexpected) > 6 else ''}"
        )

    by_name = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in loaded.items():
            if name not in by_name:
                continue
            target = by_name[name]
            if target.shape != tensor.shape:
                raise RuntimeError(
                    f"shape mismatch for {name}: checkpoint {tuple(tensor.shape)} "
                    f"vs model {tuple(target.shape)}"
                )
            target.copy_(tensor.to(target.dtype).to(target.device))

    logger.info("Loaded %d adapter tensors from %s", len(loaded), directory)
    return len(loaded)


def assert_adapters_match(model: nn.Module, directory: str | Path) -> int:
    """Verify every adapter on disk equals the one in memory, exactly.

    This is the check that would have caught the original bug. It is called
    right after saving, so a run cannot finish believing it saved a checkpoint
    that is in fact incomplete. Returns the number of tensors compared.
    """
    directory = Path(directory)
    on_disk = load_file(str(directory / WEIGHTS_NAME))
    in_memory = trainable_parameters(model)

    if set(on_disk) != set(in_memory):
        only_memory = sorted(set(in_memory) - set(on_disk))
        only_disk = sorted(set(on_disk) - set(in_memory))
        raise RuntimeError(
            f"saved checkpoint is incomplete: {len(in_memory)} trainable "
            f"tensors in memory, {len(on_disk)} on disk.\n"
            f"  missing from disk: {only_memory[:8]}\n"
            f"  extra on disk:     {only_disk[:8]}"
        )

    for name, expected in in_memory.items():
        actual = on_disk[name].to(expected.device)
        if not torch.equal(actual, expected.cpu().to(actual.device)):
            max_diff = (actual.float() - expected.float().cpu()).abs().max().item()
            raise RuntimeError(
                f"adapter {name} differs between memory and disk: "
                f"max |diff| = {max_diff:.3e}"
            )

    logger.info("Verified %d adapter tensors round-trip exactly", len(in_memory))
    return len(in_memory)
