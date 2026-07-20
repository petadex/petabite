"""Custom LoRA adapters for ESM-C's fused LayerNorm+Linear modules.

biohub/ESMC fuses QKV projection + LayerNorm into _PyTorchLayerNormLinear and
FFN + LayerNorm into _PyTorchLayerNormMLP. Neither is an nn.Linear subclass so
PEFT cannot wrap them. This module replaces each with a thin LoRA wrapper that
re-runs the same forward pass and adds a low-rank delta at the correct point in
the computation graph — covering QKV, fc1, and fc2 in addition to the out_proj
that PEFT handles, giving full-block LoRA coverage across the PlasticESM backbone.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Filename used for the fused-LoRA weights alongside PEFT's adapter_model.safetensors.
FUSED_LORA_FILENAME = "fused_lora.safetensors"


class LoRALayerNormLinear(nn.Module):
    """LoRA wrapper for _PyTorchLayerNormLinear (the fused QKV projection).

    The base forward is: LayerNorm(x) → F.linear(weight).
    We add: lora_B(lora_A(LayerNorm(x))) * scale alongside it.
    """

    def __init__(self, base: nn.Module, r: int, alpha: float, dropout: float) -> None:
        super().__init__()
        for p in base.parameters():
            p.requires_grad = False
        self.base = base
        d_in: int = base.d_in
        d_out: int = base.weight.shape[0]
        self.lora_A = nn.Linear(d_in, r, bias=False)
        self.lora_B = nn.Linear(r, d_out, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.drop = nn.Dropout(dropout)
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.base
        x_n = F.layer_norm(x, (b.d_in,), b.layer_norm_weight, b.layer_norm_bias, b.eps)
        return F.linear(x_n, b.weight) + self.lora_B(self.lora_A(self.drop(x_n))) * self.scaling


class LoRALayerNormMLP(nn.Module):
    """LoRA wrapper for _PyTorchLayerNormMLP (the fused SwiGLU FFN).

    The base forward is: LayerNorm(x) → fc1 → SwiGLU → fc2.
    We add a LoRA delta at fc1 (d_model → fc1_dim) and at fc2
    (intermediate → d_model) independently.
    """

    def __init__(self, base: nn.Module, r: int, alpha: float, dropout: float) -> None:
        super().__init__()
        for p in base.parameters():
            p.requires_grad = False
        self.base = base
        d_in: int = base.hidden_size
        fc1_out: int = base.fc1_weight.shape[0]   # 5120 for 300M
        fc2_in: int = base.fc2_weight.shape[1]    # 2560 for 300M (post-SwiGLU dim)

        self.lora_A1 = nn.Linear(d_in, r, bias=False)
        self.lora_B1 = nn.Linear(r, fc1_out, bias=False)
        self.lora_A2 = nn.Linear(fc2_in, r, bias=False)
        self.lora_B2 = nn.Linear(r, d_in, bias=False)

        for lin in (self.lora_A1, self.lora_A2):
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
        for lin in (self.lora_B1, self.lora_B2):
            nn.init.zeros_(lin.weight)

        self.drop = nn.Dropout(dropout)
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.base
        x_n = F.layer_norm(x, (b.hidden_size,), b.layer_norm_weight, b.layer_norm_bias, b.eps)

        fc1_delta = self.lora_B1(self.lora_A1(self.drop(x_n))) * self.scaling
        fc1 = F.linear(x_n, b.fc1_weight) + fc1_delta
        x1, x2 = fc1.chunk(2, dim=-1)
        mid = F.silu(x1) * x2

        fc2_delta = self.lora_B2(self.lora_A2(self.drop(mid))) * self.scaling
        return F.linear(mid, b.fc2_weight) + fc2_delta


def inject_fused_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.01,
) -> dict[str, int]:
    """Replace fused QKV and FFN modules with LoRA-wrapped versions in-place.

    Walks the model tree and swaps every _PyTorchLayerNormLinear and
    _PyTorchLayerNormMLP with LoRALayerNormLinear / LoRALayerNormMLP.
    The base weights are frozen; only the LoRA A/B matrices are trainable.

    Returns a dict with the count of injected adapters per type.

    Raises:
        RuntimeError: if the model is already injected, or if no fused module was
            found. The fused class names only exist on ESM-C's pure-PyTorch fallback
            path; with transformer-engine installed the blocks are TE modules instead
            and nothing matches. Without that check the run trains out_proj only,
            silently.
    """
    if any(
        isinstance(m, (LoRALayerNormLinear, LoRALayerNormMLP)) for m in model.modules()
    ):
        raise RuntimeError(
            "inject_fused_lora was already applied to this model. Injecting twice "
            "wraps each wrapper's own .base, producing a nested adapter that fails "
            "at forward."
        )

    counts: dict[str, int] = {"qkv": 0, "ffn": 0}
    _replace(model, r, alpha, dropout, counts)
    if counts["qkv"] == 0 or counts["ffn"] == 0:
        raise RuntimeError(
            "inject_fused_lora found no fused modules to wrap "
            f"(qkv={counts['qkv']}, ffn={counts['ffn']}). Expected "
            "_PyTorchLayerNormLinear / _PyTorchLayerNormMLP in every block. This "
            "usually means transformer-engine is installed, so ESM-C built TE fused "
            "modules under different class names — attention QKV and the FFN would "
            "not be trained. Uninstall transformer-engine, or set lora_fused=false "
            "to knowingly train out_proj only."
        )
    logger.info(
        "Injected fused LoRA: %d QKV adapters, %d FFN adapters (r=%d, alpha=%s)",
        counts["qkv"], counts["ffn"], r, alpha,
    )
    return counts


def fused_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect the trainable LoRA tensors from every injected fused module.

    PEFT's ``get_peft_model_state_dict`` cannot see these: it keeps only keys matching
    ``"lora_" in k and adapter_name in k``, and these parameters carry no ``.default.``
    adapter segment. Saving a PeftModel therefore drops every fused QKV/FFN weight.

    Args:
        model: any module containing injected fused-LoRA wrappers.

    Returns:
        Mapping of anchor-relative parameter name to detached CPU tensor.
    """
    return {key: param.detach().cpu() for key, param in fused_lora_parameters(model).items()}


def fused_lora_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    """Map anchor-relative key to live parameter for every fused-LoRA module.

    Keys are relative to the longest common ancestor of the fused modules rather than
    to the model root, so the same checkpoint loads into any model that wraps the same
    block stack at a different depth. That covers both wrappers we care about:
    ``get_peft_model`` prepending ``base_model.model.``, and Stage 1 training
    ``ESMCForMaskedLM`` (blocks under ``esmc.transformer``) while Stage 2 loads
    ``ESMCModel`` (blocks under ``transformer``). Both reduce to ``0.attn.…``.

    Modules are located by isinstance rather than by name substring, so renaming a
    parameter cannot silently drop it from the checkpoint.
    """
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (LoRALayerNormLinear, LoRALayerNormMLP))
    ]
    anchor = _common_prefix([name for name, _ in modules])
    offset = len(anchor) + 1 if anchor else 0

    params: dict[str, nn.Parameter] = {}
    for name, module in modules:
        relative = name[offset:]
        for param_name, param in module.named_parameters():
            if param_name.startswith("base."):  # frozen ESM-C weights
                continue
            params[f"{relative}.{param_name}" if relative else param_name] = param
    return params


def save_fused_lora(model: nn.Module, directory: str | Path) -> Path:
    """Write the fused-LoRA weights next to PEFT's adapter.

    Args:
        model: the (possibly PEFT-wrapped) model being trained.
        directory: destination directory; created if absent.

    Returns:
        Path to the written file.

    Raises:
        RuntimeError: if the model contains no fused-LoRA modules, which would
            otherwise write an empty checkpoint that looks successful.
    """
    from safetensors.torch import save_file

    state = fused_lora_state_dict(model)
    if not state:
        raise RuntimeError(
            "save_fused_lora found no fused-LoRA modules; nothing to save. Was "
            "inject_fused_lora called on this model?"
        )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FUSED_LORA_FILENAME
    save_file(state, str(path))
    n_params = sum(t.numel() for t in state.values())
    logger.info(
        "Saved fused LoRA: %d tensors, %d params -> %s", len(state), n_params, path
    )
    return path


def load_fused_lora(model: nn.Module, path: str | Path, strict: bool = True) -> int:
    """Load fused-LoRA weights produced by :func:`save_fused_lora`.

    Call this *after* ``inject_fused_lora``: the wrapper modules must already exist.

    Args:
        model: model with fused-LoRA wrappers already injected.
        path: the ``fused_lora.safetensors`` file, or a directory containing it.
        strict: if True, raise when the checkpoint and model keys disagree.

    Returns:
        Number of tensors loaded.

    Raises:
        FileNotFoundError: if the file does not exist.
        RuntimeError: if strict and keys or shapes do not match.
    """
    from safetensors.torch import load_file

    path = Path(path)
    if path.is_dir():
        path = path / FUSED_LORA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"No fused-LoRA checkpoint at {path}")

    loaded = load_file(str(path))
    params = fused_lora_parameters(model)
    missing = sorted(set(params) - set(loaded))
    unexpected = sorted(set(loaded) - set(params))
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Fused-LoRA checkpoint does not match the model: {len(missing)} missing "
            f"(e.g. {missing[:3]}), {len(unexpected)} unexpected "
            f"(e.g. {unexpected[:3]}). Check that lora_r matches the trained run."
        )

    n_loaded = 0
    for key, tensor in loaded.items():
        param = params.get(key)
        if param is None:
            continue  # only reachable when strict=False
        if param.shape != tensor.shape:
            raise RuntimeError(
                f"Shape mismatch for {key}: checkpoint {tuple(tensor.shape)} vs "
                f"model {tuple(param.shape)}. Check lora_r."
            )
        with torch.no_grad():
            param.copy_(tensor.to(param.device, param.dtype))
        n_loaded += 1
    logger.info("Loaded fused LoRA: %d tensors from %s", n_loaded, path)
    return n_loaded


def _common_prefix(paths: list[str]) -> str:
    """Longest shared dot-separated prefix across module paths ("" if none)."""
    if not paths:
        return ""
    common: list[str] = []
    for segments in zip(*(p.split(".") for p in paths)):
        if len(set(segments)) != 1:
            break
        common.append(segments[0])
    return ".".join(common)


def _replace(module: nn.Module, r: int, alpha: float, dropout: float, counts: dict) -> None:
    for name, child in list(module.named_children()):
        cls_name = type(child).__name__
        if cls_name == "_PyTorchLayerNormLinear":
            setattr(module, name, LoRALayerNormLinear(child, r, alpha, dropout))
            counts["qkv"] += 1
        elif cls_name == "_PyTorchLayerNormMLP":
            setattr(module, name, LoRALayerNormMLP(child, r, alpha, dropout))
            counts["ffn"] += 1
        else:
            _replace(child, r, alpha, dropout, counts)
