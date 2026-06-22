"""Custom LoRA adapters for ESM-C's fused LayerNorm+Linear modules.

biohub/ESMC uses _PyTorchLayerNormLinear (QKV) and _PyTorchLayerNormMLP (FFN)
which are not nn.Linear subclasses, so PEFT cannot wrap them. This module
replaces each with a thin LoRA wrapper that re-runs the base forward and adds
a low-rank delta at the same point in the computation graph.
"""

from __future__ import annotations

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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

        fc1 = F.linear(x_n, b.fc1_weight) + self.lora_B1(self.lora_A1(self.drop(x_n))) * self.scaling
        x1, x2 = fc1.chunk(2, dim=-1)
        mid = F.silu(x1) * x2

        return F.linear(mid, b.fc2_weight) + self.lora_B2(self.lora_A2(self.drop(mid))) * self.scaling


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
    """
    counts: dict[str, int] = {"qkv": 0, "ffn": 0}
    _replace(model, r, alpha, dropout, counts)
    logger.info(
        "Injected fused LoRA: %d QKV adapters, %d FFN adapters (r=%d, alpha=%s)",
        counts["qkv"], counts["ffn"], r, alpha,
    )
    return counts


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
