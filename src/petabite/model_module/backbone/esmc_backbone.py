"""ESM-C backbone with LoRA adapters (HuggingFace + peft)."""

from __future__ import annotations

import logging

import torch
from torch import nn

from petabite.model_module.backbone.fused_lora import inject_fused_lora

logger = logging.getLogger(__name__)


class ESMCBackbone(nn.Module):
    """Loads a HuggingFace ESM-C model and wraps it with LoRA adapters.

    The base weights are frozen; only LoRA adapters train.

    Args:
        model_id: HF checkpoint id (e.g. "biohub/ESMC-300M").
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: dropout on LoRA paths.
        target_modules: list of nn.Linear module name suffixes for PEFT LoRA.
        lora_fused: if True, also inject custom LoRA into the fused QKV/FFN
            modules (_PyTorchLayerNormLinear / _PyTorchLayerNormMLP) that PEFT
            cannot target.
        trust_remote_code: passed to from_pretrained.
    """

    def __init__(
        self,
        model_id: str,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: list[str] | None = None,
        lora_fused: bool = False,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel

        if target_modules is None:
            target_modules = ["out_proj"]

        self.model_id = model_id
        base = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        for p in base.parameters():
            p.requires_grad = False
        self.hidden_size: int = base.config.d_model

        if lora_fused:
            inject_fused_lora(base, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self._model = get_peft_model(base, lora_cfg)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return masked mean-pooled embeddings: (batch, hidden_size)."""
        out = self._model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
