"""ESM-C backbone with LoRA adapters (HuggingFace + peft)."""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class ESMCBackbone(nn.Module):
    """Loads a HuggingFace ESM-C model and wraps it with LoRA adapters.

    The base weights are frozen; only LoRA adapters train.

    Args:
        model_name: HF checkpoint id (e.g. an EvolutionaryScale ESM-C model).
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        lora_dropout: LoRA dropout.
    """

    def __init__(
        self,
        model_name: str,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel

        base = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        for p in base.parameters():
            p.requires_grad = False
        # biohub/ESMC uses d_model (not hidden_size); out_proj is the only
        # standard nn.Linear target in each transformer block.
        self.hidden_size: int = base.config.d_model
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["out_proj"],
            bias="none",
        )
        self._model = get_peft_model(base, lora_cfg)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return masked mean-pooled embeddings: (batch, hidden_size)."""
        out = self._model(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state: (batch, seq_len, hidden_size)
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
