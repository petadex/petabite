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
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.hidden_size: int = 0  # set after loading
        self._model = None  # lazy-loaded
        # TODO: load and wrap:
        #   from transformers import AutoModel
        #   from peft import LoraConfig, get_peft_model
        #   base = AutoModel.from_pretrained(model_name)
        #   for p in base.parameters(): p.requires_grad = False
        #   cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
        #                    lora_dropout=lora_dropout, target_modules=[...])
        #   self._model = get_peft_model(base, cfg)
        #   self.hidden_size = base.config.hidden_size

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return pooled (mean over residues) embeddings: (batch, hidden_size)."""
        # TODO: out = self._model(input_ids=input_ids, attention_mask=attention_mask)
        #   masked mean-pool out.last_hidden_state with attention_mask
        raise NotImplementedError("ESM-C backbone forward not implemented")
