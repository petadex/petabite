"""Stage 2 activity model: PlasticESM backbone + regression or classification head.

ESMCActivityModel composes an ESM-C backbone (optionally initialized from a
PlasticESM checkpoint from Stage 1 Logan MLM pre-training) with a lightweight MLP
head that maps pooled sequence embeddings to wet-lab activity measurements.
Supports both continuous activity regression and binary/multi-class classification.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from .backbone import BACKBONE_REGISTRY
from .heads import HEAD_REGISTRY

logger = logging.getLogger(__name__)

_LOSSES = {
    "regression": nn.MSELoss,
    "classification": nn.CrossEntropyLoss,
}


class ESMCActivityModel(nn.Module):
    """ESM-C backbone + regression/classification head.

    Args:
        backbone_name: registry key for the backbone (e.g. "esmc").
        task: "regression" or "classification".
        model_id: HF checkpoint id passed to the backbone.
        output_dim: head output dim (1 for regression, n_classes for clf).
        lora_r / lora_alpha / lora_dropout: LoRA hyperparameters.
        target_modules: PEFT LoRA target module name suffixes.
        lora_fused: inject custom LoRA into fused QKV/FFN modules.
        trust_remote_code: passed to the backbone's from_pretrained.
    """

    def __init__(
        self,
        backbone_name: str,
        task: str,
        model_id: str,
        output_dim: int = 1,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: list[str] | None = None,
        lora_fused: bool = False,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        if task not in _LOSSES:
            raise ValueError(f"Unknown task '{task}'. Valid: {sorted(_LOSSES)}")
        self.task = task
        backbone_cls = BACKBONE_REGISTRY.get(backbone_name)
        self.backbone = backbone_cls(
            model_id=model_id,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            lora_fused=lora_fused,
            trust_remote_code=trust_remote_code,
        )
        head_cls = HEAD_REGISTRY.get(task)
        self._head_cls = head_cls
        self._output_dim = output_dim
        self.head: nn.Module | None = None
        self.loss_fn = _LOSSES[task]()

    def _ensure_head(self) -> None:
        if self.head is None:
            self.head = self._head_cls(
                input_dim=self.backbone.hidden_size, output_dim=self._output_dim
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return dict with 'logits' and (if labels given) 'loss'."""
        embeddings = self.backbone(input_ids, attention_mask)
        self._ensure_head()
        assert self.head is not None
        logits = self.head(embeddings)
        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            if self.task == "regression":
                out["loss"] = self.loss_fn(logits.squeeze(-1), labels.float())
            else:
                out["loss"] = self.loss_fn(logits, labels.long())
        return out
