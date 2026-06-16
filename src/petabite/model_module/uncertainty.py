"""MC-dropout uncertainty estimation."""

from __future__ import annotations

import logging
from typing import Dict

import torch
from torch import nn

logger = logging.getLogger(__name__)


def enable_dropout(model: nn.Module) -> None:
    """Set all dropout layers to train mode while keeping the rest in eval."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    n_samples: int = 20,
) -> Dict[str, torch.Tensor]:
    """Run ``n_samples`` stochastic forward passes; return mean and variance.

    Returns:
        dict with 'mean' and 'variance' over the sampled logits.
    """
    # TODO: implement once backbone.forward works. Sketch:
    #   model.eval(); enable_dropout(model)
    #   preds = torch.stack([
    #       model(input_ids, attention_mask)["logits"] for _ in range(n_samples)
    #   ])
    #   return {"mean": preds.mean(0), "variance": preds.var(0)}
    raise NotImplementedError("MC-dropout prediction not implemented")
