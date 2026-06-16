"""Classification head for binned/active-inactive activity prediction."""

from __future__ import annotations

import torch
from torch import nn


class ClassificationHead(nn.Module):
    """MLP head mapping pooled embeddings to class logits.

    Args:
        input_dim: embedding dimension from the backbone.
        output_dim: number of classes.
        dropout: dropout probability.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape (batch, output_dim)."""
        return self.net(embeddings)
