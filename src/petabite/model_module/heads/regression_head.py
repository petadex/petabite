"""Regression head for continuous activity prediction."""

from __future__ import annotations

import torch
from torch import nn


class RegressionHead(nn.Module):
    """MLP head mapping pooled embeddings to a continuous value.

    Args:
        input_dim: embedding dimension from the backbone.
        output_dim: number of regression targets (default 1).
        dropout: dropout probability.
    """

    def __init__(self, input_dim: int, output_dim: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return predicted values of shape (batch, output_dim)."""
        return self.net(embeddings)
