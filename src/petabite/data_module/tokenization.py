"""ESM-C tokenizer wrapper for PETase sequence encoding.

Provides a thin convenience layer around the HuggingFace ESM-C tokenizer so that
both the Stage 1 MLM pipeline (mlm_finetune.py) and the Stage 2 activity pipeline
(train.py) can encode Logan-derived PETadex sequences with consistent padding,
truncation, and tensor output.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # keep this module import-light; torch is only needed for the type
    import torch

logger = logging.getLogger(__name__)


class ESMCTokenizerWrapper:
    """Wraps the HuggingFace ESM-C tokenizer for batch encoding.

    Args:
        model_name: HF checkpoint id for the ESM-C tokenizer.
        max_length: max sequence length (truncation).
    """

    def __init__(self, model_name: str, max_length: int = 1024) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer: Any | None = None  # lazy-loaded on first encode()

    def _load(self) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    def encode(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize a list of sequences into model input tensors."""
        tokenizer = self._tokenizer
        if tokenizer is None:
            tokenizer = self._load()
            self._tokenizer = tokenizer
        return tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
