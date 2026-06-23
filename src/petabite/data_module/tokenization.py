"""ESM-C tokenizer wrapper for PETase sequence encoding.

Provides a thin convenience layer around the HuggingFace ESM-C tokenizer so that
both the Stage 1 MLM pipeline (mlm_finetune.py) and the Stage 2 activity pipeline
(train.py) can encode Logan-derived PETadex sequences with consistent padding,
truncation, and tensor output.
"""

from __future__ import annotations

import logging

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
        self._tokenizer = None  # lazy-loaded

    def _load(self) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

    def encode(self, sequences: list[str]) -> dict[str, object]:
        """Tokenize a list of sequences into model input tensors."""
        if self._tokenizer is None:
            self._load()
        return self._tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
