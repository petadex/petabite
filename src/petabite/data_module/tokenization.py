"""ESM-C tokenizer wrapper (HuggingFace)."""

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
        # TODO: load HF tokenizer:
        #   from transformers import AutoTokenizer
        #   self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        raise NotImplementedError(
            "ESM-C tokenizer loading not implemented; wire transformers.AutoTokenizer"
        )

    def encode(self, sequences: list[str]) -> dict[str, object]:
        """Tokenize a list of sequences into model input tensors."""
        if self._tokenizer is None:
            self._load()
        # TODO: return self._tokenizer(sequences, padding=True, truncation=True,
        #   max_length=self.max_length, return_tensors="pt")
        raise NotImplementedError("ESM-C tokenization not implemented")
