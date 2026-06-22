"""Training wrapper around HuggingFace Trainer."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from petabite.data_module.tokenization import ESMCTokenizerWrapper
from petabite.trainer_module.metrics import classification_metrics, regression_metrics
from petabite.utils import dump_pip_freeze, log_environment, set_seed

logger = logging.getLogger(__name__)


@dataclass
class _Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class _ActivityCollator:
    """Tokenizes raw sequence records and stacks them into a batch."""

    def __init__(self, tokenizer: ESMCTokenizerWrapper) -> None:
        self.tokenizer = tokenizer

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        sequences = [r["sequence"] for r in records]
        labels = torch.tensor([r["label"] for r in records], dtype=torch.float32)
        encoded = self.tokenizer.encode(sequences)
        encoded["labels"] = labels
        return encoded


class PetAITrainer:
    """Configures and runs training for ESM-C activity models.

    Wraps transformers.Trainer; saves reproducibility artifacts (resolved
    config + pip freeze) to the output directory.

    Args:
        model: an ESMCActivityModel instance.
        output_dir: directory for checkpoints and run artifacts.
        task: "regression" or "classification" (selects metrics).
        seed: random seed.
        num_train_epochs: training epochs.
        per_device_train_batch_size: batch size per device during training.
        per_device_eval_batch_size: batch size per device during evaluation.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        logging_steps: steps between logging entries.
        bf16: use bfloat16 mixed precision (requires Ampere+ GPU).
    """

    def __init__(
        self,
        model: object,
        output_dir: Path,
        task: str,
        seed: int = 42,
        num_train_epochs: int = 5,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        logging_steps: int = 10,
        bf16: bool = False,
    ) -> None:
        set_seed(seed)
        self.model = model
        self.output_dir = Path(output_dir)
        self.task = task
        self.seed = seed
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.logging_steps = logging_steps
        self.bf16 = bf16
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Environment: %s", log_environment())

    def _build_training_args(self) -> TrainingArguments:
        kwargs: dict[str, Any] = dict(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            logging_steps=self.logging_steps,
            save_strategy="epoch",
            bf16=self.bf16,
            report_to="none",
        )
        # `evaluation_strategy` was renamed to `eval_strategy` in transformers 4.46.
        ta_params = inspect.signature(TrainingArguments.__init__).parameters
        kwargs["eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"] = "epoch"
        return TrainingArguments(**kwargs)

    def _compute_metrics(self, eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        if self.task == "regression":
            preds = logits.squeeze(-1)
            return regression_metrics(labels, preds)
        else:
            preds = np.argmax(logits, axis=-1)
            return classification_metrics(labels.astype(int), preds)

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        tokenizer: ESMCTokenizerWrapper | None = None,
    ) -> None:
        """Run training. Saves reproducibility artifacts then trains.

        Args:
            train_dataset: PETaseDataset for training.
            eval_dataset: optional PETaseDataset for evaluation.
            tokenizer: ESMCTokenizerWrapper; if None, one is built from the
                backbone's model_name.
        """
        dump_pip_freeze(self.output_dir / "requirements.txt")

        if tokenizer is None:
            backbone_name = getattr(self.model, "backbone", None)
            model_name = getattr(backbone_name, "model_name", None) if backbone_name else None
            if model_name is None:
                raise ValueError(
                    "Could not infer model_name from backbone; pass tokenizer explicitly."
                )
            tokenizer = ESMCTokenizerWrapper(model_name=model_name)

        collator = _ActivityCollator(tokenizer)
        args = self._build_training_args()

        # Wrap model so Trainer can call forward with keyword args from the collator.
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            compute_metrics=self._compute_metrics,
        )
        trainer.train()
        trainer.save_model(str(self.output_dir))
        logger.info("Training complete. Checkpoint saved to %s", self.output_dir)
