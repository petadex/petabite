"""Stage 2 — PetAI activity trainer: supervised fine-tuning on wet-lab assay data.

Wraps HuggingFace Trainer to map PETase sequences → activity scores using the
PlasticESM backbone (Stage 1) plus a task head. Wet-lab assay measurements from
the iGEM team feed into this trainer in an active-learning loop: model predicts,
wet lab validates, data returns, model improves.
"""

from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from petabite.data_module.tokenization import ESMCTokenizerWrapper
from petabite.trainer_module.metrics import classification_metrics, regression_metrics
from petabite.utils import dump_pip_freeze, log_environment, set_seed

logger = logging.getLogger(__name__)


class _ActivityCollator:
    """Tokenizes raw sequence records into model-ready batches."""

    def __init__(self, tokenizer: ESMCTokenizerWrapper) -> None:
        self.tokenizer = tokenizer

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        sequences = [r["sequence"] for r in records]
        labels = torch.tensor([r["label"] for r in records], dtype=torch.float32)
        encoded = self.tokenizer.encode(sequences)
        encoded["labels"] = labels
        return encoded


def _configure_wandb(cfg_wandb: DictConfig) -> str:
    """Set W&B env vars; return the Trainer report_to value."""
    if not cfg_wandb.enabled:
        os.environ["WANDB_DISABLED"] = "true"
        return "none"
    os.environ["WANDB_PROJECT"] = cfg_wandb.project
    if cfg_wandb.entity:
        os.environ["WANDB_ENTITY"] = cfg_wandb.entity
    return "wandb"


def _build_training_args(cfg: DictConfig, report_to: str) -> TrainingArguments:
    """Build TrainingArguments from the resolved Hydra config."""
    kwargs: dict[str, Any] = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.trainer.num_train_epochs,
        per_device_train_batch_size=cfg.trainer.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.trainer.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        learning_rate=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        warmup_ratio=cfg.trainer.warmup_ratio,
        lr_scheduler_type=cfg.trainer.lr_scheduler_type,
        logging_steps=cfg.trainer.logging_steps,
        save_steps=cfg.trainer.save_steps,
        save_total_limit=cfg.trainer.save_total_limit,
        bf16=cfg.trainer.bf16,
        fp16=cfg.trainer.fp16,
        gradient_checkpointing=cfg.trainer.gradient_checkpointing,
        dataloader_num_workers=cfg.trainer.dataloader_num_workers,
        report_to=report_to,
        run_name=cfg.wandb.run_name,
        save_strategy="steps",
    )
    # `evaluation_strategy` was renamed to `eval_strategy` in transformers 4.46.
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"
    kwargs[eval_key] = "steps"
    kwargs["eval_steps"] = cfg.trainer.eval_steps
    return TrainingArguments(**kwargs)


class PetAITrainer:
    """Configures and runs LoRA fine-tuning for ESM-C activity models.

    Wraps transformers.Trainer; saves reproducibility artifacts (frozen config
    + pip freeze) to the output directory.

    Args:
        model: an ESMCActivityModel instance.
        cfg: the full resolved Hydra config (train.yaml).
    """

    def __init__(self, model: object, cfg: DictConfig) -> None:
        set_seed(cfg.seed)
        self.model = model
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Environment: %s", log_environment())

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        tokenizer: ESMCTokenizerWrapper | None = None,
    ) -> None:
        """Run training. Saves reproducibility artifacts, then trains.

        Args:
            train_dataset: PETaseDataset for training.
            eval_dataset: optional PETaseDataset for evaluation.
            tokenizer: ESMCTokenizerWrapper; inferred from cfg.model.model_id
                if not provided.
        """
        dump_pip_freeze(self.output_dir / "requirements.txt")
        OmegaConf.save(self.cfg, self.output_dir / "config.yaml")

        if tokenizer is None:
            tokenizer = ESMCTokenizerWrapper(
                model_name=self.cfg.model.model_id,
                max_length=self.cfg.data.max_length,
            )

        report_to = _configure_wandb(self.cfg.wandb)
        args = _build_training_args(self.cfg, report_to)

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=_ActivityCollator(tokenizer),
            compute_metrics=self._compute_metrics,
        )
        trainer.train()
        trainer.save_model(str(self.output_dir))

        if self.cfg.push_to_hub.enabled:
            logger.info("Pushing to %s", self.cfg.push_to_hub.repo_id)
            self.model.push_to_hub(
                self.cfg.push_to_hub.repo_id,
                private=self.cfg.push_to_hub.private,
            )

        logger.info("Training complete. Checkpoint saved to %s", self.output_dir)

    def _compute_metrics(self, eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        if self.cfg.model.task == "regression":
            return regression_metrics(labels, logits.squeeze(-1))
        preds = np.argmax(logits, axis=-1)
        return classification_metrics(labels.astype(int), preds)
