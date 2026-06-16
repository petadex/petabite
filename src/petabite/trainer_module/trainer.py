"""Training wrapper around HuggingFace Trainer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset

from petabite.utils import dump_pip_freeze, log_environment, set_seed

logger = logging.getLogger(__name__)


class ActivityTrainer:
    """Configures and runs training for ESM-C activity models.

    Wraps transformers.Trainer; saves reproducibility artifacts (resolved
    config + pip freeze) to the output directory.

    Args:
        model: an ESMCActivityModel instance.
        output_dir: directory for checkpoints and run artifacts.
        task: "regression" or "classification" (selects metrics).
        seed: random seed.
    """

    def __init__(
        self,
        model: object,
        output_dir: Path,
        task: str,
        seed: int = 42,
    ) -> None:
        set_seed(seed)
        self.model = model
        self.output_dir = Path(output_dir)
        self.task = task
        self.seed = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Environment: %s", log_environment())

    def train(
        self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None
    ) -> None:
        """Run training. Saves reproducibility artifacts then trains."""
        dump_pip_freeze(self.output_dir / "requirements.txt")
        # TODO: build transformers.TrainingArguments + Trainer with a
        #   data collator that calls ESMCTokenizerWrapper.encode, and
        #   compute_metrics dispatching on self.task. Then trainer.train().
        raise NotImplementedError("HF Trainer wiring not implemented")
