"""Train entrypoint: LoRA fine-tune ESM-C on Logan derived PETases and plastic degrading enzyme sequence data.

Run:
    python -m petabite.train
    python -m petabite.train model.lora_r=16 trainer.num_train_epochs=10
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from petabite.data_module import PETaseDataset, make_splits
from petabite.data_module.tokenization import ESMCTokenizerWrapper
from petabite.data_module.utils import read_activity_csv
from petabite.model_module import ModelFactory
from petabite.trainer_module import PetAITrainer
from petabite.utils import setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="train")
def main(cfg: DictConfig) -> None:
    """Build dataset, model, and trainer from config, then train."""
    setup_logging()
    records = read_activity_csv(Path(cfg.data.csv_path))
    train_recs, val_recs, _ = make_splits(
        records, cfg.data.val_frac, cfg.data.test_frac, cfg.seed
    )
    train_ds = PETaseDataset(records=train_recs)
    val_ds = PETaseDataset(records=val_recs)

    model_cls = ModelFactory(cfg.model.name)
    model = model_cls(
        backbone_name=cfg.model.backbone_name,
        task=cfg.model.task,
        model_id=cfg.model.model_id,
        output_dim=cfg.model.output_dim,
        lora_r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        target_modules=list(cfg.model.target_modules),
        lora_fused=cfg.model.lora_fused,
        trust_remote_code=cfg.model.trust_remote_code,
    )

    tokenizer = ESMCTokenizerWrapper(
        model_name=cfg.model.model_id,
        max_length=cfg.data.max_length,
    )

    PetAITrainer(model=model, cfg=cfg).train(train_ds, val_ds, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
