"""Train entrypoint: LoRA fine-tune ESM-C on PETase activity data."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from petabite.data_module import DatasetFactory, make_splits
from petabite.data_module.utils import read_activity_csv
from petabite.model_module import ModelFactory
from petabite.trainer_module import ActivityTrainer
from petabite.utils import setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Build dataset, model, and trainer from config, then train."""
    setup_logging()
    records = read_activity_csv(Path(cfg.data.csv_path))
    train_recs, val_recs, _ = make_splits(
        records, cfg.data.val_frac, cfg.data.test_frac, cfg.seed
    )
    dataset_cls = DatasetFactory(cfg.data.name)
    train_ds = dataset_cls(records=train_recs)
    val_ds = dataset_cls(records=val_recs)

    model_cls = ModelFactory(cfg.model.name)
    model = model_cls(
        backbone_name=cfg.model.backbone_name,
        task=cfg.model.task,
        model_name=cfg.model.model_name,
        output_dim=cfg.model.output_dim,
        lora_r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
    )
    trainer = ActivityTrainer(
        model=model,
        output_dir=Path(cfg.output_dir),
        task=cfg.model.task,
        seed=cfg.seed,
    )
    trainer.train(train_ds, val_ds)


if __name__ == "__main__":
    main()
