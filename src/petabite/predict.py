"""Predict entrypoint: run inference on a CSV of sequences."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from petabite.data_module import PETaseDataset
from petabite.data_module.utils import read_activity_csv
from petabite.model_module import ModelFactory
from petabite.utils import setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="train")
def main(cfg: DictConfig) -> None:
    """Load model and predict activity for each sequence in the input CSV."""
    setup_logging()
    records = read_activity_csv(Path(cfg.data.csv_path))
    dataset = PETaseDataset(records=records)

    model_cls = ModelFactory(cfg.model.name)
    _ = model_cls(
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
    # TODO: load trained checkpoint, tokenize dataset, run forward, write
    #   predictions CSV to cfg.output_dir.
    raise NotImplementedError(
        f"Prediction loop not implemented (dataset has {len(dataset)} records)"
    )


if __name__ == "__main__":
    main()
