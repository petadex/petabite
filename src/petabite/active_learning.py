"""Active-learning entrypoint: train, score pool, export wet-lab queries."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from petabite.active_learning import ActiveLearningLoop
from petabite.data_module import make_al_split
from petabite.data_module.utils import read_activity_csv
from petabite.utils import setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run one active-learning round and export a query CSV.

    NOTE: scoring with real model uncertainty requires the backbone/trainer
    stubs to be implemented. Until then this wires the loop and selection.
    """
    setup_logging()
    records = read_activity_csv(Path(cfg.data.csv_path))
    labeled, pool = make_al_split(records, cfg.data.init_labeled, cfg.seed)

    _ = ActiveLearningLoop(
        acquisition_name=cfg.active_learning.acquisition_name,
        query_size=min(cfg.active_learning.query_size, len(pool)),
        output_dir=Path(cfg.output_dir),
    )
    # TODO: train model on `labeled`, run mc_dropout_predict over `pool`,
    #   pass variances into loop.run_round(..., variances=variances).
    raise NotImplementedError(
        f"AL scoring needs the model stubs implemented "
        f"({len(labeled)} labeled, {len(pool)} pool ready)"
    )


if __name__ == "__main__":
    main()
