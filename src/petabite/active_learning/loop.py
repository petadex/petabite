"""Pool-based active learning round controller."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .acquisition import ACQUISITION_REGISTRY
from .selection import export_query, select_top_k

logger = logging.getLogger(__name__)

Record = dict[str, object]


class ActiveLearningLoop:
    """Runs one active-learning round: score pool, select, export query.

    Args:
        acquisition_name: registry key ("uncertainty" or "random").
        query_size: number of items to select per round.
        output_dir: directory for query CSVs.
        acquisition_kwargs: kwargs forwarded to the acquisition constructor.
    """

    def __init__(
        self,
        acquisition_name: str,
        query_size: int,
        output_dir: Path,
        **acquisition_kwargs: object,
    ) -> None:
        acq_cls = ACQUISITION_REGISTRY.get(acquisition_name)
        self.acquisition = acq_cls(**acquisition_kwargs)
        self.query_size = query_size
        self.output_dir = Path(output_dir)

    def run_round(
        self,
        labeled: list[Record],
        pool: list[Record],
        round_idx: int,
        scores: np.ndarray | None = None,
        **score_kwargs: object,
    ) -> tuple[list[int], Path]:
        """Score the pool, select top-K, and export a query CSV.

        ``scores`` may be passed precomputed; otherwise the acquisition
        function is asked to score a pool of ``len(pool)``.
        """
        if scores is None:
            scores = self.acquisition.score(pool_size=len(pool), **score_kwargs)
        selected = select_top_k(scores, self.query_size)
        query_path = export_query(
            pool, selected, self.output_dir / f"query_round_{round_idx}.csv", round_idx
        )
        logger.info(
            "AL round %d: %d labeled, %d pool, selected %d",
            round_idx,
            len(labeled),
            len(pool),
            len(selected),
        )
        return selected, query_path
