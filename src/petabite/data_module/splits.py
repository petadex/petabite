"""Train/val/test splits for wet-lab activity data.

The PETadex sequences used in Stage 2 are already 90%-identity dereplicated
(Logan 90pid centroids), so a plain random split is safe — no cluster leakage.
The active-learning split helpers will live here as wet-lab rounds accumulate.
"""

from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)

Record = dict[str, object]


def make_splits(
    records: list[Record],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[Record], list[Record], list[Record]]:
    """Shuffle and partition records into train/val/test."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return train, val, test
