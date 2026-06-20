"""Train/val/test and active-learning splits."""

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


def make_al_split(
    records: list[Record], init_labeled: int, seed: int = 42
) -> tuple[list[Record], list[Record]]:
    """Split into an initial labeled set and an unlabeled pool."""
    if init_labeled > len(records):
        raise ValueError("init_labeled exceeds dataset size")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:init_labeled], shuffled[init_labeled:]
