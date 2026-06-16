"""Random acquisition baseline."""

from __future__ import annotations

import numpy as np

from .base import AcquisitionFunction


class RandomAcquisition(AcquisitionFunction):
    """Assigns uniform random scores (reproducible under ``seed``)."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def score(self, pool_size: int, **kwargs: object) -> np.ndarray:
        """Return random scores of length ``pool_size``."""
        return self._rng.random(pool_size)
