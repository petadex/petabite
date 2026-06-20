"""Acquisition function interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AcquisitionFunction(ABC):
    """Scores pool items; higher score == higher priority to label."""

    @abstractmethod
    def score(self, pool_size: int, **kwargs: object) -> np.ndarray:
        """Return a 1-D array of length ``pool_size`` of acquisition scores."""
        raise NotImplementedError
