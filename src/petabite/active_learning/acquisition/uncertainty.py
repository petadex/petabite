"""Uncertainty (MC-dropout variance) acquisition."""

from __future__ import annotations

import logging

import numpy as np

from .base import AcquisitionFunction

logger = logging.getLogger(__name__)


class UncertaintyAcquisition(AcquisitionFunction):
    """Scores pool items by predictive variance (higher == more uncertain)."""

    def score_from_variance(self, variances: np.ndarray) -> np.ndarray:
        """Return scores directly proportional to per-item variance."""
        return np.asarray(variances, dtype=float)

    def score(self, pool_size: int, **kwargs: object) -> np.ndarray:
        """Score the pool using precomputed ``variances`` (kwarg).

        Raises:
            ValueError: if ``variances`` is missing or wrong length.
        """
        variances = kwargs.get("variances")
        if variances is None:
            raise ValueError(
                "UncertaintyAcquisition.score requires a 'variances' kwarg "
                "(from model_module.mc_dropout_predict)"
            )
        arr = np.asarray(variances, dtype=float)
        if arr.shape[0] != pool_size:
            raise ValueError("variances length must equal pool_size")
        return arr
