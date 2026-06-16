from .acquisition import (
    ACQUISITION_REGISTRY,
    AcquisitionFunction,
    RandomAcquisition,
    UncertaintyAcquisition,
)
from .loop import ActiveLearningLoop
from .selection import export_query, select_top_k


def AcquisitionFactory(name: str):
    """Return the acquisition class registered under ``name``."""
    return ACQUISITION_REGISTRY.get(name)


__all__ = [
    "AcquisitionFactory",
    "ACQUISITION_REGISTRY",
    "AcquisitionFunction",
    "RandomAcquisition",
    "UncertaintyAcquisition",
    "ActiveLearningLoop",
    "select_top_k",
    "export_query",
]
