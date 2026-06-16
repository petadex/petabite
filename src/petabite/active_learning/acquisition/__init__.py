from petabite.utils import Registry

from .base import AcquisitionFunction
from .random_baseline import RandomAcquisition
from .uncertainty import UncertaintyAcquisition

ACQUISITION_REGISTRY = Registry("acquisition")
ACQUISITION_REGISTRY.register("random")(RandomAcquisition)
ACQUISITION_REGISTRY.register("uncertainty")(UncertaintyAcquisition)

__all__ = [
    "ACQUISITION_REGISTRY",
    "AcquisitionFunction",
    "RandomAcquisition",
    "UncertaintyAcquisition",
]
