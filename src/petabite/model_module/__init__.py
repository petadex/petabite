from petabite.utils import Registry

from .backbone import BACKBONE_REGISTRY, ESMCBackbone
from .heads import HEAD_REGISTRY, ClassificationHead, RegressionHead
from .model import ESMCActivityModel
from .uncertainty import enable_dropout, mc_dropout_predict

MODEL_REGISTRY = Registry("model")
MODEL_REGISTRY.register("esmc_activity")(ESMCActivityModel)


def ModelFactory(name: str) -> type:  # noqa: N802
    """Return the model class registered under ``name``."""
    return MODEL_REGISTRY.get(name)


def BackboneFactory(name: str) -> type:  # noqa: N802
    """Return the backbone class registered under ``name``."""
    return BACKBONE_REGISTRY.get(name)


__all__ = [
    "ModelFactory",
    "BackboneFactory",
    "ESMCActivityModel",
    "ESMCBackbone",
    "RegressionHead",
    "ClassificationHead",
    "MODEL_REGISTRY",
    "HEAD_REGISTRY",
    "BACKBONE_REGISTRY",
    "mc_dropout_predict",
    "enable_dropout",
]
