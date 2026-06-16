from .backbone import BACKBONE_REGISTRY, ESMCBackbone
from .heads import HEAD_REGISTRY, ClassificationHead, RegressionHead
from .model import ESMCActivityModel
from .uncertainty import enable_dropout, mc_dropout_predict

_MODELS = {"esmc_activity": ESMCActivityModel}


def ModelFactory(name: str):
    """Return the model class registered under ``name``."""
    if name not in _MODELS:
        raise KeyError(f"Unknown model '{name}'. Valid: {sorted(_MODELS)}")
    return _MODELS[name]


def BackboneFactory(name: str):
    """Return the backbone class registered under ``name``."""
    return BACKBONE_REGISTRY.get(name)


__all__ = [
    "ModelFactory",
    "BackboneFactory",
    "ESMCActivityModel",
    "ESMCBackbone",
    "RegressionHead",
    "ClassificationHead",
    "HEAD_REGISTRY",
    "BACKBONE_REGISTRY",
    "mc_dropout_predict",
    "enable_dropout",
]
