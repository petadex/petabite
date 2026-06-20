from petabite.utils import Registry

from .esmc_backbone import ESMCBackbone

BACKBONE_REGISTRY = Registry("backbone")
BACKBONE_REGISTRY.register("esmc")(ESMCBackbone)

__all__ = ["BACKBONE_REGISTRY", "ESMCBackbone"]
