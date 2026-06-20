from petabite.utils import Registry

from .classification_head import ClassificationHead
from .regression_head import RegressionHead

HEAD_REGISTRY = Registry("head")
HEAD_REGISTRY.register("regression")(RegressionHead)
HEAD_REGISTRY.register("classification")(ClassificationHead)

__all__ = ["HEAD_REGISTRY", "RegressionHead", "ClassificationHead"]
