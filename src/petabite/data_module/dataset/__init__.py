from petabite.utils import Registry

from .petase_dataset import PETaseDataset

DATASET_REGISTRY = Registry("dataset")
DATASET_REGISTRY.register("petase")(PETaseDataset)

__all__ = ["DATASET_REGISTRY", "PETaseDataset"]
