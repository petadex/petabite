from .dataset import DATASET_REGISTRY, PETaseDataset
from .splits import make_al_split, make_splits


def DatasetFactory(name: str):
    """Return the dataset class registered under ``name``."""
    return DATASET_REGISTRY.get(name)


__all__ = [
    "DatasetFactory",
    "DATASET_REGISTRY",
    "PETaseDataset",
    "make_splits",
    "make_al_split",
]
