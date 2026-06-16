from torch.utils.data import Dataset

from .dataset import DATASET_REGISTRY, PETaseDataset
from .splits import make_al_split, make_splits
from .tokenization import ESMCTokenizerWrapper


def DatasetFactory(name: str) -> type[Dataset]:  # noqa: N802
    """Return the dataset class registered under ``name``."""
    return DATASET_REGISTRY.get(name)


__all__ = [
    "DatasetFactory",
    "DATASET_REGISTRY",
    "PETaseDataset",
    "ESMCTokenizerWrapper",
    "make_splits",
    "make_al_split",
]
