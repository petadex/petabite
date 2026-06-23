from torch.utils.data import Dataset

from petabite.utils import Registry

from .petase_dataset import PETaseDataset
from .splits import make_splits
from .tokenization import ESMCTokenizerWrapper

DATASET_REGISTRY = Registry("dataset")
DATASET_REGISTRY.register("petase")(PETaseDataset)


def DatasetFactory(name: str) -> type[Dataset]:  # noqa: N802
    """Return the dataset class registered under ``name``."""
    return DATASET_REGISTRY.get(name)


__all__ = [
    "DatasetFactory",
    "DATASET_REGISTRY",
    "PETaseDataset",
    "ESMCTokenizerWrapper",
    "make_splits",
]
