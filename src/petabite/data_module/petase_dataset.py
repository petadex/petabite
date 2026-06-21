"""PETase activity dataset."""

from __future__ import annotations

import logging
from pathlib import Path

from torch.utils.data import Dataset

from .utils import read_activity_csv

logger = logging.getLogger(__name__)


class PETaseDataset(Dataset):
    """Sequence-label dataset for PETase activity.

    Provide either ``csv_path`` or pre-loaded ``records``.

    Args:
        csv_path: optional path to a CSV with sequence,label[,id].
        records: optional list of dicts with keys id,sequence,label.
    """

    def __init__(
        self,
        csv_path: Path | None = None,
        records: list[dict[str, object]] | None = None,
    ) -> None:
        if records is not None:
            self.records = records
        elif csv_path is not None:
            self.records = read_activity_csv(Path(csv_path))
        else:
            raise ValueError("PETaseDataset needs either csv_path or records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.records[idx]
