"""Batch selection and wet-lab query export."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

Record = dict[str, object]


def select_top_k(scores: np.ndarray, k: int) -> list[int]:
    """Return indices of the top-``k`` highest scores (descending)."""
    if k > len(scores):
        raise ValueError("k exceeds number of scored items")
    return list(np.argsort(scores)[::-1][:k])


def export_query(
    pool: list[Record], indices: list[int], out_path: Path, round_idx: int
) -> Path:
    """Write selected pool records to a CSV for wet-lab labeling."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "id", "sequence"])
        for i in indices:
            rec = pool[i]
            writer.writerow([round_idx, rec.get("id", i), rec["sequence"]])
    logger.info("Wrote %d query records to %s", len(indices), out_path)
    return out_path
