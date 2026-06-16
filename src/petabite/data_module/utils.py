"""Data IO and hashing utilities."""

from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"sequence", "label"}


def read_activity_csv(path: Path) -> List[Dict[str, object]]:
    """Read a PETase activity CSV with columns sequence,label[,id].

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if required columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Activity CSV not found: {path}")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"CSV {path} missing required columns: {sorted(missing)}")
        rows: List[Dict[str, object]] = []
        for i, raw in enumerate(reader):
            rows.append(
                {
                    "id": raw.get("id", str(i)),
                    "sequence": raw["sequence"].strip(),
                    "label": float(raw["label"]),
                }
            )
    return rows


def dataset_hash(path: Path) -> str:
    """Return a 12-char SHA256 prefix of the dataset file for version tracking."""
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()[:12]
