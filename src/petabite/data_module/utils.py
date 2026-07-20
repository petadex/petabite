"""Data IO utilities for wet-lab activity CSVs.

Reads and validates the CSV files produced by the iGEM wet-lab team after each
assay round. Expected columns: sequence (amino acid string), label (activity
measurement), and an optional id. The dataset_hash utility fingerprints each CSV
so training runs can be tied back to the exact assay data they were trained on.
"""

from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"sequence", "label"}


def read_activity_csv(path: Path) -> list[dict[str, object]]:
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
        rows: list[dict[str, object]] = []
        for i, raw in enumerate(reader):
            rows.append(
                {
                    "id": raw.get("id", str(i)),
                    "sequence": raw["sequence"].strip(),
                    "label": float(raw["label"]),
                }
            )
    return rows


def read_sequence_csv(path: Path) -> list[dict[str, object]]:
    """Read an unlabelled candidate CSV with columns sequence[,id].

    Inference runs on PETadex candidates that have no activity measurement yet, so
    unlike :func:`read_activity_csv` this does not require a ``label`` column. A
    ``label`` column is carried through when present so predictions can be scored
    against a held-out set.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the ``sequence`` column is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {path}")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if "sequence" not in fieldnames:
            raise ValueError(f"CSV {path} missing required column: 'sequence'")
        rows: list[dict[str, object]] = []
        for i, raw in enumerate(reader):
            row: dict[str, object] = {
                "id": raw.get("id", str(i)),
                "sequence": raw["sequence"].strip(),
            }
            if raw.get("label") not in (None, ""):
                row["label"] = float(raw["label"])
            rows.append(row)
    return rows


def dataset_hash(path: Path) -> str:
    """Return a 12-char SHA256 prefix of the dataset file for version tracking."""
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()[:12]
