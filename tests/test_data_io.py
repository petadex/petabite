"""Tests for activity-CSV reading and dataset fingerprinting.

Fixtures are generated per-test rather than read from ``data/``: that directory is
gitignored, so a committed sample CSV cannot survive a fresh clone, and a repo-relative
path would also couple the suite to the working directory pytest was invoked from.
"""

from pathlib import Path

import pytest

from petabite.data_module.utils import dataset_hash, read_activity_csv

_CSV = """id,sequence,label
a,MSKGEELFT,1.0
b,GVVPILVEL,2.5
c,DGDVNGHKF,0.25
d,SVSGEGEGD,3.0
e,ATYGKLTLK,1.75
"""


@pytest.fixture
def activity_csv(tmp_path: Path) -> Path:
    path = tmp_path / "sample_petase.csv"
    path.write_text(_CSV)
    return path


def test_read_activity_csv(activity_csv: Path):
    rows = read_activity_csv(activity_csv)
    assert len(rows) == 5
    assert set(rows[0].keys()) >= {"id", "sequence", "label"}
    assert isinstance(rows[0]["sequence"], str)
    assert isinstance(rows[0]["label"], float)
    assert rows[0]["label"] == 1.0


def test_read_activity_csv_strips_whitespace(tmp_path: Path):
    path = tmp_path / "padded.csv"
    path.write_text("sequence,label\n  ACDE  ,1.0\n")
    assert read_activity_csv(path)[0]["sequence"] == "ACDE"


def test_read_activity_csv_defaults_missing_id(tmp_path: Path):
    path = tmp_path / "no_id.csv"
    path.write_text("sequence,label\nACDE,1.0\nFGHI,2.0\n")
    assert [r["id"] for r in read_activity_csv(path)] == ["0", "1"]


def test_read_activity_csv_missing_columns(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    bad.write_text("seq,value\nABC,1.0\n")
    with pytest.raises(ValueError, match="sequence"):
        read_activity_csv(bad)


def test_read_activity_csv_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_activity_csv(tmp_path / "nope.csv")


def test_dataset_hash_stable(activity_csv: Path):
    h1 = dataset_hash(activity_csv)
    h2 = dataset_hash(activity_csv)
    assert h1 == h2 and len(h1) == 12


def test_dataset_hash_changes_with_content(tmp_path: Path):
    """The hash exists to tie a run to an assay round — it must track edits."""
    first = tmp_path / "round1.csv"
    first.write_text(_CSV)
    second = tmp_path / "round2.csv"
    second.write_text(_CSV + "f,KFICTTGKL,4.0\n")
    assert dataset_hash(first) != dataset_hash(second)
