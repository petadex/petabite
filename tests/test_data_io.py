from pathlib import Path

from petabite.data_module.utils import dataset_hash, read_activity_csv


def test_read_activity_csv():
    rows = read_activity_csv(Path("data/sample_petase.csv"))
    assert len(rows) == 5
    assert set(rows[0].keys()) >= {"id", "sequence", "label"}
    assert isinstance(rows[0]["sequence"], str)
    assert isinstance(rows[0]["label"], float)


def test_read_activity_csv_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("seq,value\nABC,1.0\n")
    try:
        read_activity_csv(bad)
    except ValueError as e:
        assert "sequence" in str(e)
    else:
        raise AssertionError("expected ValueError for missing columns")


def test_dataset_hash_stable():
    h1 = dataset_hash(Path("data/sample_petase.csv"))
    h2 = dataset_hash(Path("data/sample_petase.csv"))
    assert h1 == h2 and len(h1) == 12
