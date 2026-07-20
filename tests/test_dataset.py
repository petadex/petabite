from petabite.data_module import DatasetFactory, make_splits


def _records():
    return [
        {"id": str(i), "sequence": "ACDEFGHIKL", "label": float(i) / 10}
        for i in range(10)
    ]


def test_dataset_factory_builds_petase_dataset():
    cls = DatasetFactory("petase")
    ds = cls(records=_records())
    assert len(ds) == 10
    sample = ds[0]
    assert "sequence" in sample and "label" in sample


def test_make_splits_partitions_without_overlap():
    train, val, test = make_splits(_records(), val_frac=0.2, test_frac=0.2, seed=1)
    ids = [r["id"] for r in train + val + test]
    assert len(ids) == len(set(ids)) == 10
