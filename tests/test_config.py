from pathlib import Path

from hydra import compose, initialize_config_dir


def test_config_composes():
    conf_dir = str(Path("run/conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config")
    assert cfg.model.task in {"regression", "classification"}
    assert cfg.model.backbone_name == "esmc"
    assert cfg.active_learning.query_size > 0
    assert cfg.data.csv_path.endswith(".csv")
