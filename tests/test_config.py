from pathlib import Path

from hydra import compose, initialize_config_dir


def test_train_config_composes():
    conf_dir = str(Path(__file__).parent.parent / "config")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="train")
    assert cfg.model.task in {"regression", "classification"}
    assert cfg.model.backbone_name == "esmc"
    assert cfg.model.lora_r > 0
    assert cfg.data.csv_path.endswith(".csv")
    assert cfg.data.max_length > 0
    assert cfg.trainer.learning_rate > 0


def test_mlm_config_composes():
    conf_dir = str(Path(__file__).parent.parent / "config")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="mlm")
    assert cfg.model.lora_r > 0
    assert cfg.data.dataset_id == "petadex/catalytic-orfs-90pid"
    assert cfg.trainer.learning_rate > 0
