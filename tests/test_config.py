from pathlib import Path

from hydra import compose, initialize_config_dir


def _mlm_cfg():
    conf_dir = str(Path(__file__).parent.parent / "config")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        return compose(config_name="mlm")


def test_mlm_config_composes():
    cfg = _mlm_cfg()
    assert cfg.model.model_id
    assert cfg.model.lora_r > 0
    assert cfg.data.dataset_id
    assert 0.0 < cfg.data.mlm_probability < 1.0
    assert cfg.data.max_length > 0


def test_mlm_config_targets_out_proj_and_fused_modules():
    """PEFT can only reach out_proj; the fused QKV/FFN adapters carry the rest."""
    cfg = _mlm_cfg()
    assert list(cfg.model.target_modules) == ["out_proj"]
    assert cfg.model.lora_fused is True
