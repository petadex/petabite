"""Tests that config/mlm.yaml composes and agrees with the code reading it."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

# The projection names apply_lora knows how to wrap. A typo here would adapt
# fewer modules than intended, which is exactly the class of silent failure
# this config is guarding against.
VALID_LORA_TARGETS = {"qkv", "out_proj", "fc1", "fc2"}


def _mlm_cfg():
    conf_dir = str(Path(__file__).parent.parent / "config")
    with initialize_config_dir(version_base="1.3", config_dir=conf_dir):
        return compose(config_name="mlm")


def test_mlm_config_composes():
    cfg = _mlm_cfg()

    assert cfg.model.model_id
    assert cfg.model.lora_r > 0
    assert cfg.data.dataset_id
    assert 0.0 < cfg.data.mlm_probability < 1.0
    assert cfg.data.max_length > 0


def test_lora_targets_are_names_lora_py_knows():
    cfg = _mlm_cfg()
    targets = set(cfg.model.lora_targets)

    assert targets, "an empty target list would adapt nothing"
    assert targets <= VALID_LORA_TARGETS, f"unknown targets: {targets - VALID_LORA_TARGETS}"


def test_eval_split_is_one_the_corpus_actually_ships():
    """petadex/catalytic-orfs-90pid has train and test, and no validation.

    The config asked for `validation` until this was fixed, which raised a bare
    KeyError naming no alternatives.
    """
    cfg = _mlm_cfg()

    assert cfg.data.eval_split in {"train", "test"}


def test_wandb_mode_is_one_the_runtime_understands():
    cfg = _mlm_cfg()

    assert cfg.wandb.mode in {"online", "offline", "disabled"}


def test_trainer_block_has_every_key_build_training_args_reads():
    """build_training_args reads each of these explicitly, so a missing key is
    an AttributeError at startup rather than a silently ignored setting."""
    cfg = _mlm_cfg()
    required = {
        "num_train_epochs", "max_steps", "per_device_train_batch_size",
        "per_device_eval_batch_size", "gradient_accumulation_steps",
        "learning_rate", "weight_decay", "warmup_ratio", "lr_scheduler_type",
        "max_grad_norm", "logging_steps", "eval_strategy", "eval_steps",
        "save_strategy", "save_steps", "save_total_limit", "bf16", "fp16",
        "gradient_checkpointing", "dataloader_num_workers",
        "dataloader_pin_memory", "dataloader_persistent_workers",
        "group_by_length", "length_column_name", "optim",
    }
    missing = required - set(cfg.trainer)

    assert not missing, f"config/mlm.yaml trainer block is missing {sorted(missing)}"


def test_length_column_matches_the_one_tokenization_writes():
    """group_by_length reads this column, and tokenize_corpus writes it."""
    from petabite.mlm_finetune import LENGTH_COLUMN

    cfg = _mlm_cfg()

    assert cfg.trainer.length_column_name == LENGTH_COLUMN
