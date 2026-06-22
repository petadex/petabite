"""LoRA masked-language-model fine-tuning of ESM-C on the petadex catalytic ORFs.

This is *domain adaptation* (continued pretraining): the petadex dataset is a set
of unlabeled protein sequences, so we keep ESM-C's masked-LM objective and only
train LoRA adapters on top of the frozen backbone. Training is logged to W&B.

Run:
    python -m petabite.mlm_finetune                       # uses config/mlm.yaml
    python -m petabite.mlm_finetune trainer.num_train_epochs=3 data.max_train_samples=100000
"""

from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path

import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from petabite.model_module.backbone.fused_lora import inject_fused_lora
from petabite.utils import dump_pip_freeze, log_environment, set_seed, setup_logging

logger = logging.getLogger(__name__)


def _configure_wandb(cfg: DictConfig) -> str:
    """Set up W&B via env vars and return the Trainer ``report_to`` value."""
    if not cfg.wandb.enabled:
        os.environ["WANDB_DISABLED"] = "true"
        return "none"
    os.environ["WANDB_PROJECT"] = cfg.wandb.project
    if cfg.wandb.entity:
        os.environ["WANDB_ENTITY"] = cfg.wandb.entity
    return "wandb"


def _build_training_args(cfg: DictConfig, report_to: str) -> TrainingArguments:
    """Construct TrainingArguments, tolerating the eval_strategy rename."""
    kwargs = dict(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=cfg.trainer.num_train_epochs,
        per_device_train_batch_size=cfg.trainer.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.trainer.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        learning_rate=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        warmup_ratio=cfg.trainer.warmup_ratio,
        lr_scheduler_type=cfg.trainer.lr_scheduler_type,
        logging_steps=cfg.trainer.logging_steps,
        save_steps=cfg.trainer.save_steps,
        save_total_limit=cfg.trainer.save_total_limit,
        bf16=cfg.trainer.bf16,
        fp16=cfg.trainer.fp16,
        gradient_checkpointing=cfg.trainer.gradient_checkpointing,
        dataloader_num_workers=cfg.trainer.dataloader_num_workers,
        report_to=report_to,
        run_name=cfg.wandb.run_name,
        save_strategy="steps",
    )
    # `evaluation_strategy` was renamed to `eval_strategy` in transformers 4.46.
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"
    kwargs[eval_key] = "steps"
    kwargs["eval_steps"] = cfg.trainer.eval_steps
    return TrainingArguments(**kwargs)


@hydra.main(version_base=None, config_path="../../config", config_name="mlm")
def main(cfg: DictConfig) -> None:
    """Tokenize the dataset, LoRA-wrap ESM-C, and run masked-LM training."""
    setup_logging()
    set_seed(cfg.seed)
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    logger.info("Environment: %s", log_environment())
    report_to = _configure_wandb(cfg)

    # --- Data + tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_id, trust_remote_code=cfg.model.trust_remote_code
    )
    raw = load_dataset(cfg.data.dataset_id)
    train_raw = raw[cfg.data.train_split]
    eval_raw = raw[cfg.data.eval_split]
    if cfg.data.max_train_samples:
        train_raw = train_raw.select(range(min(cfg.data.max_train_samples, len(train_raw))))
    if cfg.data.max_eval_samples:
        eval_raw = eval_raw.select(range(min(cfg.data.max_eval_samples, len(eval_raw))))

    def tokenize(batch: dict[str, list]) -> dict[str, list]:
        return tokenizer(
            batch[cfg.data.sequence_column],
            truncation=True,
            max_length=cfg.data.max_length,
        )

    train_ds = train_raw.map(
        tokenize, batched=True, num_proc=cfg.data.num_proc,
        remove_columns=train_raw.column_names, desc="Tokenizing train",
    )
    eval_ds = eval_raw.map(
        tokenize, batched=True, num_proc=cfg.data.num_proc,
        remove_columns=eval_raw.column_names, desc="Tokenizing eval",
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=cfg.data.mlm_probability
    )

    # --- Model + LoRA ---
    model = AutoModelForMaskedLM.from_pretrained(
        cfg.model.model_id, trust_remote_code=cfg.model.trust_remote_code
    )

    # Custom LoRA for fused QKV and FFN modules (not targetable via PEFT).
    if cfg.model.get("lora_fused", False):
        inject_fused_lora(
            model,
            r=cfg.model.lora_r,
            alpha=cfg.model.lora_alpha,
            dropout=cfg.model.lora_dropout,
        )

    # PEFT LoRA for out_proj (standard nn.Linear in each attention block).
    lora_config = LoraConfig(
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        target_modules=list(cfg.model.target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    if cfg.trainer.gradient_checkpointing:
        # PEFT + gradient checkpointing needs input grads enabled and cache off.
        model.enable_input_require_grads()
        model.config.use_cache = False

    # --- Train ---
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_pip_freeze(output_dir / "requirements.txt")
    OmegaConf.save(cfg, output_dir / "config.yaml")

    args = _build_training_args(cfg, report_to)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))  # saves the LoRA adapter
    tokenizer.save_pretrained(str(output_dir))

    if cfg.push_to_hub.enabled:
        logger.info("Pushing LoRA adapter to %s", cfg.push_to_hub.repo_id)
        model.push_to_hub(cfg.push_to_hub.repo_id, private=cfg.push_to_hub.private)
        tokenizer.push_to_hub(cfg.push_to_hub.repo_id, private=cfg.push_to_hub.private)


if __name__ == "__main__":
    main()
