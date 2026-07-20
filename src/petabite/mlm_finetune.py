"""Stage 1 — PlasticESM: domain-adapt ESM-C on the Logan-derived PETadex.

The PETadex (petadex/catalytic-orfs-90pid) contains ~300 million plastic-degrading
enzyme homologs mined from the Logan planetary-scale SRA assembly. This script
continues pretraining ESM-C on that unlabeled sequence corpus using the masked-LM
objective, producing PlasticESM — a foundation model that 'understands' PETase
diversity before any activity labels are introduced.

Only LoRA adapters are trained; backbone weights stay frozen. The resulting adapter
is the starting point for downstream activity work, which lives outside this repo —
this repo does MLM fine-tuning and nothing else.

Run:
    python -m petabite.mlm_finetune                       # uses config/mlm.yaml
    python -m petabite.mlm_finetune trainer.num_train_epochs=3 data.max_train_samples=100000
"""

from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import Any

import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftMixedModel, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from petabite.model_module.fused_lora import (
    FUSED_LORA_FILENAME,
    inject_fused_lora,
    save_fused_lora,
)
from petabite.trainer_module import FusedLoRASaveCallback
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


def _prepare_datasets(cfg: DictConfig, tokenizer: Any) -> tuple[Any, Any]:
    """Load, optionally subsample, and tokenize the PETadex train/eval splits.

    Returns:
        (train_dataset, eval_dataset), with the original text columns dropped so the
        collator sees only model inputs.
    """
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
    return train_ds, eval_ds


def _build_lora_model(cfg: DictConfig) -> tuple[PeftModel | PeftMixedModel, bool]:
    """Load ESM-C and attach both adapter families.

    Order matters: the fused adapters must be injected before ``get_peft_model``,
    whose ``_mark_only_adapters_as_trainable`` freezes every parameter whose name
    lacks the ``lora_`` prefix.

    Returns:
        (peft_wrapped_model, lora_fused) — the flag tells the caller whether fused
        weights need saving alongside PEFT's adapter.
    """
    model = AutoModelForMaskedLM.from_pretrained(
        cfg.model.model_id, trust_remote_code=cfg.model.trust_remote_code
    )

    # Custom LoRA for fused QKV and FFN modules (not targetable via PEFT).
    lora_fused = cfg.model.get("lora_fused", False)
    if lora_fused:
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
        # ESM-C is an encoder used here for masked-LM domain adaptation. PEFT has no
        # MASKED_LM task type; FEATURE_EXTRACTION keeps the backbone head-free and,
        # critically, writes a non-null task_type to adapter_config.json (the Hub's
        # model-card validator rejects a null peft.task_type).
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, lora_fused


def _push_fused_lora(output_dir: Path, repo_id: str) -> None:
    """Upload fused_lora.safetensors; ``model.push_to_hub`` sends the adapter only."""
    from huggingface_hub import HfApi

    path = output_dir / FUSED_LORA_FILENAME
    HfApi().upload_file(
        path_or_fileobj=str(path),
        path_in_repo=FUSED_LORA_FILENAME,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload fused QKV/FFN LoRA weights",
    )
    logger.info("Pushed %s to %s", FUSED_LORA_FILENAME, repo_id)


@hydra.main(version_base=None, config_path="../../config", config_name="mlm")
def main(cfg: DictConfig) -> None:
    """Tokenize the dataset, LoRA-wrap ESM-C, and run masked-LM training."""
    setup_logging()
    set_seed(cfg.seed)
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    logger.info("Environment: %s", log_environment())
    report_to = _configure_wandb(cfg)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_id, trust_remote_code=cfg.model.trust_remote_code
    )
    train_ds, eval_ds = _prepare_datasets(cfg, tokenizer)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=cfg.data.mlm_probability
    )
    model, lora_fused = _build_lora_model(cfg)

    # --- Train ---
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_pip_freeze(output_dir / "requirements.txt")
    OmegaConf.save(cfg, output_dir / "config.yaml")

    # None/False = fresh run; True = latest checkpoint in output_dir; str = a path.
    resume = cfg.trainer.get("resume_from_checkpoint", None)

    args = _build_training_args(cfg, report_to)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
        # Trainer saves and restores the PEFT adapter only; this adds the fused
        # QKV/FFN weights to every checkpoint and reloads them on resume.
        callbacks=[FusedLoRASaveCallback(resume_from_checkpoint=resume)] if lora_fused else None,
    )
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir))  # PEFT adapter: out_proj only
    tokenizer.save_pretrained(str(output_dir))
    if lora_fused:
        save_fused_lora(model, output_dir)  # the other 87.5% of trained params

    if cfg.push_to_hub.enabled:
        logger.info("Pushing LoRA adapter to %s", cfg.push_to_hub.repo_id)
        model.push_to_hub(cfg.push_to_hub.repo_id, private=cfg.push_to_hub.private)
        tokenizer.push_to_hub(cfg.push_to_hub.repo_id, private=cfg.push_to_hub.private)
        if lora_fused:
            _push_fused_lora(output_dir, cfg.push_to_hub.repo_id)


if __name__ == "__main__":
    main()
