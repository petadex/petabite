"""Stage 1, PlasticESM: domain-adapt ESM-C on the Logan-derived PETadex.

The PETadex (``petadex/catalytic-orfs-90pid``) holds the 90%-identity centroids
of the plastic-degrading enzyme homologs mined from the Logan planetary-scale
SRA assembly. This script continues pretraining ESM-C on that unlabeled corpus
with the masked-LM objective, producing PlasticESM: a backbone that has seen
PETase diversity before any activity label is introduced.

Only LoRA adapters train; the backbone stays frozen. The resulting adapter is
the starting point for downstream activity work, which lives outside this repo.

Run:
    python -m petabite.mlm_finetune                      # uses config/mlm.yaml
    python -m petabite.mlm_finetune trainer.max_steps=100 data.max_train_samples=50000

Both the checkpoint and the corpus must be in ``HF_HOME`` before a job starts.
Stage them from a login node with ``scripts/prestage_assets.py``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
import torch
from datasets import Dataset, DatasetDict, load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from petabite.data_module import load_esmc_tokenizer, load_tokenized, split_tokenized
from petabite.model_module.lora import (
    LoraConfig,
    apply_lora,
    assert_adapters_match,
    save_lora,
)
from petabite.trainer_module import LoraSaveCallback
from petabite.utils import dump_pip_freeze, log_environment, set_seed, setup_logging

logger = logging.getLogger(__name__)

# Column the tokenizer adds so `group_by_length` can bucket without rescanning
# the dataset. Must match `trainer.length_column_name` in the config.
LENGTH_COLUMN = "length"


# --------------------------------------------------------------------------
# Run setup
# --------------------------------------------------------------------------


def configure_wandb(cfg_wandb: DictConfig) -> str:
    """Set the W&B environment and return the Trainer's ``report_to`` value.

    Compute nodes on this cluster do have outbound network, verified from
    kn089, so runs stream live. ``mode: offline`` remains available for a node
    that turns out to be isolated; such a run lands under ``wandb/`` and is
    uploaded later with ``wandb sync``.
    """
    if not cfg_wandb.enabled:
        os.environ["WANDB_MODE"] = "disabled"
        return "none"

    os.environ["WANDB_PROJECT"] = cfg_wandb.project
    if cfg_wandb.get("entity"):
        os.environ["WANDB_ENTITY"] = cfg_wandb.entity

    mode = cfg_wandb.get("mode") or os.environ.get("WANDB_MODE") or "online"
    os.environ["WANDB_MODE"] = mode
    if mode == "offline":
        logger.info("W&B is offline; upload later with `wandb sync wandb/offline-run-*`")
    return "wandb"


def build_training_args(cfg: DictConfig, report_to: str) -> TrainingArguments:
    """Build TrainingArguments from the ``trainer`` block.

    Every key is read explicitly rather than splatted, so a typo in the YAML
    raises here instead of being silently ignored by TrainingArguments.
    """
    trainer_cfg = cfg.trainer
    return TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=trainer_cfg.num_train_epochs,
        max_steps=trainer_cfg.max_steps,
        per_device_train_batch_size=trainer_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=trainer_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=trainer_cfg.gradient_accumulation_steps,
        learning_rate=trainer_cfg.learning_rate,
        weight_decay=trainer_cfg.weight_decay,
        warmup_ratio=trainer_cfg.warmup_ratio,
        lr_scheduler_type=trainer_cfg.lr_scheduler_type,
        max_grad_norm=trainer_cfg.max_grad_norm,
        logging_steps=trainer_cfg.logging_steps,
        eval_strategy=trainer_cfg.eval_strategy,
        eval_steps=trainer_cfg.eval_steps,
        save_strategy=trainer_cfg.save_strategy,
        save_steps=trainer_cfg.save_steps,
        save_total_limit=trainer_cfg.save_total_limit,
        bf16=trainer_cfg.bf16,
        fp16=trainer_cfg.fp16,
        gradient_checkpointing=trainer_cfg.gradient_checkpointing,
        dataloader_num_workers=trainer_cfg.dataloader_num_workers,
        dataloader_pin_memory=trainer_cfg.dataloader_pin_memory,
        dataloader_persistent_workers=trainer_cfg.dataloader_persistent_workers,
        group_by_length=trainer_cfg.group_by_length,
        length_column_name=trainer_cfg.length_column_name,
        optim=trainer_cfg.optim,
        seed=cfg.seed,
        report_to=report_to,
        run_name=cfg.wandb.run_name,
        # The length column has to stay on the dataset for the length-grouped
        # sampler to see it, and MlmCollator strips it per batch.
        remove_unused_columns=False,
        # Masked-LM evaluation only wants perplexity. Gathering (batch, length,
        # vocab) logits across the eval set would be gigabytes.
        prediction_loss_only=True,
    )


def enable_fast_matmul() -> None:
    """Allow TF32 on the tensor cores.

    The L40S is Ada, so fp32 matmuls take the TF32 path at a large speedup for
    a precision loss that does not matter to LoRA training. No-op on hardware
    without tensor cores.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def prepare_datasets(
    cfg: DictConfig, tokenizer: PreTrainedTokenizerFast
) -> tuple[Dataset, Dataset]:
    """Return the train and eval sets, pre-tokenized if a cache was built.

    Assembling 18,172,960 rows and tokenizing them costs minutes at the start
    of every job, repeated for every rung of an ablation ladder. Build the cache
    once with ``scripts/pretokenize_corpus.py`` and set ``data.tokenized_path``.
    """
    tokenized_path = cfg.data.get("tokenized_path")
    if tokenized_path:
        corpus = load_tokenized(tokenized_path)
        corpus = _cap(corpus, cfg.data.max_train_samples, "corpus")
        eval_size = min(cfg.data.max_eval_samples, len(corpus) // 10)
        train, evaluation = split_tokenized(corpus, eval_size, cfg.seed)
        logger.info(
            "Pre-tokenized corpus: %d train, %d eval", len(train), len(evaluation)
        )
        return train, evaluation

    raw = load_dataset(cfg.data.dataset_id)
    train_raw, eval_raw = select_splits(raw, cfg)
    train = tokenize_corpus(train_raw, tokenizer, cfg, "Tokenizing train")
    evaluation = tokenize_corpus(eval_raw, tokenizer, cfg, "Tokenizing eval")
    return train, evaluation


def select_splits(corpus: DatasetDict, cfg: DictConfig) -> tuple[Dataset, Dataset]:
    """Pick the train and eval splits, carving one out if the corpus lacks it.

    ``petadex/catalytic-orfs-90pid`` ships ``train`` and ``test`` only. The
    config used to ask for ``validation``, which raised a bare ``KeyError``
    naming no alternatives, so the available splits are reported here.
    """
    train_split = cfg.data.train_split
    if train_split not in corpus:
        raise KeyError(
            f"train split {train_split!r} not in the corpus. Available: {sorted(corpus)}"
        )
    train = corpus[train_split]

    eval_split = cfg.data.eval_split
    if eval_split in corpus:
        evaluation = corpus[eval_split]
    else:
        holdout = int(cfg.data.eval_holdout_size)
        if len(train) <= holdout:
            raise ValueError(
                f"corpus has no {eval_split!r} split and only {len(train)} training "
                f"rows, too few to hold out {holdout}"
            )
        logger.warning(
            "No %r split in the corpus (available: %s). Holding out the last %d "
            "training rows for evaluation.",
            eval_split, sorted(corpus), holdout,
        )
        evaluation = train.select(range(len(train) - holdout, len(train)))
        train = train.select(range(len(train) - holdout))

    return (
        _cap(train, cfg.data.max_train_samples, "train"),
        _cap(evaluation, cfg.data.max_eval_samples, "eval"),
    )


def _cap(dataset: Dataset, limit: int | None, name: str) -> Dataset:
    """Truncate a split to ``limit`` rows. ``None`` means use all of it."""
    if not limit or limit >= len(dataset):
        return dataset
    logger.info("Capping %s at %d of %d rows", name, limit, len(dataset))
    return dataset.select(range(limit))


def tokenize_corpus(
    dataset: Dataset, tokenizer: PreTrainedTokenizerFast, cfg: DictConfig, desc: str
) -> Dataset:
    """Tokenize to input ids, keeping a length column for length bucketing.

    Padding is deliberately not applied here. The collator pads per batch, so
    with ``group_by_length`` a batch of short sequences stays short instead of
    every batch being padded to ``max_length``.
    """
    column = cfg.data.sequence_column
    if column not in dataset.column_names:
        raise KeyError(
            f"sequence column {column!r} not in the dataset. "
            f"Available: {dataset.column_names}"
        )

    def encode(batch: dict[str, list]) -> dict[str, list]:
        encoded = tokenizer(
            batch[column], truncation=True, max_length=cfg.data.max_length
        )
        encoded[LENGTH_COLUMN] = [len(ids) for ids in encoded["input_ids"]]
        return encoded

    return dataset.map(
        encode,
        batched=True,
        num_proc=cfg.data.num_proc,
        remove_columns=dataset.column_names,
        desc=desc,
    )


class MlmCollator:
    """Pads and masks a batch, dropping the bookkeeping columns.

    ``length`` has to stay on the dataset for the length-grouped sampler, but
    ``DataCollatorForLanguageModeling`` would try to pad it as if it were a
    sequence.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerFast, mlm_probability: float) -> None:
        self.inner = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
        )

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        trimmed = [
            {k: v for k, v in feature.items() if k != LENGTH_COLUMN}
            for feature in features
        ]
        return self.inner(trimmed)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def build_lora_model(cfg: DictConfig) -> tuple[torch.nn.Module, LoraConfig]:
    """Load ESM-C and wrap every target projection in a LoRA adapter.

    The returned object is an ``esm`` model, not a transformers
    ``PreTrainedModel``: ``biohub/ESMC-300M`` ships a remote-code shim whose
    ``from_pretrained`` hands back ``esm.models.esmc.EsmcForMaskedLM``. Its
    forward does take ``input_ids``/``attention_mask``/``labels`` and return a
    loss, which is what makes it usable with ``Trainer``, but it has no
    ``save_pretrained``, so saving goes through ``save_lora``.
    """
    model = AutoModelForMaskedLM.from_pretrained(
        cfg.model.model_id, trust_remote_code=cfg.model.trust_remote_code
    )

    lora_cfg = LoraConfig(
        r=cfg.model.lora_r,
        alpha=cfg.model.lora_alpha,
        dropout=cfg.model.lora_dropout,
        targets=tuple(cfg.model.lora_targets),
        verify=cfg.model.lora_verify,
    )
    report = apply_lora(model, lora_cfg)
    logger.info("LoRA: %s", report.summary())
    return model, lora_cfg


class MlmTrainer(Trainer):
    """Trainer that calls ESM-C's forward and reads its output by name.

    Two things need overriding because ESM-C is not a transformers model:

    - ``compute_sae`` defaults to True on ESM-C's forward, which runs a sparse
      autoencoder head that masked-LM training never reads. Measurement found
      no throughput difference, but the output is unused either way.
    - ``prediction_step`` in the base class falls back to ``outputs[1:]`` to
      locate the logits, and ``EsmcMaskedLMOutput`` is a dataclass rather than
      a tuple or a ``ModelOutput``, so evaluation died with
      ``TypeError: 'EsmcMaskedLMOutput' object is not subscriptable``.
    """

    def __init__(self, *args, compute_sae: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.compute_sae = compute_sae

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(**inputs, compute_sae=self.compute_sae)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs, compute_sae=self.compute_sae)
            loss = outputs.loss.detach()

        if prediction_loss_only:
            return loss, None, None
        return loss, outputs.logits.detach(), inputs.get("labels")


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------


def save_and_verify(model: torch.nn.Module, output_dir: Path, lora_cfg: LoraConfig) -> None:
    """Write the adapters, then prove the file matches the live model.

    The verification is the point. The previous implementation saved through
    PEFT, which kept only the ``out_proj`` adapters, and nothing ever compared
    the file against the model. A run that cannot save its adapters should fail
    at the end of training rather than be discovered weeks later.
    """
    save_lora(model, output_dir, lora_cfg)
    n_verified = assert_adapters_match(model, output_dir)
    logger.info("Saved and verified %d adapter tensors to %s", n_verified, output_dir)


def push_to_hub(output_dir: Path, repo_id: str, private: bool) -> None:
    """Upload the whole output directory.

    The model has no ``push_to_hub`` of its own, and uploading the folder keeps
    the adapters, the manifest, the tokenizer and the resolved config together,
    which is what makes a published checkpoint reproducible.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="model",
        # Per-step checkpoints are large and not what a published adapter is.
        ignore_patterns=["checkpoint-*/**"],
        commit_message="Upload LoRA adapters, manifest and tokenizer",
    )
    logger.info("Pushed %s to %s", output_dir, repo_id)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


@hydra.main(version_base="1.3", config_path="../../config", config_name="mlm")
def main(cfg: DictConfig) -> None:
    setup_logging()
    set_seed(cfg.seed)
    enable_fast_matmul()
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    logger.info("Environment: %s", log_environment())

    report_to = configure_wandb(cfg.wandb)

    # Needed either way: even a pre-tokenized corpus needs the mask token for
    # the collator.
    tokenizer = load_esmc_tokenizer(cfg.model.model_id, cfg.data.max_length)
    train_ds, eval_ds = prepare_datasets(cfg, tokenizer)
    logger.info("Training on %d sequences, evaluating on %d", len(train_ds), len(eval_ds))

    model, lora_cfg = build_lora_model(cfg)

    output_dir = Path(hydra.utils.to_absolute_path(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_pip_freeze(output_dir / "requirements.txt")
    OmegaConf.save(cfg, output_dir / "config.yaml")

    # None/False = fresh run; True = latest checkpoint in output_dir; str = a path.
    resume = cfg.trainer.get("resume_from_checkpoint", None)

    trainer = MlmTrainer(
        model=model,
        args=build_training_args(cfg, report_to),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=MlmCollator(tokenizer, cfg.data.mlm_probability),
        compute_sae=cfg.model.compute_sae,
        # Trainer cannot save or restore this model's adapters on its own, so
        # this is what puts them in every checkpoint and reloads them on resume.
        callbacks=[LoraSaveCallback(lora_cfg, resume_from_checkpoint=resume)],
    )

    trainer.train(resume_from_checkpoint=resume)
    save_and_verify(model, output_dir, lora_cfg)
    tokenizer.save_pretrained(str(output_dir))

    metrics = trainer.evaluate()
    logger.info("Final eval metrics: %s", metrics)

    if cfg.push_to_hub.enabled:
        push_to_hub(output_dir, cfg.push_to_hub.repo_id, cfg.push_to_hub.private)


if __name__ == "__main__":
    main()
