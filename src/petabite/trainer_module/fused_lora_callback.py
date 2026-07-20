"""Trainer callback that persists and restores fused-LoRA weights.

``Trainer.save_model`` on a PeftModel writes only PEFT's own adapter tensors, which
covers ``out_proj`` and nothing else. The fused QKV and FFN adapters injected by
``inject_fused_lora`` are silently dropped — 87.5% of the trainable parameters at
r=32 on ESM-C 300M. Symmetrically, ``Trainer._load_from_checkpoint`` restores only
the PEFT adapter on resume, leaving the fused adapters at their zero-init no-op
while the optimizer carries their moments forward.

This callback closes both halves: it writes the fused weights into every checkpoint
directory and reloads them when training resumes.
"""

from __future__ import annotations

import logging
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from petabite.model_module.backbone.fused_lora import (
    FUSED_LORA_FILENAME,
    load_fused_lora,
    save_fused_lora,
)

logger = logging.getLogger(__name__)


class FusedLoRASaveCallback(TrainerCallback):
    """Save fused-LoRA weights with each checkpoint and restore them on resume.

    Args:
        resume_from_checkpoint: the same value passed to ``Trainer.train``. ``None``
            or ``False`` means a fresh run and nothing is loaded; ``True`` resolves
            the latest checkpoint under ``args.output_dir``, mirroring Trainer.
        require_on_resume: if True, resuming from a checkpoint that has no
            ``fused_lora.safetensors`` raises instead of warning. Checkpoints written
            before this callback existed have no such file, and silently continuing
            would restart the QKV/FFN adapters from zero against stale optimizer
            moments — a corrupted resume that still reports a plausible loss.
    """

    def __init__(
        self,
        resume_from_checkpoint: str | bool | None = None,
        require_on_resume: bool = True,
    ) -> None:
        self._resume_from_checkpoint = resume_from_checkpoint
        self._require_on_resume = require_on_resume

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Reload fused weights after Trainer has restored the PEFT adapter.

        Trainer calls ``_load_from_checkpoint`` before entering the training loop, so
        by the time this fires the adapter is already in place and the fused tensors
        are the only thing left missing.
        """
        checkpoint = self._resolve_resume_dir(args)
        if checkpoint is None:
            return

        model = kwargs.get("model")
        if model is None:
            logger.warning("No model passed to on_train_begin; skipping fused-LoRA load")
            return

        path = checkpoint / FUSED_LORA_FILENAME
        if not path.is_file():
            message = (
                f"Resuming from {checkpoint} but {FUSED_LORA_FILENAME} is missing. The "
                "fused QKV/FFN adapters would restart from zero-init while the "
                "optimizer resumes their stale moments. This checkpoint predates "
                "fused-LoRA saving. Restart training from scratch, or pass "
                "require_on_resume=False to continue with out_proj only."
            )
            if self._require_on_resume:
                raise FileNotFoundError(message)
            logger.warning(message)
            return

        load_fused_lora(model, path)
        logger.info("Restored fused LoRA from %s", checkpoint)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Mirror Trainer's checkpoint layout and drop the fused weights inside."""
        # Callbacks fire on every rank. Without this, all N processes write the same
        # path concurrently and interleave into a corrupt file. Trainer guards its own
        # saves the same way. (on_train_begin has no such guard: every rank must load.)
        if not args.should_save:
            return
        model = kwargs.get("model")
        if model is None or args.output_dir is None:
            logger.warning("No model or output_dir on save; skipping fused-LoRA save")
            return
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        try:
            save_fused_lora(model, ckpt_dir)
        except (RuntimeError, OSError) as exc:
            # Never kill a long training run over a checkpoint-side failure; the
            # end-of-training save in mlm_finetune is the backstop.
            logger.error("Failed to save fused LoRA to %s: %s", ckpt_dir, exc)

    def _resolve_resume_dir(self, args: TrainingArguments) -> Path | None:
        """Return the checkpoint directory being resumed, or None for a fresh run.

        Mirrors ``Trainer.train``: ``True`` means "find the latest checkpoint under
        output_dir", anything else is an explicit path.
        """
        resume = self._resume_from_checkpoint
        if resume is None or resume is False:
            return None
        if resume is True:
            if args.output_dir is None:
                return None
            last = get_last_checkpoint(args.output_dir)
            if last is None:
                logger.warning(
                    "resume_from_checkpoint=True but no checkpoint found in %s",
                    args.output_dir,
                )
                return None
            return Path(last)
        return Path(str(resume))
