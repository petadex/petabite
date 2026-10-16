"""Trainer callback that persists and restores LoRA adapters.

``Trainer`` cannot save this model's adapters by itself. ESM-C arrives as an
``esm`` model rather than a transformers ``PreTrainedModel``, so
``Trainer.save_model`` falls through to dumping a raw state dict, and
``Trainer._load_from_checkpoint`` has nothing to restore on resume. Without this
callback the per-step checkpoints under ``output_dir`` hold no usable adapters,
and a resumed run would carry the optimizer's moments forward against adapters
still sitting at their zero initialisation: a corrupted resume that still
reports a plausible loss.

That is the same hazard the earlier PEFT-based version had, for a different
reason. Then, PEFT's state-dict filter kept only the ``out_proj`` adapters and
dropped the fused QKV and FFN ones, 87.5% of the trainable parameters at r=32.
Now there is one adapter family and one save path, and this callback is what
puts it into every checkpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from petabite.model_module.lora import WEIGHTS_NAME, LoraConfig, load_lora, save_lora

logger = logging.getLogger(__name__)


class LoraSaveCallback(TrainerCallback):
    """Write adapters into each checkpoint and restore them on resume.

    Args:
        lora_cfg: the config the adapters were built with. Written into each
            checkpoint's manifest so a checkpoint is self-describing.
        resume_from_checkpoint: the same value passed to ``Trainer.train``.
            ``None`` or ``False`` means a fresh run and nothing is loaded;
            ``True`` resolves the latest checkpoint under ``args.output_dir``,
            mirroring Trainer.
        require_on_resume: if True, resuming from a checkpoint with no adapter
            file raises instead of warning. Checkpoints written before this
            existed have no such file, and continuing silently would restart
            every adapter from zero against stale optimizer moments.
    """

    def __init__(
        self,
        lora_cfg: LoraConfig,
        resume_from_checkpoint: str | bool | None = None,
        require_on_resume: bool = True,
    ) -> None:
        self._lora_cfg = lora_cfg
        self._resume_from_checkpoint = resume_from_checkpoint
        self._require_on_resume = require_on_resume

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Load the adapters for a resumed run, before the first step."""
        checkpoint = self._resolve_resume_dir(args)
        if checkpoint is None:
            return

        model = kwargs.get("model")
        if model is None:
            logger.warning("No model passed to on_train_begin; skipping adapter load")
            return

        path = checkpoint / WEIGHTS_NAME
        if not path.is_file():
            message = (
                f"Resuming from {checkpoint} but {WEIGHTS_NAME} is missing. Every "
                "adapter would restart from its zero initialisation while the "
                "optimizer resumes their stale moments, which trains but means "
                "nothing. Start a fresh run, or pass require_on_resume=False to "
                "continue from zeroed adapters knowingly."
            )
            if self._require_on_resume:
                raise FileNotFoundError(message)
            logger.warning(message)
            return

        n_loaded = load_lora(model, checkpoint)
        logger.info("Restored %d adapter tensors from %s", n_loaded, checkpoint)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Mirror Trainer's checkpoint layout and write the adapters inside."""
        # Callbacks fire on every rank. Without this guard all N processes write
        # the same path concurrently and interleave into a corrupt file. Trainer
        # guards its own saves the same way. on_train_begin needs no such guard:
        # every rank must load.
        if not args.should_save:
            return

        model = kwargs.get("model")
        if model is None or args.output_dir is None:
            logger.warning("No model or output_dir on save; skipping adapter save")
            return

        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        try:
            save_lora(model, checkpoint, self._lora_cfg)
        except (RuntimeError, OSError) as exc:
            # A checkpoint-side failure should not kill a long run. The
            # end-of-training save in mlm_finetune is the backstop, and unlike
            # this one it verifies what it wrote.
            logger.error("Failed to save adapters to %s: %s", checkpoint, exc)

    def _resolve_resume_dir(self, args: TrainingArguments) -> Path | None:
        """Return the checkpoint being resumed, or None for a fresh run.

        Mirrors ``Trainer.train``: ``True`` means "the latest checkpoint under
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
