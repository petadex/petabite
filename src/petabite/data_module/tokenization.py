"""Tokenizer loading for the ESM-C checkpoints.

``AutoTokenizer.from_pretrained("biohub/ESMC-300M", trust_remote_code=True)``
cannot work, and the failure is not obvious from the error it gives. The
checkpoint's ``config.json`` maps the Auto model classes to its remote modeling
module but has no ``AutoTokenizer`` entry, while ``tokenizer_config.json``
declares ``tokenizer_class: "EsmcTokenizer"``, a class that exists neither in
``transformers`` nor in the remote module. ``AutoTokenizer`` therefore raises::

    ValueError: Tokenizer class EsmcTokenizer does not exist or is not
    currently imported.

The checkpoint does ship a complete ``tokenizers`` file, so the fix is to build
a ``PreTrainedTokenizerFast`` from ``tokenizer.json`` directly and attach the
special tokens from ``tokenizer_config.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# The special tokens the ESM-C tokenizer config declares. Read from the
# checkpoint rather than hardcoded, so a different ESM-C revision cannot
# silently disagree with us about which id is <mask>.
SPECIAL_TOKEN_KEYS = (
    "bos_token",
    "cls_token",
    "eos_token",
    "mask_token",
    "pad_token",
    "unk_token",
)

# tokenizer_config.json ships model_max_length as 1e19, the "unset" sentinel.
# Leaving it would make truncation silently do nothing, so a real ceiling is
# required. ESM-C's config.json gives max_position_embeddings = 2048.
DEFAULT_MODEL_MAX_LENGTH = 2048


def resolve_checkpoint(model_id: str) -> Path:
    """Return the local directory for ``model_id``.

    A path that already exists is returned unchanged, which is what lets a run
    point at a locally saved checkpoint. Otherwise the Hub cache is consulted.
    ``snapshot_download`` serves from cache when ``HF_HUB_OFFLINE=1``, so this
    works on compute nodes provided the checkpoint was staged from a login
    node first.
    """
    local = Path(model_id)
    if local.is_dir():
        return local
    return Path(snapshot_download(model_id))


def load_esmc_tokenizer(
    model_id: str, max_length: int = DEFAULT_MODEL_MAX_LENGTH
) -> PreTrainedTokenizerFast:
    """Build the ESM-C tokenizer from the checkpoint's ``tokenizer.json``.

    ``max_length`` becomes the tokenizer's ``model_max_length``, so callers that
    pass ``truncation=True`` without an explicit length still get truncated.
    """
    checkpoint = resolve_checkpoint(model_id)
    tokenizer_file = checkpoint / "tokenizer.json"
    if not tokenizer_file.exists():
        raise FileNotFoundError(
            f"{model_id} has no tokenizer.json at {tokenizer_file}. The ESM-C "
            "tokenizer cannot be built without it, and AutoTokenizer cannot "
            "load this checkpoint (see this module's docstring)."
        )

    config_file = checkpoint / "tokenizer_config.json"
    config = json.loads(config_file.read_text()) if config_file.exists() else {}
    special = {key: config[key] for key in SPECIAL_TOKEN_KEYS if key in config}

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        model_max_length=max_length,
        **special,
    )

    # A missing mask token would make the MLM collator fail much later, inside
    # the first batch, with a far less obvious message.
    if tokenizer.mask_token_id is None:
        raise ValueError(
            f"{model_id} tokenizer has no mask token, so masked-LM training is "
            "not possible with it"
        )
    if tokenizer.pad_token_id is None:
        raise ValueError(f"{model_id} tokenizer has no pad token")

    logger.info(
        "Loaded ESM-C tokenizer from %s: %d tokens, mask=%d, pad=%d, max_length=%d",
        tokenizer_file, len(tokenizer), tokenizer.mask_token_id,
        tokenizer.pad_token_id, max_length,
    )
    return tokenizer
