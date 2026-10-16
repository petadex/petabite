"""Load a corpus that ``scripts/pretokenize_corpus.py`` already tokenized.

Tokenizing at the start of every job means re-reading 18 million rows and
re-running the tokenizer over all of them before the first step. Doing it once
and memory-mapping the result makes startup effectively free, which matters most
for an ablation ladder where the same corpus is read several times.

The on-disk layout is one Arrow dataset per input shard, so neither writing nor
reading ever needs the whole corpus in memory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


def load_tokenized(path: str | Path) -> Dataset:
    """Concatenate the shard datasets under ``path`` into one memory-mapped set.

    ``concatenate_datasets`` on memory-mapped Arrow tables does not copy the
    data, so this stays cheap no matter how many shards there are.
    """
    path = Path(path)
    shard_dirs = sorted(p for p in path.glob("shard_*") if p.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(
            f"no shard_* directories under {path}. Build it with "
            "scripts/pretokenize_corpus.py"
        )

    shards = [load_from_disk(str(directory)) for directory in shard_dirs]
    dataset = shards[0] if len(shards) == 1 else concatenate_datasets(shards)

    manifest_path = path / MANIFEST_NAME
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        logger.info(
            "Loaded pre-tokenized corpus from %s: %d sequences, %d tokens, "
            "max_length=%s, tokenizer=%s",
            path, len(dataset), manifest.get("tokens", -1),
            manifest.get("max_length"), manifest.get("model"),
        )
    else:
        logger.info(
            "Loaded pre-tokenized corpus from %s: %d sequences (no manifest)",
            path, len(dataset),
        )

    expected = {"input_ids", "length"}
    missing = expected - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"pre-tokenized corpus at {path} is missing columns {sorted(missing)}; "
            f"it has {dataset.column_names}"
        )
    return dataset


def split_tokenized(
    dataset: Dataset, eval_size: int, seed: int
) -> tuple[Dataset, Dataset]:
    """Shuffle once, then take a held-out tail for evaluation.

    The shuffle matters because the shards arrive in corpus order, which is
    grouped by origin: the head is NCBI nr and the long tail is Logan. An
    unshuffled tail would evaluate on Logan only.
    """
    if eval_size >= len(dataset):
        raise ValueError(
            f"eval_size {eval_size} is not smaller than the corpus "
            f"({len(dataset)} sequences)"
        )
    shuffled = dataset.shuffle(seed=seed)
    evaluation = shuffled.select(range(eval_size))
    train = shuffled.select(range(eval_size, len(shuffled)))
    return train, evaluation
