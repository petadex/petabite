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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import Dataset, concatenate_datasets, load_from_disk

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"

# Column written by the corpus builders and read by length-grouped batching.
LENGTH_COLUMN = "length"


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

    expected = {"input_ids", LENGTH_COLUMN}
    missing = expected - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"pre-tokenized corpus at {path} is missing columns {sorted(missing)}; "
            f"it has {dataset.column_names}"
        )
    return dataset


def read_lengths(dataset: Dataset) -> np.ndarray:
    """Read the length column straight out of Arrow, in storage order.

    Going through ``dataset["length"]`` is not an option at corpus scale. On
    ``datasets`` 5.0.0 that returns a lazy ``Column`` rather than a list, and
    each element access costs about 3.6 ms. Anything that indexes it per
    element, which is what the length-grouped sampler does, becomes unusable:
    1.8M sequences need roughly 23.5M lookups, which is about 24 hours of doing
    nothing.

    Reading the Arrow column directly takes 0.018 s for the same data.

    The result is in *storage* order, so it only lines up with a dataset that
    has no indices mapping. :func:`split_tokenized` is what keeps the alignment
    honest for the shuffled splits.
    """
    if dataset._indices is not None:
        raise ValueError(
            "read_lengths needs a dataset in storage order, but this one carries "
            "an indices mapping from a shuffle or select. Use split_tokenized, "
            "which tracks its own permutation."
        )
    column = dataset.data.column(LENGTH_COLUMN)
    return column.to_numpy(zero_copy_only=False)


@dataclass
class TokenizedSplit:
    """A train/eval split plus the train lengths, kept aligned.

    The lengths travel with the split because the only cheap way to get them is
    to reorder the storage-order column by the same permutation used to build
    the split. Recovering them afterwards from the shuffled dataset is the slow
    path described in :func:`read_lengths`.
    """

    train: Dataset
    evaluation: Dataset
    train_lengths: list[int]


def split_tokenized(dataset: Dataset, eval_size: int, seed: int) -> TokenizedSplit:
    """Shuffle once, hold out an evaluation slice, and keep the train lengths.

    The shuffle matters because the shards arrive in corpus order, which is
    grouped by origin: the head is NCBI nr and the long tail is Logan. An
    unshuffled tail would evaluate on Logan only.

    The permutation is drawn here rather than delegated to
    ``Dataset.shuffle`` so that the same ordering can be applied to the length
    column, which is what makes length-grouped batching affordable.
    """
    if eval_size >= len(dataset):
        raise ValueError(
            f"eval_size {eval_size} is not smaller than the corpus "
            f"({len(dataset)} sequences)"
        )

    lengths = read_lengths(dataset)
    permutation = np.random.default_rng(seed).permutation(len(dataset))
    eval_indices, train_indices = permutation[:eval_size], permutation[eval_size:]

    return TokenizedSplit(
        train=dataset.select(train_indices),
        evaluation=dataset.select(eval_indices),
        train_lengths=lengths[train_indices].tolist(),
    )
