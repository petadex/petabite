"""Tests for the pre-tokenized corpus loader and its train/eval split.

The split carries the train lengths because reading them back off a shuffled
dataset is pathologically slow: on ``datasets`` 5.0.0 ``dataset["length"]`` is a
lazy ``Column`` costing about 3.6 ms per element, and the length-grouped sampler
indexes it once per comparison while sorting. For the real corpus that is about
23.5M lookups, roughly 24 hours, spent before the first training step.

The hazard in carrying the lengths separately is that they stop matching their
rows, which would quietly degrade batching rather than fail. That is what these
tests are mostly about.
"""

from __future__ import annotations

import numpy as np
import pytest
from datasets import Dataset

from petabite.data_module.tokenized import (
    load_tokenized,
    read_lengths,
    split_tokenized,
)


def _corpus(n: int = 500, seed: int = 0) -> Dataset:
    """A toy pre-tokenized corpus with deliberately varied lengths."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(8, 64, size=n)
    return Dataset.from_dict(
        {
            "input_ids": [list(range(int(length))) for length in lengths],
            "length": [int(length) for length in lengths],
            "c60_id": list(range(n)),
        }
    )


def test_read_lengths_matches_the_rows() -> None:
    corpus = _corpus()

    lengths = read_lengths(corpus)

    assert len(lengths) == len(corpus)
    assert [len(row["input_ids"]) for row in corpus] == lengths.tolist()


def test_read_lengths_refuses_a_shuffled_dataset() -> None:
    """A shuffled dataset's storage order no longer matches its row order.

    read_lengths reads Arrow directly, which is the only affordable way, so it
    has to refuse rather than return a silently misaligned array.
    """
    corpus = _corpus().shuffle(seed=1)

    with pytest.raises(ValueError, match="indices mapping"):
        read_lengths(corpus)


def test_split_lengths_stay_aligned_with_the_train_rows() -> None:
    """The whole point: train_lengths[i] is the length of train[i].

    If the permutation were applied to the rows but not the lengths, batching
    would group the wrong sequences together and nothing would raise.
    """
    split = split_tokenized(_corpus(), eval_size=50, seed=7)

    assert len(split.train) == 450
    assert len(split.evaluation) == 50
    assert len(split.train_lengths) == 450

    actual = [len(row["input_ids"]) for row in split.train]
    assert actual == split.train_lengths


def test_split_is_a_partition() -> None:
    """Every sequence lands in exactly one side, and none is duplicated."""
    corpus = _corpus()
    split = split_tokenized(corpus, eval_size=50, seed=7)

    train_ids = [row["c60_id"] for row in split.train]
    eval_ids = [row["c60_id"] for row in split.evaluation]

    assert set(train_ids).isdisjoint(eval_ids)
    assert sorted(train_ids + eval_ids) == list(range(len(corpus)))


def test_split_is_seeded() -> None:
    corpus = _corpus()

    first = split_tokenized(corpus, eval_size=50, seed=7)
    same = split_tokenized(corpus, eval_size=50, seed=7)
    other = split_tokenized(corpus, eval_size=50, seed=8)

    assert first.train_lengths == same.train_lengths
    assert first.train_lengths != other.train_lengths


def test_split_rejects_an_eval_size_that_eats_the_corpus() -> None:
    with pytest.raises(ValueError, match="not smaller than"):
        split_tokenized(_corpus(n=100), eval_size=100, seed=0)


def test_lengths_are_usable_by_the_length_grouped_sampler() -> None:
    """The regression guard: the sampler must get a plain list, not a Column.

    A lazy Column satisfies len() and indexing, so the sampler accepts it and
    merely becomes thousands of times slower. Asserting the concrete type is
    what catches a regression here, because behaviour alone looks fine.
    """
    from transformers.trainer_pt_utils import LengthGroupedSampler

    split = split_tokenized(_corpus(), eval_size=50, seed=7)

    assert isinstance(split.train_lengths, list)
    assert all(isinstance(value, int) for value in split.train_lengths[:10])

    sampler = LengthGroupedSampler(16, lengths=split.train_lengths)
    order = list(iter(sampler))

    assert sorted(order) == list(range(len(split.train)))


def test_load_tokenized_needs_the_expected_columns(tmp_path) -> None:
    Dataset.from_dict({"input_ids": [[1, 2]]}).save_to_disk(
        str(tmp_path / "shard_000")
    )

    with pytest.raises(ValueError, match="missing columns"):
        load_tokenized(tmp_path)


def test_load_tokenized_reports_an_empty_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="shard_"):
        load_tokenized(tmp_path)
