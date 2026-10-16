"""Tokenize the corpus once and save it to disk, so training does not re-do it.

``load_dataset`` on ``petadex/catalytic-orfs-90pid`` has to assemble 18,172,960
rows from 13 parquet shards before training can start, and the ``.map()``
tokenization that follows walks all of them again. Paying that at the start of
every job is wasted time, and it is wasted once per rung of an ablation ladder.

This script does it once. It reads the parquet shards with pyarrow, which is
fast (about 1.8 s per 295k-row row group), tokenizes with the Rust tokenizer in
batches, and writes one Arrow dataset per input shard. Training then memory-maps
the result, so startup is effectively instant and needs no tokenizer at all.

Output layout, one directory per input shard so nothing has to fit in memory:

    <out>/shard_000/   an Arrow dataset with input_ids and length
    <out>/shard_001/
    ...

Load it with ``petabite.data_module.tokenized.load_tokenized(<out>)``.

Usage:
    # small dataset for smoke tests, from the first shard only
    python scripts/pretokenize_corpus.py --out data/tokenized-smoke --limit 8192

    # the whole 90%-identity corpus
    python scripts/pretokenize_corpus.py --out /project/6122149/purav/petabite-tokenized/90pid

    # the 60%-identity subset is the better pretraining target; pass its ids
    python scripts/pretokenize_corpus.py --out ... --max-length 512
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset

from petabite.data_module.tokenization import load_esmc_tokenizer

DEFAULT_MODEL = "biohub/ESMC-300M"
DEFAULT_DATASET = "petadex/catalytic-orfs-90pid"
SEQUENCE_COLUMN = "Sequence"

# Rows handed to the tokenizer at once. The Rust tokenizer is batched, and this
# size keeps peak memory modest while still amortising the call overhead.
TOKENIZE_BATCH = 20_000


def find_shards(dataset_id: str, split: str) -> list[str]:
    """Locate the parquet shards for a split inside the Hub cache."""
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(repo_id=dataset_id, repo_type="dataset")
    pattern = str(Path(snapshot) / "data" / f"{split}-*.parquet")
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise FileNotFoundError(
            f"no parquet shards matching {pattern}. Stage the dataset first "
            "with scripts/prestage_assets.py"
        )
    return shards


def tokenize_shard(
    path: str, tokenizer, max_length: int, limit: int | None
) -> tuple[list[list[int]], list[int]]:
    """Tokenize one parquet shard, reading it a row group at a time."""
    parquet = pq.ParquetFile(path)
    all_ids: list[list[int]] = []
    all_lengths: list[int] = []

    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group, columns=[SEQUENCE_COLUMN])
        sequences = table[SEQUENCE_COLUMN].to_pylist()

        for start in range(0, len(sequences), TOKENIZE_BATCH):
            batch = sequences[start : start + TOKENIZE_BATCH]
            encoded = tokenizer(batch, truncation=True, max_length=max_length)
            all_ids.extend(encoded["input_ids"])
            all_lengths.extend(len(ids) for ids in encoded["input_ids"])

            if limit is not None and len(all_ids) >= limit:
                return all_ids[:limit], all_lengths[:limit]

    return all_ids, all_lengths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="stop after this many sequences, for building a smoke dataset",
    )
    parser.add_argument(
        "--max-shards", type=int, default=None,
        help="read at most this many input shards",
    )
    args = parser.parse_args()

    tokenizer = load_esmc_tokenizer(args.model, args.max_length)
    shards = find_shards(args.dataset, args.split)
    if args.max_shards:
        shards = shards[: args.max_shards]
    print(f"{len(shards)} input shards, writing to {args.out}")

    args.out.mkdir(parents=True, exist_ok=True)
    total_sequences = 0
    total_tokens = 0
    started = time.time()

    for index, shard in enumerate(shards):
        remaining = None if args.limit is None else args.limit - total_sequences
        if remaining is not None and remaining <= 0:
            break

        shard_started = time.time()
        ids, lengths = tokenize_shard(shard, tokenizer, args.max_length, remaining)
        if not ids:
            continue

        destination = args.out / f"shard_{index:03d}"
        Dataset.from_dict({"input_ids": ids, "length": lengths}).save_to_disk(
            str(destination)
        )

        total_sequences += len(ids)
        total_tokens += sum(lengths)
        print(
            f"  shard {index:3d}: {len(ids):>9,} sequences, "
            f"{sum(lengths):>12,} tokens, {time.time() - shard_started:6.1f}s "
            f"-> {destination.name}"
        )

    elapsed = time.time() - started
    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "max_length": args.max_length,
        "sequences": total_sequences,
        "tokens": total_tokens,
        "mean_tokens": round(total_tokens / max(1, total_sequences), 1),
        "shards": sorted(p.name for p in args.out.glob("shard_*")),
        "seconds": round(elapsed, 1),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(
        f"\n{total_sequences:,} sequences, {total_tokens:,} tokens "
        f"(mean {manifest['mean_tokens']}) in {elapsed:.1f}s"
    )
    print(f"wrote {args.out}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
