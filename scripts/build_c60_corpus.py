"""Build the 60%-identity pretraining corpus, tokenized and ready to train on.

Why the 60% centroids
---------------------
The published corpus is the 18,172,960 90%-identity centroids. Training on all
of them costs about 57 h per epoch on one L40S, and most of the extra data is
near-duplicates: 90%-identity neighbours carry little new signal. Collapsing to
one representative per 60%-identity cluster gives 1,814,296 sequences, about
5.7 h per epoch, with better diversity per sequence.

Choosing the representative
---------------------------
Nothing in the parquet marks which 90%-centroid represents its 60% cluster.
``c60_id`` is an independent integer index, so a rule has to be chosen, and the
choice is not cosmetic: picking the cluster's largest member yields a subset
that is 89.7% Logan with a median length of 306, because Logan dominates the
corpus and its ORFs are called stop to stop.

This script draws one member per cluster uniformly at random, seeded. That is
unbiased with respect to origin and length, and re-running with a different
``--seed`` draws a different corpus of the same size, which is how a multi-epoch
run gets fresh members instead of repeating one fixed subset.

Excluding the labelled sequences
--------------------------------
Every 60% cluster containing a sequence from the PETai label set is dropped, so
no downstream evaluation sequence is seen during pretraining in any form. This
uses the full 1,219-sequence label set rather than one particular split's
val/test, so the exclusion stays valid however the labels are re-split later.
The cost is negligible: only about 120 labelled sequences are corpus centroids
at all.

Usage:
    python scripts/build_c60_corpus.py --out /project/6122149/purav/petabite-tokenized/c60-seed0
    python scripts/build_c60_corpus.py --out ... --seed 1        # a second epoch's draw
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset

from petabite.data_module.tokenization import load_esmc_tokenizer

CENTROIDS = Path(
    "/project/6122149/purav/petadex-corpus/centroids90/centroids90.parquet"
)
LABEL_SEQUENCES = Path("/project/6122149/purav/petai-labels/combined/sequences.csv")

DEFAULT_MODEL = "biohub/ESMC-300M"
# Sequences per output shard. Keeps any single Arrow file to a sane size and
# lets the loader memory-map them independently. Kept modest because the buffer
# is Python lists of ints until it is flushed, which is several times the size
# of the Arrow file it becomes.
SHARD_SIZE = 100_000
# Rows handed to the Rust tokenizer at once.
TOKENIZE_BATCH = 20_000


def read_labelled_md5s(path: Path) -> set[str]:
    """The md5s of every labelled sequence, split-independent."""
    if not path.is_file():
        raise FileNotFoundError(
            f"label set not found at {path}. It is needed to exclude labelled "
            "sequences from the pretraining corpus."
        )
    with path.open(newline="") as handle:
        md5s = {row["seq_md5"] for row in csv.DictReader(handle)}
    print(f"label set: {len(md5s):,} sequence md5s to exclude")
    return md5s


def choose_members(
    parquet: pq.ParquetFile, labelled: set[str], seed: int
) -> tuple[dict[int, set[int]], dict]:
    """Pick one row per 60% cluster, and find the clusters to exclude.

    Reservoir sampling of size one: the nth member of a cluster replaces the
    current pick with probability 1/n. One pass, and it holds only a counter and
    a row reference per cluster rather than any sequence data.

    Returns a mapping from row-group index to the chosen local row offsets, plus
    a stats dict.
    """
    rng = random.Random(seed)
    seen: dict[int, int] = defaultdict(int)
    picked: dict[int, tuple[int, int]] = {}
    excluded: set[int] = set()

    rows_read = 0
    started = time.time()
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group, columns=["c60_id", "centroid_md5"])
        c60s = table["c60_id"].to_pylist()
        md5s = table["centroid_md5"].to_pylist()

        for local, (c60, md5) in enumerate(zip(c60s, md5s)):
            if md5 in labelled:
                excluded.add(c60)
            seen[c60] += 1
            if rng.random() < 1.0 / seen[c60]:
                picked[c60] = (group, local)

        rows_read += len(c60s)
        if (group + 1) % 10 == 0:
            print(
                f"  pass 1: {group + 1}/{parquet.num_row_groups} row groups, "
                f"{rows_read:,} rows, {len(seen):,} clusters, {time.time() - started:.0f}s",
                flush=True,
            )

    for cluster in excluded:
        picked.pop(cluster, None)

    by_group: dict[int, set[int]] = defaultdict(set)
    for group, local in picked.values():
        by_group[group].add(local)

    stats = {
        "rows_scanned": rows_read,
        "clusters_total": len(seen),
        "clusters_excluded": len(excluded),
        "sequences_selected": len(picked),
        "pass1_seconds": round(time.time() - started, 1),
    }
    print(
        f"pass 1 done: {len(seen):,} clusters, {len(excluded)} excluded for holding "
        f"a labelled sequence, {len(picked):,} selected"
    )
    return by_group, stats


def tokenize_selected(
    parquet: pq.ParquetFile,
    by_group: dict[int, set[int]],
    tokenizer,
    max_length: int,
    out: Path,
) -> dict:
    """Second pass: read the chosen rows, tokenize them, write Arrow shards."""
    out.mkdir(parents=True, exist_ok=True)

    buffer_ids: list[list[int]] = []
    buffer_len: list[int] = []
    buffer_c60: list[int] = []
    shard_index = 0
    totals = {"sequences": 0, "tokens": 0}
    started = time.time()

    def flush() -> None:
        nonlocal shard_index, buffer_ids, buffer_len, buffer_c60
        if not buffer_ids:
            return
        destination = out / f"shard_{shard_index:03d}"
        Dataset.from_dict(
            {"input_ids": buffer_ids, "length": buffer_len, "c60_id": buffer_c60}
        ).save_to_disk(str(destination))
        print(
            f"  wrote {destination.name}: {len(buffer_ids):,} sequences "
            f"({time.time() - started:.0f}s elapsed)",
            flush=True,
        )
        shard_index += 1
        buffer_ids, buffer_len, buffer_c60 = [], [], []

    for group in range(parquet.num_row_groups):
        wanted = by_group.get(group)
        if not wanted:
            continue
        table = parquet.read_row_group(
            group, columns=["c60_id", "centroid_sequence"]
        )
        c60s = table["c60_id"].to_pylist()
        sequences = table["centroid_sequence"].to_pylist()

        chosen = sorted(wanted)
        for start in range(0, len(chosen), TOKENIZE_BATCH):
            offsets = chosen[start : start + TOKENIZE_BATCH]
            batch = [sequences[i] for i in offsets]
            encoded = tokenizer(batch, truncation=True, max_length=max_length)
            for offset, ids in zip(offsets, encoded["input_ids"]):
                buffer_ids.append(ids)
                buffer_len.append(len(ids))
                buffer_c60.append(c60s[offset])
                totals["tokens"] += len(ids)
            totals["sequences"] += len(offsets)

            if len(buffer_ids) >= SHARD_SIZE:
                flush()

    flush()
    totals["pass2_seconds"] = round(time.time() - started, 1)
    totals["shards"] = shard_index
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--seed", type=int, default=0,
        help="which member of each cluster to draw; change it for a second epoch",
    )
    parser.add_argument("--centroids", type=Path, default=CENTROIDS)
    parser.add_argument("--labels", type=Path, default=LABEL_SEQUENCES)
    args = parser.parse_args()

    if not args.centroids.is_file():
        raise FileNotFoundError(f"centroid parquet not found at {args.centroids}")

    tokenizer = load_esmc_tokenizer(args.model, args.max_length)
    labelled = read_labelled_md5s(args.labels)
    parquet = pq.ParquetFile(args.centroids)
    print(
        f"{args.centroids.name}: {parquet.metadata.num_rows:,} rows in "
        f"{parquet.num_row_groups} row groups"
    )

    by_group, stats = choose_members(parquet, labelled, args.seed)
    totals = tokenize_selected(
        parquet, by_group, tokenizer, args.max_length, args.out
    )

    manifest = {
        "source": str(args.centroids),
        "unit": "one member per 60%-identity cluster, drawn uniformly at random",
        "seed": args.seed,
        "model": args.model,
        "max_length": args.max_length,
        "excluded": "60% clusters containing any PETai labelled sequence",
        **stats,
        **totals,
        "mean_tokens": round(totals["tokens"] / max(1, totals["sequences"]), 1),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print()
    print(
        f"{totals['sequences']:,} sequences, {totals['tokens']:,} tokens "
        f"(mean {manifest['mean_tokens']}) in {totals['shards']} shards"
    )
    print(f"wrote {args.out}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
