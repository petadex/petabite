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
``--seed`` draws a different corpus of the same size. Each seed is a separate
corpus directory: a multi-epoch run that sees fresh members each epoch means
training on ``c60-seed0``, then ``c60-seed1``, and so on. Nothing switches
corpora mid-run yet, and resuming a checkpoint onto a different corpus would
misalign the batches the Trainer skips, so that is not wired up here.

Excluding the labelled sequences
--------------------------------
Every 60% cluster containing a sequence from the PETai label set is dropped.
The clusters are found by looking every labelled md5 up in the full corpus spine
(``petadex-corpus-full`` ``orfs/``, which has ``md5`` and ``c60_id`` for all
307,155,746 ORFs), not in the centroid table: only about 120 labelled sequences
are 90% centroids themselves, and checking centroids alone leaves the clusters
of the other ~1,100 in the corpus, where a 90% centroid at least 90% identical
to an evaluation sequence can be drawn.

This uses the full 1,219-sequence label set rather than one particular split's
val/test, so the exclusion stays valid however the labels are re-split later.

Labelled sequences that are not in the corpus at all (literature mutants, for
example) cannot be excluded by identity here. Their count goes in the manifest
and their md5s in ``labels_not_in_corpus.txt``; keeping their homologues out
needs a sequence search against the centroids.

Usage:
    python scripts/build_c60_corpus.py --out /project/6122149/purav/petabite-tokenized/c60-seed0
    python scripts/build_c60_corpus.py --out ... --seed 1        # a different draw

The spine has to be on disk first, for example:
    hf download petadex/petadex-corpus-full --repo-type dataset --include "orfs/*" \\
        --local-dir /project/6122149/purav/petadex-corpus/full
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import Dataset

from petabite.data_module.tokenization import load_esmc_tokenizer

CENTROIDS = Path(
    "/project/6122149/purav/petadex-corpus/centroids90/centroids90.parquet"
)
ORFS = Path("/project/6122149/purav/petadex-corpus/full/orfs")
LABEL_SEQUENCES = Path("/project/6122149/purav/petai-labels/combined/sequences.csv")

DEFAULT_MODEL = "biohub/ESMC-300M"
# Sequences per output shard. Keeps any single Arrow file to a sane size and
# lets the loader memory-map them independently. Kept modest because the buffer
# is Python lists of ints until it is flushed, which is several times the size
# of the Arrow file it becomes.
SHARD_SIZE = 100_000
# Rows handed to the Rust tokenizer at once.
TOKENIZE_BATCH = 20_000


def corpus_md5(sequence: str) -> str:
    """The corpus's md5 convention: uppercase, trailing stop removed."""
    return hashlib.md5(sequence.strip().upper().rstrip("*").encode()).hexdigest()


def read_labelled_md5s(path: Path) -> set[str]:
    """The md5s of every labelled sequence, split-independent.

    When the file carries the sequences themselves, the md5 is recomputed with
    the corpus convention rather than trusted, because a label md5 taken over a
    lowercase sequence or one ending in ``*`` would silently match nothing.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"label set not found at {path}. It is needed to exclude labelled "
            "sequences from the pretraining corpus."
        )
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "sequence" in (reader.fieldnames or []):
            md5s = {corpus_md5(row["sequence"]) for row in reader if row["sequence"]}
            source = "recomputed from the sequence column"
        else:
            md5s = {row["seq_md5"].strip().lower() for row in reader}
            source = "seq_md5 column, assumed to follow the corpus convention"
    print(f"label set: {len(md5s):,} sequence md5s to exclude ({source})")
    return md5s


def find_labelled_clusters(
    orfs: Path, labelled: set[str]
) -> tuple[set[int], set[str]]:
    """The 60% clusters holding a labelled sequence anywhere, not only as a centroid.

    Scans ``md5`` and ``c60_id`` over every part of the corpus spine, one row
    group at a time. Returns the clusters to exclude and the labelled md5s that
    were found in the corpus.
    """
    parts = sorted(orfs.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"no orfs/part-*.parquet under {orfs}. Download the corpus spine "
            "(see this script's docstring); it is what maps labelled sequences "
            "to their 60% clusters."
        )

    wanted = pa.array(sorted(labelled), type=pa.string())
    clusters: set[int] = set()
    found: set[str] = set()
    started = time.time()
    for index, part in enumerate(parts):
        parquet = pq.ParquetFile(part)
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group, columns=["md5", "c60_id"])
            hits = table.filter(pc.is_in(table["md5"], value_set=wanted))
            clusters.update(hits["c60_id"].to_pylist())
            found.update(hits["md5"].to_pylist())
        if (index + 1) % 10 == 0:
            print(
                f"  labels: {index + 1}/{len(parts)} parts, {len(found):,} labelled "
                f"sequences found, {time.time() - started:.0f}s",
                flush=True,
            )

    print(
        f"labels: {len(found):,} of {len(labelled):,} labelled sequences are in the "
        f"corpus, in {len(clusters):,} 60% clusters; {len(labelled) - len(found):,} "
        "are not in the corpus and cannot be excluded by identity"
    )
    return clusters, found


def choose_members(
    parquet: pq.ParquetFile, excluded: set[int], seed: int
) -> tuple[dict[int, set[int]], dict]:
    """Pick one row per 60% cluster, leaving out the ``excluded`` clusters.

    Reservoir sampling of size one: the nth member of a cluster replaces the
    current pick with probability 1/n. One pass, and it holds only a counter and
    a row reference per cluster rather than any sequence data.

    Returns a mapping from row-group index to the chosen local row offsets, plus
    a stats dict.
    """
    rng = random.Random(seed)
    seen: dict[int, int] = defaultdict(int)
    picked: dict[int, tuple[int, int]] = {}

    rows_read = 0
    started = time.time()
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group, columns=["c60_id"])
        c60s = table["c60_id"].to_pylist()

        for local, c60 in enumerate(c60s):
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

    dropped = 0
    for cluster in excluded:
        if picked.pop(cluster, None) is not None:
            dropped += 1

    by_group: dict[int, set[int]] = defaultdict(set)
    for group, local in picked.values():
        by_group[group].add(local)

    stats = {
        "rows_scanned": rows_read,
        "clusters_total": len(seen),
        "clusters_excluded": dropped,
        "sequences_selected": len(picked),
        "pass1_seconds": round(time.time() - started, 1),
    }
    print(
        f"pass 1 done: {len(seen):,} clusters, {dropped:,} excluded for holding "
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
    parser.add_argument(
        "--orfs", type=Path, default=ORFS,
        help="the corpus spine's orfs/ directory, used to find labelled clusters",
    )
    parser.add_argument("--labels", type=Path, default=LABEL_SEQUENCES)
    args = parser.parse_args()

    if not args.centroids.is_file():
        raise FileNotFoundError(f"centroid parquet not found at {args.centroids}")

    tokenizer = load_esmc_tokenizer(args.model, args.max_length)
    labelled = read_labelled_md5s(args.labels)
    excluded, found = find_labelled_clusters(args.orfs, labelled)
    parquet = pq.ParquetFile(args.centroids)
    print(
        f"{args.centroids.name}: {parquet.metadata.num_rows:,} rows in "
        f"{parquet.num_row_groups} row groups"
    )

    by_group, stats = choose_members(parquet, excluded, args.seed)
    totals = tokenize_selected(
        parquet, by_group, tokenizer, args.max_length, args.out
    )

    missing = sorted(labelled - found)
    (args.out / "labels_not_in_corpus.txt").write_text(
        "".join(f"{md5}\n" for md5 in missing)
    )

    manifest = {
        "source": str(args.centroids),
        "unit": "one member per 60%-identity cluster, drawn uniformly at random",
        "seed": args.seed,
        "model": args.model,
        "max_length": args.max_length,
        "excluded": "60% clusters containing any PETai labelled sequence, "
        "looked up by md5 over every ORF in orfs/",
        "labels_total": len(labelled),
        "labels_in_corpus": len(found),
        "labels_not_in_corpus": len(missing),
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
