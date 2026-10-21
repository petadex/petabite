"""Fold the labelled sequences with ESMFold2, and summarise each structure.

Item 6 of the search space asked for structure features from the ESMFold2
structures "already computed". They are not computed. The corpus holds a
308,394-row manifest of AlphaFold S3 keys and no coordinates, and the
per-sequence annotation layers reach only 4.7% of the labelled set, because
95% of these sequences are engineered variants that do not appear in a
metagenomic ORF corpus.

But ESMFold2 is installed and its weights are cached, so the structures can
simply be computed. That is what this does.

Only the sequences carrying a scoreable target are folded, 397 rather than
1137, since a sequence with no usable label cannot contribute to any score.
Longest first, so padding stays low and a memory problem appears in the first
structure rather than hours in, and the results are written after every batch
because a timeout that discarded the lot would be the third time that lesson
came up.

Three families of feature come out, chosen because they say different things.
Confidence is ESMFold2's own view of whether the sequence folds at all, which
is a plausible proxy for a disrupted enzyme. Geometry is the shape: how
compact, how densely packed, how buried. Secondary structure is the fold's
composition. The pair-pooled structure embedding is kept alongside them as the
learned alternative to all three.

Usage:
    python scripts/fold_labels.py --out logs/structures.npz --limit 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pydssp
import torch

from esm.models.esmfold2 import (ESMFold2InputBuilder, EsmFold2Model,
                                 ProteinInput, StructurePredictionInput)

MODEL_ID = "biohub/ESMFold2"

# Reduced from the defaults of 20 loops and 200 sampling steps. The features
# below are coarse, a radius of gyration and a contact count, and do not need
# the last fraction of coordinate accuracy; 428 structures at the default
# settings would not fit the allocation. The gate reports per-structure time
# so this choice stays visible rather than buried.
NUM_LOOPS = 4
NUM_SAMPLING_STEPS = 50

CONTACT_RADIUS = 8.0
NEIGHBOUR_RADIUS = 10.0

# How many of the longest sequences to fold before shuffling the remainder.
LONGEST_FIRST = 3


def read_fasta(path: Path) -> dict[str, str]:
    """The sequences to fold, chosen by a step in the other environment.

    This script runs in the esm 3.4.1 virtualenv, which has no petabite
    package, so it cannot reuse the label-reading code. Rather than duplicate
    which targets are scoreable and which need a log, that decision is made
    once where the evaluation makes it and handed over as a FASTA.
    """
    sequences: dict[str, str] = {}
    name, parts = None, []
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                if name:
                    sequences[name] = "".join(parts)
                name, parts = line[1:].split()[0], []
            else:
                parts.append(line.strip())
    if name:
        sequences[name] = "".join(parts)
    return sequences


def backbone_atoms(complex_) -> np.ndarray | None:
    """The N, CA, C, O coordinates per residue, for secondary structure.

    Returns None where any residue is missing one of the four, since pydssp
    needs the full backbone and a partial one would be silently wrong.
    """
    if complex_.atom_names is None:
        return None

    names = np.asarray(complex_.atom_names)
    positions = np.asarray(complex_.atom_positions, dtype=np.float32)
    wanted = ("N", "CA", "C", "O")

    out = np.zeros((len(complex_.token_to_atoms), 4, 3), dtype=np.float32)
    for token, (start, end) in enumerate(complex_.token_to_atoms):
        found = {names[i]: positions[i] for i in range(start, end)}
        if not all(name in found for name in wanted):
            return None
        for slot, name in enumerate(wanted):
            out[token, slot] = found[name]
    return out


def alpha_carbons(complex_) -> np.ndarray:
    """One coordinate per residue, the alpha carbon where there is one."""
    names = (np.asarray(complex_.atom_names)
             if complex_.atom_names is not None else None)
    positions = np.asarray(complex_.atom_positions, dtype=np.float32)

    out = []
    for start, end in complex_.token_to_atoms:
        if names is not None:
            picked = [i for i in range(start, end) if names[i] == "CA"]
            if picked:
                out.append(positions[picked[0]])
                continue
        # No atom names, or a residue without a CA: its centroid will do for
        # the coarse geometry these features describe.
        out.append(positions[start:end].mean(axis=0))
    return np.stack(out)


def geometry(coordinates: np.ndarray) -> dict[str, float]:
    """Shape descriptors from one coordinate per residue."""
    centred = coordinates - coordinates.mean(axis=0)
    gyration = float(np.sqrt((centred ** 2).sum(axis=1).mean()))

    distances = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    off_diagonal = ~np.eye(len(coordinates), dtype=bool)

    contacts = (distances < CONTACT_RADIUS) & off_diagonal
    neighbours = ((distances < NEIGHBOUR_RADIUS) & off_diagonal).sum(axis=1)

    return {
        "radius_of_gyration": gyration,
        # A compact globular protein scales as length to the 0.4 or so, and
        # dividing it out makes compactness comparable across lengths rather
        # than another way of measuring length.
        "compactness": gyration / max(len(coordinates) ** 0.4, 1e-6),
        "end_to_end": float(np.linalg.norm(coordinates[0] - coordinates[-1])),
        "contact_density": float(contacts.sum() / (2 * len(coordinates))),
        "mean_neighbours": float(neighbours.mean()),
        "sd_neighbours": float(neighbours.std()),
        "surface_fraction": float((neighbours < 8).mean()),
    }


def secondary_structure(backbone: np.ndarray | None) -> dict[str, float]:
    """Helix, sheet and loop fractions, or nan where the backbone is partial."""
    if backbone is None:
        return {"helix": float("nan"), "sheet": float("nan"),
                "loop": float("nan")}

    assignment = pydssp.assign(backbone, out_type="c3")
    letters = np.asarray(assignment)
    return {
        "helix": float((letters == "H").mean()),
        "sheet": float((letters == "E").mean()),
        "loop": float((letters == "-").mean()),
    }


def confidence(result) -> dict[str, float]:
    """ESMFold2's own view of the prediction."""
    plddt = np.asarray(result.complex.plddt, dtype=np.float32)
    # pLDDT is reported on 0 to 1 by this port and on 0 to 100 elsewhere, so
    # the scale is normalised rather than assumed.
    if np.nanmax(plddt) > 1.5:
        plddt = plddt / 100.0

    return {
        "ptm": float(result.ptm) if result.ptm is not None else float("nan"),
        "plddt_mean": float(np.nanmean(plddt)),
        "plddt_min": float(np.nanmin(plddt)),
        "plddt_sd": float(np.nanstd(plddt)),
        "plddt_confident_fraction": float(np.nanmean(plddt > 0.7)),
    }


def summarise(result) -> tuple[dict[str, float], np.ndarray | None]:
    """Every scalar feature for one structure, plus its pair embedding."""
    coordinates = alpha_carbons(result.complex)
    features = confidence(result)
    features.update(geometry(coordinates))
    features.update(secondary_structure(backbone_atoms(result.complex)))

    embedding = result.output_embedding_pair_pooled
    if embedding is not None:
        # "Pooled" here means pooled over the pair axis, not to a fixed-size
        # vector: what comes back still has a token axis, so its raveled
        # length varies with the sequence and job 6354579 died trying to stack
        # 397 different widths. Averaging over every axis but the last gives
        # one vector per sequence, of the model's embedding width.
        array = embedding.float().cpu().numpy()
        if array.ndim > 1:
            array = array.reshape(-1, array.shape[-1]).mean(axis=0)
        embedding = array.astype(np.float32)
    return features, embedding


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, required=True,
                        help="FASTA of the sequences to fold")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-loops", type=int, default=NUM_LOOPS)
    parser.add_argument("--num-sampling-steps", type=int,
                        default=NUM_SAMPLING_STEPS)
    args = parser.parse_args()

    sequences = read_fasta(args.sequences)
    # The longest few first, so a memory problem appears in the first
    # structure rather than hours in, then the rest shuffled. Folding the
    # whole set longest-first would mean a timeout left a length-biased
    # subset, and length is one of the controls this has to beat.
    by_length = sorted(sequences, key=lambda name: -len(sequences[name]))
    head, tail = by_length[:LONGEST_FIRST], by_length[LONGEST_FIRST:]
    np.random.default_rng(0).shuffle(tail)
    names = head + list(tail)
    if args.limit:
        names = names[: args.limit]
    lengths = [len(sequences[name]) for name in names]
    print(f"{len(names)} sequences to fold, from {args.sequences}")
    print(f"lengths: min {min(lengths)}, median {int(np.median(lengths))}, "
          f"max {max(lengths)}")
    print(f"{args.num_loops} loops, {args.num_sampling_steps} sampling steps")

    # The native loader, not EsmFold2HFAdapter. The adapter wraps the port in
    # transformers and job 6354254 found it needs transformers 5.16.0.dev0,
    # which this environment does not have; esm 3.4.1 ships the model itself.
    #
    # No dtype override. Job 6354314 passed bfloat16 and the distogram head
    # failed with "mat1 and mat2 must have the same dtype", because it is
    # handed an fp32 tensor. The loader's default keeps the trunk in fp32 and
    # casts only the bundled ESMC, through esmc_precision, which is the
    # combination it supports.
    model = EsmFold2Model.from_pretrained(MODEL_ID, device="cuda",
                                          esmc_precision="bf16")
    builder = ESMFold2InputBuilder()
    print("model loaded")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    scalars: list[list[float]] = []
    embeddings: list[np.ndarray] = []
    columns: list[str] | None = None

    # Resume, because job 6355499 was requeued onto another node two minutes
    # in and started again from nothing. Over a run of several hours that
    # would eventually cost the lot, and the structures already written are
    # perfectly good: folding is deterministic at a fixed seed and nothing
    # about one sequence depends on another.
    if args.out.exists():
        done = np.load(args.out, allow_pickle=False)
        rows = [str(name) for name in done["names"]]
        columns = [str(column) for column in done["columns"]]
        scalars = [list(row) for row in done["scalars"]]
        if "pair_embedding" in done:
            embeddings = [row for row in done["pair_embedding"]]
        else:
            embeddings = [np.zeros(1, dtype=np.float32)] * len(rows)
        finished = set(rows)
        names = [name for name in names if name not in finished]
        print(f"resuming: {len(finished)} already folded, "
              f"{len(names)} to go")
        if not names:
            print("nothing left to fold")
            return 0

    started = time.time()

    for position, name in enumerate(names, start=1):
        spi = StructurePredictionInput(
            sequences=[ProteinInput(id="A", sequence=sequences[name])])
        result = builder.fold(model, spi, num_loops=args.num_loops,
                              num_sampling_steps=args.num_sampling_steps,
                              include_embeddings=True, seed=0,
                              complex_id=name[:8])
        if isinstance(result, list):
            result = result[0]

        features, embedding = summarise(result)
        if columns is None:
            columns = sorted(features)
        rows.append(name)
        scalars.append([features[key] for key in columns])
        embeddings.append(embedding if embedding is not None
                          else np.zeros(1, dtype=np.float32))
        if embeddings[-1].shape != embeddings[0].shape:
            raise SystemExit(
                f"{name} gave a {embeddings[-1].shape} embedding against "
                f"{embeddings[0].shape} for the first sequence; the widths "
                f"have to match to be stacked")

        if position == 1 or position % 5 == 0 or position == len(names):
            elapsed = time.time() - started
            print(f"  {len(rows)} done, {position}/{len(names)} this run, "
                  f"{elapsed / position:.1f}s each, "
                  f"{(len(names) - position) * elapsed / position / 60:.0f}"
                  f"min left")
            payload = {
                "names": np.array(rows),
                "columns": np.array(columns),
                "scalars": np.array(scalars, dtype=np.float32),
            }
            if embeddings[0].size > 1:
                payload["pair_embedding"] = np.stack(embeddings)
            np.savez_compressed(args.out, **payload)

    print(f"wrote {args.out}")
    print(json.dumps(dict(zip(columns, scalars[0])), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
