"""Read the ESMFold2 feature table and cut it into named blocks.

`scripts/fold_labels.py` writes one row per folded sequence: fifteen scalar
features and a 256-wide pooled pair embedding. Two scripts need to turn that
into feature blocks, `eval_structure.py` to score them and `eval_robust.py` to
stress whatever scored, so the reading lives here and they share it. The point
of sharing it is that a block means the same thing in both, which matters when
one script is checking the other's result.

The blocks are kept separate rather than concatenated because they say
different things, and pooling them would hide which one carried anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Which scalar columns belong to which group. Named here rather than inferred
# so that a column added to the folding script cannot quietly join a group.
GROUPS = {
    "confidence": ("ptm", "plddt_mean", "plddt_min", "plddt_sd",
                   "plddt_confident_fraction"),
    "geometry": ("radius_of_gyration", "compactness", "end_to_end",
                 "contact_density", "mean_neighbours", "sd_neighbours",
                 "surface_fraction"),
    "secondary": ("helix", "sheet", "loop"),
}

BLOCK_NAMES = tuple(GROUPS) + ("structure_all", "pair_embedding")


def load(path: Path) -> tuple[dict[str, int], np.ndarray, list[str],
                             np.ndarray | None]:
    """The folded set: an index, the scalar table, its columns, the embedding."""
    cached = np.load(path, allow_pickle=False)
    names = list(cached["names"])
    index = {name: position for position, name in enumerate(names)}
    columns = [str(c) for c in cached["columns"]]
    embedding = (cached["pair_embedding"] if "pair_embedding" in cached
                 else None)
    return index, cached["scalars"], columns, embedding


def build_blocks(scalars: np.ndarray, columns: list[str],
                 embedding: np.ndarray | None,
                 rows: np.ndarray) -> dict[str, np.ndarray]:
    """One feature block per group, plus the union and the embedding."""
    position = {name: column for column, name in enumerate(columns)}
    blocks: dict[str, np.ndarray] = {}

    for group, wanted in GROUPS.items():
        present = [position[name] for name in wanted if name in position]
        if present:
            blocks[group] = scalars[np.ix_(rows, present)].astype(np.float64)

    everything = sorted({c for wanted in GROUPS.values() for c in wanted
                         if c in position})
    blocks["structure_all"] = scalars[
        np.ix_(rows, [position[c] for c in everything])].astype(np.float64)

    if embedding is not None and embedding.shape[1] > 1:
        blocks["pair_embedding"] = embedding[rows].astype(np.float64)
    return blocks


def usable(block: np.ndarray) -> np.ndarray:
    """Rows where every column is finite.

    Secondary structure is nan where a predicted backbone was incomplete, and
    a ridge cannot be handed a nan, so those rows are dropped from that group
    only rather than from the whole table.
    """
    return np.isfinite(block).all(axis=1)
