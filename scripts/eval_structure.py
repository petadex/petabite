"""Score ESMFold2 structure features against the usual controls.

Item 6 of the search space wanted structure features. The precomputed
structures it referred to do not exist locally, so `scripts/fold_labels.py`
computes them with ESMFold2 and this scores what comes out, under exactly the
protocol everything else in the evaluation uses: grouped 5-fold on homology
components, fold-centred pooled scoring, features standardised on the training
rows, a jackknife interval and a permutation test over components.

The features arrive in four groups, kept separate because they say different
things and pooling them would hide which one carried anything.

`confidence` is ESMFold2's own view of whether the sequence folds: pTM and the
spread of per-residue pLDDT. This is the group with a mechanistic story, since
an engineered variant that has broken its fold should be both inactive and
badly predicted.

`geometry` is shape, from one coordinate per residue: radius of gyration,
compactness with the length scaling divided out, contact density, burial and
surface fraction.

`secondary` is the helix, sheet and loop composition from pydssp.

`pair_embedding` is ESMFold2's own pooled pair representation, the learned
alternative to all three.

Usage:
    python scripts/eval_structure.py --structures logs/structures.npz \\
        --embeddings logs/label_embeddings.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from petabite.eval_module import labels, stats, structures

CONTROLS = ("length", "composition")
BACKBONES = ("stock", "plasticesm")


def score_block(x: np.ndarray, y: np.ndarray, groups: list[str]) -> dict:
    """The standard protocol, identical to every other score in the project."""
    keep = structures.usable(x)
    if keep.sum() < labels.MIN_SEQUENCES:
        return {"dim": int(x.shape[1]), "n": int(keep.sum()),
                "spearman": None, "note": "too few finite rows"}

    x, y = x[keep], y[keep]
    groups = [group for group, take in zip(groups, keep) if take]
    if len(set(groups)) < labels.MIN_COMPONENTS:
        return {"dim": int(x.shape[1]), "n": int(len(y)),
                "spearman": None, "note": "too few components"}

    predicted = stats.grouped_out_of_fold(x, y, groups, n_splits=5,
                                          standardise=True)
    value = spearmanr(predicted, y).statistic
    low, high = stats.jackknife_ci(y, predicted, groups)
    folds = stats.fold_scores(x, y, groups, 5, standardise=True)

    return {
        "dim": int(x.shape[1]),
        "n": int(len(y)),
        "n_components": len(set(groups)),
        "spearman": round(float(value), 3) if np.isfinite(value) else None,
        "jackknife95": [low, high],
        "permutation_p": stats.permutation_p(y, predicted, groups),
        "per_fold_mean": round(float(np.mean(folds)), 3) if folds else None,
    }


def verdict(scores: dict[str, float | None]) -> str:
    """Whether any structure block beat both controls."""
    control = max((value for name, value in scores.items()
                   if name in CONTROLS and value is not None), default=None)
    model = max((value for name, value in scores.items()
                 if name not in CONTROLS and value is not None), default=None)
    if control is None or model is None:
        return "unscoreable"
    if model > control:
        return f"structure wins by {model - control:+.3f}"
    return f"controls win by {control - model:+.3f}"


def evaluate(target: str, structure_index: dict[str, int],
             scalars: np.ndarray, columns: list[str],
             embedding: np.ndarray | None, cached,
             embedding_index: dict[str, int]) -> dict | None:
    """One target, structure blocks against controls and backbones."""
    measurements = labels.read_measurements(target)
    per_sequence = labels.aggregate(measurements)
    components = labels.read_components()

    names = sorted(name for name in per_sequence
                   if name in structure_index and name in embedding_index)
    if not names:
        print(f"{target}: no folded sequences")
        return None

    y = np.array([per_sequence[name] for name in names])
    groups = [components[name] for name in names]
    if not labels.is_scoreable(len(names), len(set(groups))):
        print(f"{target}: {len(names)} folded sequences, "
              f"{len(set(groups))} components -> too few to score")
        return None

    print(f"{target}: {len(names)} folded sequences, "
          f"{len(set(groups))} components")
    structure_rows = np.array([structure_index[name] for name in names])
    embedding_rows = np.array([embedding_index[name] for name in names])

    blocks = structures.build_blocks(scalars, columns, embedding, structure_rows)
    for name in CONTROLS + BACKBONES:
        blocks[name] = np.asarray(cached[name],
                                  dtype=np.float64)[embedding_rows]

    print(f"  {'features':18} {'dim':>5} {'n':>4} {'rho':>7} "
          f"{'per-fold':>9} {'p':>7}  jackknife 95%")
    result: dict[str, object] = {"n_sequences": len(names),
                                 "n_components": len(set(groups)),
                                 "features": {}}
    scores: dict[str, float | None] = {}

    order = list(CONTROLS) + list(BACKBONES) + [
        name for name in ("confidence", "geometry", "secondary",
                          "structure_all", "pair_embedding")
        if name in blocks]
    for name in order:
        entry = score_block(blocks[name], y, groups)
        result["features"][name] = entry
        scores[name] = entry["spearman"]
        if entry["spearman"] is None:
            print(f"  {name:18} {entry['dim']:>5} {entry.get('n', 0):>4} "
                  f"{'-':>7}  {entry.get('note', '')}")
            continue
        print(f"  {name:18} {entry['dim']:>5} {entry['n']:>4} "
              f"{entry['spearman']:>7} {str(entry['per_fold_mean']):>9} "
              f"{entry['permutation_p']:>7}  {entry['jackknife95']}")

    # The backbones are reported for context but are not the question here, so
    # the verdict is structure against the controls only.
    result["verdict"] = verdict({k: v for k, v in scores.items()
                                 if k not in BACKBONES})
    print(f"  {result['verdict']}")
    print()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--targets", default="")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    structure_index, scalars, columns, embedding = structures.load(
        args.structures)
    cached = np.load(args.embeddings, allow_pickle=False)
    embedding_index = {name: position
                       for position, name in enumerate(cached["names"])}

    print(f"{len(structure_index)} folded structures, "
          f"{len(columns)} scalar features")
    print(f"columns: {', '.join(columns)}")
    if embedding is not None:
        print(f"pair embedding: {embedding.shape}")
    print()

    chosen = ([t.strip() for t in args.targets.split(",") if t.strip()]
              or list(labels.TARGETS))
    results = {}
    for target in chosen:
        outcome = evaluate(target, structure_index, scalars, columns,
                           embedding, cached, embedding_index)
        if outcome is not None:
            results[target] = outcome

    print("=" * 78)
    print("A structure block is only interesting where it beats both controls,")
    print("and the controls are computed on exactly these folded sequences.")
    print("=" * 78)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
