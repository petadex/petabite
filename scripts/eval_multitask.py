"""Search item 5: fit every target at once, to borrow strength across families.

This is the one remaining idea that addresses the binding constraint rather
than working around it. Every single-target analysis is limited by its
component count: `kcat` has 40, `melting_temperature` has 8, and the null
calibration showed the 95th percentile of pure noise rising from 0.126 to
0.316 as that count falls. The union of all targets covers 184 components,
because most sequences carry only one target and different targets reach
different families.

So a shared direction fitted across every target is estimated from 184
components rather than 40, even though no single target is. If there is a
direction in the representation that relates to activity at all, this is the
setting with the most evidence to find it in.

The construction. One row per sequence and target pair. Each target's values
are z-scored within that target, since a melting temperature in Celsius and a
log10 kcat cannot share a scale. The features are the sequence block, shared
across targets, joined to a target one-hot so each target keeps its own
intercept. Scoring is per target, so the result is comparable with the
single-target sweep row by row.

The homology component remains the unit of independence, and because a
sequence's rows all carry its own component they cannot straddle a fold
boundary. Fold centring is applied per fold and target rather than per fold,
because each target sits at its own level inside a fold.

Usage:
    python scripts/eval_multitask.py --embeddings logs/label_embeddings.npz
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from petabite.eval_module import labels, stats

FEATURE_SETS = ("length", "composition", "stock", "plasticesm")
CONTROLS = ("length", "composition")

# A target needs this many sequences in the stacked table to be worth scoring
# on its own row, even though all of them contribute to the shared fit.
MIN_SEQUENCES = 25


def build_table(index: dict[str, int], include_llm: bool
                ) -> tuple[list[str], list[str], np.ndarray, list[str]]:
    """One row per sequence and target, with each target z-scored separately.

    Returns the sequence of each row, the target of each row, the z-scored
    value, and the target names in a fixed order for the one-hot.
    """
    sequences: list[str] = []
    targets: list[str] = []
    values: list[float] = []

    for target in labels.TARGETS:
        measurements = labels.read_measurements(target,
                                                include_llm=include_llm)
        per_sequence = labels.aggregate(
            [m for m in measurements if m.seq_md5 in index])
        if len(per_sequence) < MIN_SEQUENCES:
            continue

        names = sorted(per_sequence)
        raw = np.array([per_sequence[name] for name in names])
        spread = float(raw.std())
        if spread < 1e-9:
            continue

        centred = (raw - raw.mean()) / spread
        sequences.extend(names)
        targets.extend([target] * len(names))
        values.extend(centred.tolist())

    order = sorted(set(targets))
    return sequences, targets, np.array(values), order


def design(sequences: list[str], targets: list[str], order: list[str],
           index: dict[str, int], block: np.ndarray | None) -> np.ndarray:
    """The sequence block beside a target one-hot, or the one-hot alone."""
    one_hot = np.zeros((len(targets), len(order)), dtype=np.float64)
    position = {name: column for column, name in enumerate(order)}
    for row, target in enumerate(targets):
        one_hot[row, position[target]] = 1.0

    if block is None:
        return one_hot
    rows = np.array([index[name] for name in sequences], dtype=int)
    return np.column_stack([np.asarray(block, dtype=np.float64)[rows],
                            one_hot])


def centre_per_fold_and_target(predicted: np.ndarray, held: np.ndarray,
                               targets: list[str]) -> np.ndarray:
    """Z-score inside each target within one fold, not across the fold.

    A fold holds rows from several targets at different levels, so centring
    the fold as a whole would leave the contrast between targets in the
    pooled vector and rank rows partly by which target they came from.
    """
    centred = np.zeros(len(held), dtype=float)
    by_target: dict[str, list[int]] = collections.defaultdict(list)
    for position, row in enumerate(held):
        by_target[targets[row]].append(position)

    for positions in by_target.values():
        chosen = np.array(positions)
        centred[chosen] = stats.centre_fold(predicted[chosen])
    return centred


def out_of_fold(x: np.ndarray, y: np.ndarray, groups: list[str],
                targets: list[str]) -> np.ndarray:
    """Grouped out-of-fold predictions for the stacked table."""
    predicted = np.zeros(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(set(groups))))

    for fit, held in splitter.split(x, y, groups):
        alpha = stats.pick_alpha(x[fit], y[fit], [groups[i] for i in fit],
                                 standardise=True)
        fold = stats.fit_predict(x[fit], y[fit], x[held], alpha,
                                 standardise=True)
        predicted[held] = centre_per_fold_and_target(fold, held, targets)
    return predicted


def score_per_target(predicted: np.ndarray, y: np.ndarray, targets: list[str],
                     groups: list[str]) -> dict[str, dict]:
    """One row per target, scored on that target's rows only."""
    by_target: dict[str, list[int]] = collections.defaultdict(list)
    for row, target in enumerate(targets):
        by_target[target].append(row)

    scored = {}
    for target, rows in sorted(by_target.items()):
        chosen = np.array(rows)
        if len(chosen) < MIN_SEQUENCES:
            continue
        own_groups = [groups[row] for row in rows]
        score = spearmanr(predicted[chosen], y[chosen]).statistic
        low, high = stats.jackknife_ci(y[chosen], predicted[chosen],
                                       own_groups)
        scored[target] = {
            "n_sequences": len(chosen),
            "n_components": len(set(own_groups)),
            "spearman": round(float(score), 3) if np.isfinite(score) else None,
            "jackknife95": [low, high],
            "permutation_p": stats.permutation_p(y[chosen], predicted[chosen],
                                                 own_groups),
        }
    return scored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--include-llm", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cached = np.load(args.embeddings, allow_pickle=False)
    index = {name: position for position, name in enumerate(cached["names"])}
    components = labels.read_components()

    sequences, targets, y, order = build_table(index, args.include_llm)
    groups = [components[name] for name in sequences]

    print(f"{len(y)} rows over {len(set(sequences))} sequences and "
          f"{len(order)} targets")
    print(f"{len(set(groups))} components in the stacked table, against "
          f"{min(len(set(g for g, t in zip(groups, targets) if t == name)) for name in order)} "
          f"to {max(len(set(g for g, t in zip(groups, targets) if t == name)) for name in order)} "
          f"for any single target")
    for name in order:
        rows = sum(1 for t in targets if t == name)
        own = len({g for g, t in zip(groups, targets) if t == name})
        print(f"  {name:22} {rows:>4} sequences, {own:>3} components")
    print()

    results: dict[str, dict] = {"shared": {}}
    for feature_name in (None,) + FEATURE_SETS:
        label = feature_name or "target_only"
        block = None if feature_name is None else cached[feature_name]
        x = design(sequences, targets, order, index, block)
        predicted = out_of_fold(x, y, groups, targets)
        scored = score_per_target(predicted, y, targets, groups)
        results["shared"][label] = {"dim": int(x.shape[1]), "targets": scored}

        overall = spearmanr(predicted, y).statistic
        print(f"{label:14} dim {x.shape[1]:>4}  pooled rho "
              f"{float(overall):+.3f}")
        for target, entry in scored.items():
            print(f"    {target:22} {str(entry['spearman']):>7}  "
                  f"p {entry['permutation_p']:>7}  "
                  f"jack {entry['jackknife95']}")
        print()

    print("=" * 78)
    print(f"{'target':22} {'length':>8} {'compos':>8} {'stock':>8} "
          f"{'plastic':>8}  verdict")
    for target in order:
        row = {}
        for feature_name in FEATURE_SETS:
            entry = results["shared"][feature_name]["targets"].get(target)
            row[feature_name] = entry["spearman"] if entry else None
        if any(value is None for value in row.values()):
            continue
        control = max(row[name] for name in CONTROLS)
        model = max(row[name] for name in FEATURE_SETS
                    if name not in CONTROLS)
        verdict = (f"model by {model - control:+.3f}" if model > control
                   else f"controls by {control - model:+.3f}")
        print(f"{target:22} {row['length']:>8} {row['composition']:>8} "
              f"{row['stock']:>8} {row['plasticesm']:>8}  {verdict}")
    print("=" * 78)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
