"""How much would a conventional random split have flattered these models?

Every number in the rest of this evaluation holds whole homology components
out of each fold, which is strict. Published sequence-to-function probes
commonly report a random split instead, with close homologs of the test
proteins sitting in the training set. Those two protocols answer different
questions: a random split measures interpolation within known families, a
grouped split measures extrapolation to a family never seen.

This script runs both on the same targets with the same features, so the gap
between them can be stated as a number rather than asserted. The gap is the
part of a published-style score that is attributable to homology leakage
rather than to anything the model learned about function.

It matters for two reasons. It says whether our null results would have looked
like successes under the looser protocol, which is the honest context for
reporting them. And it puts a figure on how much of the literature's apparent
performance on these targets may be leakage.

Usage:
    python scripts/eval_leakage.py --embeddings logs/label_embeddings.npz \\
        --out logs/eval_leakage.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from petabite.eval_module import labels, stats

FEATURE_SETS = ("length", "composition", "stock", "plasticesm")


def random_out_of_fold(x: np.ndarray, y: np.ndarray, seed: int = 0,
                       n_splits: int = 5,
                       standardise: bool = False) -> np.ndarray:
    """Out-of-fold predictions from a plain shuffled split.

    No group structure at all: homologs of a held-out sequence are free to sit
    in the fold it was predicted from. This is the protocol being measured,
    not one being recommended.
    """
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    predicted = np.zeros_like(y)

    for fit, held in splitter.split(x):
        # Alpha is chosen by a nested random split, to keep the comparison
        # with the grouped protocol about the split and nothing else.
        inner = [str(i % 5) for i in range(len(fit))]
        alpha = stats.pick_alpha(x[fit], y[fit], inner, standardise)
        fold = stats.fit_predict(x[fit], y[fit], x[held], alpha, standardise)
        # Centred per fold for the same reason the grouped protocol is, so the
        # two sides of the comparison differ only in how the split was made.
        predicted[held] = stats.centre_fold(fold)
    return predicted


def nearest_training_identity(groups: list[str]) -> dict:
    """How much of each target sits in families large enough to leak.

    A sequence can only be flattered by a random split if a relative of it
    exists in the data at all. Singleton components cannot leak, so the share
    of sequences in multi-member components bounds how large the inflation
    can be.
    """
    sizes: dict[str, int] = {}
    for group in groups:
        sizes[group] = sizes.get(group, 0) + 1

    leakable = sum(count for count in sizes.values() if count > 1)
    return {
        "n_components": len(sizes),
        "largest_component": max(sizes.values()),
        "sequences_with_a_relative": leakable,
        "leakable_share": round(leakable / len(groups), 3),
    }


def compare(target: str, include_llm: bool, components: dict[str, str],
            index: dict[str, int], features: dict[str, np.ndarray],
            standardise: bool) -> dict:
    """Both protocols, every feature set, for one target."""
    measurements = labels.read_measurements(target, include_llm=include_llm)
    per_sequence = labels.aggregate(measurements)

    usable = sorted(md5 for md5 in per_sequence if md5 in index)
    groups = [components[md5] for md5 in usable]
    n_components = len(set(groups))

    print("=" * 78)
    print(f"{target}: {len(usable)} sequences, {n_components} components")
    if not labels.is_scoreable(len(usable), n_components):
        print("  too few to score")
        return {"skipped": True}

    structure = nearest_training_identity(groups)
    print(f"  {structure['leakable_share']:.1%} of sequences have a relative in "
          f"the set, so that is the ceiling on how much leakage can help")

    rows = np.array([index[md5] for md5 in usable], dtype=int)
    y = np.array([per_sequence[md5] for md5 in usable])

    result = {"n_sequences": len(usable), "n_components": n_components,
              "structure": structure, "features": {}}

    print(f"  {'feature set':13} {'random':>8} {'grouped':>8} {'inflation':>10}")
    for name in FEATURE_SETS:
        x = features[name][rows]

        # Average over several shuffles, because a single random split of a
        # couple of hundred points is itself noisy.
        random_scores = []
        for seed in range(5):
            predicted = random_out_of_fold(x, y, seed=seed,
                                           standardise=standardise)
            score = spearmanr(predicted, y).statistic
            if np.isfinite(score):
                random_scores.append(float(score))
        random_rho = float(np.mean(random_scores)) if random_scores else None

        grouped = stats.grouped_out_of_fold(x, y, groups, n_splits=5,
                                            standardise=standardise)
        grouped_rho = spearmanr(grouped, y).statistic
        grouped_rho = float(grouped_rho) if np.isfinite(grouped_rho) else None

        inflation = (round(random_rho - grouped_rho, 3)
                     if random_rho is not None and grouped_rho is not None
                     else None)
        result["features"][name] = {
            "random_spearman": round(random_rho, 3) if random_rho is not None else None,
            "random_spread": round(float(np.std(random_scores)), 3)
            if random_scores else None,
            "grouped_spearman": round(grouped_rho, 3)
            if grouped_rho is not None else None,
            "inflation": inflation,
        }
        print(f"  {name:13} {str(round(random_rho, 3)):>8} "
              f"{str(round(grouped_rho, 3)):>8} {str(inflation):>10}")

    print()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--standardise", action="store_true",
                        help="z-score features per fold, so one alpha grid "
                             "means the same thing for a length column and a "
                             "960-dimensional embedding")
    parser.add_argument("--include-llm", action="store_true")
    parser.add_argument("--targets", nargs="*", default=list(labels.TARGETS))
    args = parser.parse_args()

    cached = np.load(args.embeddings, allow_pickle=False)
    names = list(cached["names"])
    index = {name: position for position, name in enumerate(names)}
    features = {key: cached[key] for key in FEATURE_SETS}
    components = labels.read_components()

    print(f"embeddings: {len(names)} sequences")
    print(f"llm labels included: {args.include_llm}\n")

    results = {
        target: compare(target, args.include_llm, components, index, features,
                        args.standardise)
        for target in args.targets
    }

    print("=" * 78)
    print("SUMMARY: inflation from dropping the homology grouping")
    print("=" * 78)
    print(f"{'target':22} {'length':>9} {'compos':>9} {'stock':>9} {'plastic':>9}")
    for target, result in results.items():
        if result.get("skipped"):
            continue
        inflations = [result["features"][name]["inflation"] for name in FEATURE_SETS]
        cells = "".join(f"{str(value):>10}" for value in inflations)
        print(f"{target:22}{cells}")
    print("=" * 78)
    print("A large positive number means that feature set looks much better")
    print("than it is when relatives are allowed on both sides of the split.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
