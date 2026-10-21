"""Stress a candidate result until it breaks, or does not.

A single number from a single fold assignment is not enough to decide whether
a correlation is real, so anything that clears the controls comes here before
it is reported. Three stresses run inside one invocation and the fourth is the
invocation itself, repeated with and without `--standardise`.

  folds        GroupKFold assigns components to folds deterministically by
               size. Reshuffling that assignment many times gives a
               distribution rather than a point, and a result that only holds
               for one assignment is not a result.

  families     Refit with each homology component dropped in turn. If one
               family carries the whole correlation, removing it collapses
               the score. A score that *rises* when the largest family goes
               is the same warning read from the other side.

  alphas       Score at each fixed ridge penalty rather than selecting one
               nested. A correlation that exists only at the selected alpha
               is a selection artefact.

  standardise  A ridge penalty is not scale invariant, and the cached
               embeddings have a column standard deviation of 0.008 against
               42 for a radius of gyration, so one alpha grid over all of
               them compares penalties rather than representations. A result
               that holds raw and standardised is not an artefact of scale.

Every pooled score here is fold-centred, which it was not when this script was
written: `score_with_folds` and `alpha_sensitivity` pooled raw predictions,
so the three stresses disagreed with the headline they were stressing and any
number they produced before this fix is withdrawn. Leave-one-component-out is
gone for the same reason it is gone from the sweep: a fold holding a single
sequence has no within-fold ordering, so it cannot be centred and cannot be
scored.

Either feature source can be stressed. `--embeddings` gives length,
composition and the two backbones; `--structures` adds the ESMFold2 blocks.
`--substrate` is optional and restricts to one substrate stratum.

Usage:
    python scripts/eval_robust.py --embeddings logs/label_embeddings.npz \\
        --structures logs/structures/structures.npz \\
        --target melting_temperature --standardise
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from petabite.eval_module import labels, stats, structures

FEATURE_SETS = ("length", "composition", "stock", "plasticesm")


def restrict(target: str, substrate: str | None, include_llm: bool,
             components: dict[str, str], index: dict[str, int]):
    """Per-sequence values for one target, aligned to the feature cache.

    `substrate` of None takes every measurement of the target, which is what
    the sweep and the structure scoring both do. Naming one restricts to that
    stratum.
    """
    measurements = [m for m in labels.read_measurements(target, include_llm)
                    if substrate is None or m.substrate == substrate]
    per_sequence = labels.aggregate(measurements)

    usable = sorted(md5 for md5 in per_sequence if md5 in index)
    y = np.array([per_sequence[md5] for md5 in usable])
    groups = [components[md5] for md5 in usable]
    rows = np.array([index[md5] for md5 in usable], dtype=int)
    return y, groups, rows


def shuffled_group_folds(groups: list[str], rng, n_splits: int = 5):
    """Assign whole components to folds at random, then yield the folds.

    GroupKFold balances fold sizes deterministically, which is sensible but
    gives exactly one answer. Randomising the assignment is what turns the
    score into a distribution.
    """
    unique = sorted(set(groups))
    assignment = {name: int(fold) for name, fold
                  in zip(rng.permutation(unique),
                         np.arange(len(unique)) % n_splits)}
    folds = np.array([assignment[group] for group in groups])

    for fold in range(n_splits):
        held = np.flatnonzero(folds == fold)
        fit = np.flatnonzero(folds != fold)
        if len(held) and len(fit):
            yield fit, held


def score_with_folds(x, y, groups, folds, standardise: bool) -> float | None:
    """Pooled fold-centred out-of-fold Spearman, for a given fold generator.

    The centring is the same one `stats.grouped_out_of_fold` applies and is
    not optional: without it the pooled vector ranks sequences partly by
    which fold they landed in, so the stress would be measuring the fold
    assignment it is varying.
    """
    predicted = np.zeros_like(y)
    seen = np.zeros(len(y), dtype=bool)

    for fit, held in folds:
        inner = [groups[i] for i in fit]
        alpha = stats.pick_alpha(x[fit], y[fit], inner, standardise)
        fold = stats.fit_predict(x[fit], y[fit], x[held], alpha, standardise)
        predicted[held] = stats.centre_fold(fold)
        seen[held] = True

    if seen.sum() < 10:
        return None
    score = spearmanr(predicted[seen], y[seen]).statistic
    return float(score) if np.isfinite(score) else None


def drop_one_family(x, y, groups, standardise: bool) -> dict:
    """Refit with each component removed, and report the spread."""
    unique = sorted(set(groups))
    scores: dict[str, float] = {}

    for dropped in unique:
        keep = np.array([group != dropped for group in groups])
        if keep.sum() < 20 or len(set(np.array(groups)[keep])) < 5:
            continue
        kept_groups = [group for group, k in zip(groups, keep) if k]
        splitter = GroupKFold(n_splits=min(5, len(set(kept_groups))))
        folds = list(splitter.split(x[keep], y[keep], kept_groups))
        score = score_with_folds(x[keep], y[keep], kept_groups, folds,
                                 standardise)
        if score is not None:
            scores[dropped] = round(score, 3)

    if not scores:
        return {}
    values = np.array(list(scores.values()))
    worst = min(scores, key=scores.get)
    return {
        "n_refits": len(scores),
        "min": round(float(values.min()), 3),
        "median": round(float(np.median(values)), 3),
        "max": round(float(values.max()), 3),
        "most_influential_component": worst,
        "score_without_it": scores[worst],
    }


def fold_sensitivity(x, y, groups, standardise: bool,
                     n_shuffles: int = 40) -> dict:
    """Distribution of the score over random component-to-fold assignments."""
    rng = np.random.default_rng(0)
    scores = []
    for _ in range(n_shuffles):
        folds = list(shuffled_group_folds(groups, rng))
        score = score_with_folds(x, y, groups, folds, standardise)
        if score is not None:
            scores.append(score)

    if not scores:
        return {}
    values = np.array(scores)
    return {
        "n_shuffles": len(scores),
        "mean": round(float(values.mean()), 3),
        "sd": round(float(values.std()), 3),
        "min": round(float(values.min()), 3),
        "max": round(float(values.max()), 3),
        "share_above_zero": round(float((values > 0).mean()), 3),
    }


def alpha_sensitivity(x, y, groups, standardise: bool) -> dict[str, float | None]:
    """Score at each fixed penalty, with no nested selection."""
    unique = len(set(groups))
    splitter = GroupKFold(n_splits=min(5, unique))
    out: dict[str, float | None] = {}

    for alpha in stats.ALPHAS:
        predicted = np.zeros_like(y)
        for fit, held in splitter.split(x, y, groups):
            fold = stats.fit_predict(x[fit], y[fit], x[held], alpha,
                                     standardise)
            predicted[held] = stats.centre_fold(fold)
        score = spearmanr(predicted, y).statistic
        out[str(alpha)] = round(float(score), 3) if np.isfinite(score) else None
    return out


def examine(name: str, x, y, groups, standardise: bool,
            permutations: int) -> dict:
    """Every stress, for one feature set."""
    print(f"  {name}")

    headline = stats.grouped_out_of_fold(x, y, groups, n_splits=5,
                                         standardise=standardise)
    rho = spearmanr(headline, y).statistic
    low, high = stats.jackknife_ci(y, headline, groups)
    p = stats.permutation_p(y, headline, groups, n=permutations)
    print(f"    headline grouped 5-fold rho {rho:+.3f}  "
          f"jackknife [{low}, {high}]  p {p} over {permutations} draws")

    folds = fold_sensitivity(x, y, groups, standardise)
    if folds:
        print(f"    over {folds['n_shuffles']} random fold assignments: "
              f"mean {folds['mean']:+.3f} sd {folds['sd']:.3f}, "
              f"range {folds['min']:+.3f} to {folds['max']:+.3f}, "
              f"{folds['share_above_zero']:.0%} above zero")

    dropped = drop_one_family(x, y, groups, standardise)
    if dropped:
        print(f"    dropping any one family: median {dropped['median']:+.3f}, "
              f"worst {dropped['min']:+.3f} without {dropped['most_influential_component']}")

    alphas = alpha_sensitivity(x, y, groups, standardise)
    shown = ", ".join(f"{a}:{v:+.2f}" for a, v in alphas.items() if v is not None)
    print(f"    by fixed alpha: {shown}")

    return {
        "headline_kfold5": round(float(rho), 3) if np.isfinite(rho) else None,
        "jackknife95": [low, high],
        "permutation_p": p,
        "n_permutations": permutations,
        "fold_sensitivity": folds,
        "drop_one_family": dropped,
        "alpha_sensitivity": alphas,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--structures", type=Path, default=None,
                        help="the ESMFold2 feature table, to stress the "
                             "structure blocks alongside the controls")
    parser.add_argument("--target", required=True)
    parser.add_argument("--substrate", default=None,
                        help="restrict to one substrate stratum; the default "
                             "takes every measurement of the target")
    parser.add_argument("--permutations", type=int, default=1000,
                        help="draws for the permutation test; 1000 resolves a "
                             "p of 0.01 only to within a factor of two")
    parser.add_argument("--include-llm", action="store_true")
    parser.add_argument("--standardise", action="store_true",
                        help="z-score features per fold, so the alpha grid "
                             "means the same thing for every feature set")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cached = np.load(args.embeddings, allow_pickle=False)
    index = {name: position for position, name in enumerate(cached["names"])}
    components = labels.read_components()

    y, groups, rows = restrict(args.target, args.substrate, args.include_llm,
                               components, index)
    where = args.substrate or "every substrate"
    print(f"{args.target} on {where}: {len(y)} sequences, "
          f"{len(set(groups))} components")

    blocks = {name: cached[name][rows] for name in FEATURE_SETS}

    if args.structures is not None:
        # Only the folded sequences can be compared, and the controls have to
        # be cut down to exactly the same rows or the comparison is between
        # two different datasets.
        folded, scalars, columns, embedding = structures.load(args.structures)
        names = list(cached["names"])
        keep = np.array([names[row] in folded for row in rows])
        structure_rows = np.array([folded[names[row]] for row in rows[keep]],
                                  dtype=int)
        blocks = {name: block[keep] for name, block in blocks.items()}
        blocks.update(structures.build_blocks(scalars, columns, embedding,
                                              structure_rows))
        y, groups = y[keep], [g for g, k in zip(groups, keep) if k]
        print(f"folded: {len(y)} sequences, {len(set(groups))} components")

        # A block with a nan anywhere cannot be handed to a ridge, and
        # dropping those rows from one block only would again compare
        # different datasets, so they go from all of them.
        finite = np.ones(len(y), dtype=bool)
        for block in blocks.values():
            finite &= structures.usable(block)
        if not finite.all():
            blocks = {name: block[finite] for name, block in blocks.items()}
            y, groups = y[finite], [g for g, k in zip(groups, finite) if k]
            print(f"finite in every block: {len(y)} sequences, "
                  f"{len(set(groups))} components")

    print(f"\nstandardised: {args.standardise}\n")
    results = {name: examine(name, block, y, groups, args.standardise,
                             args.permutations)
               for name, block in blocks.items()}

    print()
    print("=" * 70)
    print(f"{'feature set':15} {'kfold5':>8} {'shuffled':>10} {'sd':>7} "
          f"{'worst drop':>11} {'best drop':>10}")
    for name, result in results.items():
        folds = result["fold_sensitivity"]
        dropped = result["drop_one_family"]
        print(f"{name:15} {str(result['headline_kfold5']):>8} "
              f"{str(folds.get('mean')):>10} {str(folds.get('sd')):>7} "
              f"{str(dropped.get('min')):>11} {str(dropped.get('max')):>10}")
    print("=" * 70)
    print("A result that holds across every column, and again under the other")
    print("setting of --standardise, is worth reporting. Nothing less is.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "target": args.target, "substrate": args.substrate,
            "standardised": args.standardise,
            "n_sequences": len(y), "n_components": len(set(groups)),
            "features": results,
        }, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
