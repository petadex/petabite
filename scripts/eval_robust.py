"""Stress the one positive result until it breaks, or does not.

The substrate-stratified sweep found both backbones clearing both controls on
p-nitrophenyl butyrate, with bootstrap intervals excluding zero. It is the only
homology-controlled positive result the project has, so everything downstream
depends on whether it is real, and a single number from a single fold
assignment is not enough to decide that.

Four stresses, each targeting a different way the result could be an artefact.

  families     Refit with each homology component dropped in turn. If one
               family carries the whole correlation, removing it collapses
               the score, and 22 families means one of them is 5% of the
               data but could be much more of the signal.

  folds        GroupKFold assigns components to folds deterministically by
               size. Reshuffling that assignment many times gives a
               distribution rather than a point, and a result that only holds
               for one assignment is not a result.

  loco         Leave one component out, the limiting case: the most training
               data per fold and the most folds. It should agree with 5-fold.

  alphas       Score at each fixed ridge penalty rather than selecting one
               nested. A correlation that exists only at the selected alpha
               is a selection artefact.

`--standardise` repeats everything with features z-scored per fold. It
matters because a ridge penalty is not scale invariant and the cached
embeddings have a column standard deviation of 0.008, so the alpha grid means
something very different for them than for a single length column. A result
that holds both ways is not an artefact of the penalty.

Usage:
    python scripts/eval_robust.py --embeddings logs/label_embeddings.npz \\
        --target kcat_over_km --substrate p-np-c4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from petabite.eval_module import labels, stats

FEATURE_SETS = ("length", "composition", "stock", "plasticesm")


def restrict(target: str, substrate: str, include_llm: bool,
             components: dict[str, str], index: dict[str, int]):
    """Per-sequence values for one target on one substrate, aligned to cache."""
    measurements = [m for m in labels.read_measurements(target, include_llm)
                    if m.substrate == substrate]
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
    """Pooled out-of-fold Spearman for a given fold generator."""
    predicted = np.zeros_like(y)
    seen = np.zeros(len(y), dtype=bool)

    for fit, held in folds:
        inner = [groups[i] for i in fit]
        alpha = stats.pick_alpha(x[fit], y[fit], inner, standardise)
        predicted[held] = stats.fit_predict(x[fit], y[fit], x[held], alpha,
                                             standardise)
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
            predicted[held] = stats.fit_predict(x[fit], y[fit], x[held], alpha,
                                                 standardise)
        score = spearmanr(predicted, y).statistic
        out[str(alpha)] = round(float(score), 3) if np.isfinite(score) else None
    return out


def examine(name: str, x, y, groups, standardise: bool) -> dict:
    """Every stress, for one feature set."""
    print(f"  {name}")

    headline = stats.grouped_out_of_fold(x, y, groups, n_splits=5,
                                         standardise=standardise)
    rho = spearmanr(headline, y).statistic
    print(f"    headline grouped 5-fold rho {rho:+.3f}")

    folds = fold_sensitivity(x, y, groups, standardise)
    if folds:
        print(f"    over {folds['n_shuffles']} random fold assignments: "
              f"mean {folds['mean']:+.3f} sd {folds['sd']:.3f}, "
              f"range {folds['min']:+.3f} to {folds['max']:+.3f}, "
              f"{folds['share_above_zero']:.0%} above zero")

    loco_predicted = stats.grouped_out_of_fold(x, y, groups, n_splits=None,
                                               standardise=standardise)
    loco = spearmanr(loco_predicted, y).statistic
    loco = float(loco) if np.isfinite(loco) else None
    print(f"    leave-one-component-out rho {loco:+.3f}")

    dropped = drop_one_family(x, y, groups, standardise)
    if dropped:
        print(f"    dropping any one family: median {dropped['median']:+.3f}, "
              f"worst {dropped['min']:+.3f} without {dropped['most_influential_component']}")

    alphas = alpha_sensitivity(x, y, groups, standardise)
    shown = ", ".join(f"{a}:{v:+.2f}" for a, v in alphas.items() if v is not None)
    print(f"    by fixed alpha: {shown}")

    return {
        "headline_kfold5": round(float(rho), 3) if np.isfinite(rho) else None,
        "loco": round(loco, 3) if loco is not None else None,
        "fold_sensitivity": folds,
        "drop_one_family": dropped,
        "alpha_sensitivity": alphas,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--substrate", required=True)
    parser.add_argument("--include-llm", action="store_true")
    parser.add_argument("--standardise", action="store_true",
                        help="z-score features per fold, so the alpha grid "
                             "means the same thing for every feature set")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cached = np.load(args.embeddings, allow_pickle=False)
    index = {name: position for position, name in enumerate(cached["names"])}
    features = {key: cached[key] for key in FEATURE_SETS}
    components = labels.read_components()

    y, groups, rows = restrict(args.target, args.substrate, args.include_llm,
                               components, index)
    print(f"{args.target} on {args.substrate}: {len(y)} sequences, "
          f"{len(set(groups))} components\n")

    print(f"standardised: {args.standardise}\n")
    results = {name: examine(name, features[name][rows], y, groups,
                             args.standardise)
               for name in FEATURE_SETS}

    print()
    print("=" * 70)
    print(f"{'feature set':13} {'kfold5':>8} {'loco':>8} {'shuffled':>10} "
          f"{'worst drop':>11}")
    for name, result in results.items():
        folds = result["fold_sensitivity"]
        dropped = result["drop_one_family"]
        print(f"{name:13} {str(result['headline_kfold5']):>8} "
              f"{str(result['loco']):>8} "
              f"{str(folds.get('mean')):>10} {str(dropped.get('min')):>11}")
    print("=" * 70)
    print("A result that holds under all four columns is worth reporting.")

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
