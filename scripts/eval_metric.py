"""Is the pooled out-of-fold correlation measuring the model or the folds?

Every score in this evaluation pools the out-of-fold predictions into one
vector and correlates that against the target. It is the usual thing to do and
it is wrong under heavy shrinkage, which is exactly the regime the stratified
analysis landed in.

The reason shows up in the controls. `length` is a single column, so a ridge
fit on it predicts an ordering monotone in length whatever the penalty, and its
Spearman correlation against the target cannot change with alpha. Measured by
pooling, it slid from -0.03 to -0.39 across the grid. That drift is not the
model: as alpha grows each fold's predictions shrink toward that fold's own
training mean, so the pooled vector ranks sequences by which fold they happened
to fall in. With homology components of very unequal mean, the fold means
differ a lot, and the pooled correlation picks that up as signal of either
sign.

The fix is to score each fold separately and average, so a comparison is only
ever made between sequences predicted by the same model. The cost is noise:
a fold holds a dozen sequences and a correlation on a dozen points is rough.
The benefit is that the number means what it says.

This script prints both metrics across the whole alpha grid, for each feature
set, so the size of the artefact is visible rather than argued about.

Usage:
    python scripts/eval_metric.py --embeddings logs/label_embeddings.npz \\
        --target kcat_over_km --substrate p-np-c4
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.model_selection import GroupKFold

from petabite.eval_module import labels, stats

FEATURE_SETS = ("length", "composition", "stock", "plasticesm")

# At the strong end of the grid a fold's predictions can collapse to a single
# value, which makes a rank correlation undefined. That is information, not an
# error, so it is counted rather than printed once per occurrence.
warnings.simplefilter("ignore", ConstantInputWarning)


def both_metrics(x, y, groups, alpha: float, standardise: bool,
                 n_splits: int = 5):
    """Three scorings of the same fits, for one fixed penalty.

    pooled    Out-of-fold predictions concatenated, then correlated. The
              status quo, and the one with the artefact.

    per_fold  One correlation per fold, averaged. Correct, because it only
              ever compares sequences scored by the same model, but noisy:
              a fold holds about fifteen sequences.

    centred   Predictions z-scored within their fold before concatenating.
              This removes the fold offset and scale that create the
              artefact, while keeping all 73 points for one correlation, so
              it should agree with per_fold and be steadier.

    Returns the three scores and how many folds gave a usable correlation.
    """
    splitter = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    pooled = np.zeros_like(y)
    centred = np.zeros_like(y)
    per_fold = []

    for fit, held in splitter.split(x, y, groups):
        predicted = stats.fit_predict(x[fit], y[fit], x[held], alpha, standardise)
        pooled[held] = predicted

        # Z-scoring within the fold keeps the ordering this model produced and
        # discards the fold-specific level, which is the part that leaks.
        spread = predicted.std()
        centred[held] = ((predicted - predicted.mean()) / spread
                         if spread > 1e-12 else 0.0)

        # A fold needs at least three distinct values on both sides for a
        # rank correlation to exist.
        if len(set(predicted)) >= 3 and len(set(y[held])) >= 3:
            score = spearmanr(predicted, y[held]).statistic
            if np.isfinite(score):
                per_fold.append(float(score))

    def finite(value) -> float | None:
        return float(value) if np.isfinite(value) else None

    return (finite(spearmanr(pooled, y).statistic),
            float(np.mean(per_fold)) if per_fold else None,
            finite(spearmanr(centred, y).statistic),
            len(per_fold))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--substrate", default=None)
    parser.add_argument("--raw", action="store_true",
                        help="score unstandardised features as well")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cached = np.load(args.embeddings, allow_pickle=False)
    index = {name: position for position, name in enumerate(cached["names"])}
    components = labels.read_components()

    measurements = labels.read_measurements(args.target)
    if args.substrate is not None:
        measurements = [m for m in measurements if m.substrate == args.substrate]
    per_sequence = labels.aggregate(measurements)

    usable = sorted(md5 for md5 in per_sequence if md5 in index)
    y = np.array([per_sequence[md5] for md5 in usable])
    groups = [components[md5] for md5 in usable]
    rows = np.array([index[md5] for md5 in usable], dtype=int)

    print(f"{args.target} on {args.substrate or 'pooled'}: {len(y)} sequences, "
          f"{len(set(groups))} components")

    # The target's own correlation with length, computed directly, with no
    # model in the way. The pooled metric should never stray far from this for
    # a single-column feature, and how far it does stray is the artefact.
    direct = spearmanr(cached["length"][rows].ravel(), y).statistic
    print(f"direct Spearman of length against the target: {direct:+.3f}")
    print("A single-column ridge cannot beat or drift from this, whatever alpha.\n")

    modes = [("standardised", True)] + ([("raw", False)] if args.raw else [])
    results: dict[str, dict] = {}

    for mode_name, standardise in modes:
        print(f"=== {mode_name} ===")
        results[mode_name] = {}
        for name in FEATURE_SETS:
            x = cached[name][rows]
            print(f"  {name}")
            print(f"    {'alpha':>9}  {'pooled':>8}  {'per-fold':>9}  "
                  f"{'centred':>8}  folds")
            entries = {}
            for alpha in stats.ALPHAS:
                pooled, folded, centred, n_folds = both_metrics(
                    x, y, groups, alpha, standardise)
                entries[str(alpha)] = {"pooled": pooled, "per_fold": folded,
                                       "fold_centred": centred,
                                       "usable_folds": n_folds}
                print(f"    {alpha:>9g}  {_show(pooled):>8}  {_show(folded):>9}"
                      f"  {_show(centred):>8}  {n_folds}")
            results[mode_name][name] = entries
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "target": args.target, "substrate": args.substrate,
            "n_sequences": len(y), "n_components": len(set(groups)),
            "direct_length_spearman": round(float(direct), 3),
            "metrics": results,
        }, indent=2))
        print(f"wrote {args.out}")
    return 0


def _show(value: float | None) -> str:
    return "none" if value is None else f"{value:+.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
