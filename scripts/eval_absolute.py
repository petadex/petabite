"""How close are the predicted values, not just how well are they ordered?

Everything else in this evaluation reports a rank correlation, because the
question was always "does this feature set carry information the controls do
not". That is the right question for comparing representations and the wrong
one for asking whether a probe is usable. A Spearman of +0.3 says the ordering
is somewhat right; it says nothing about whether a predicted kcat is within a
factor of two of the measured one.

So this reports R squared and mean absolute error against the one baseline
that matters for a deployed model: predicting the training mean and ignoring
the sequence entirely. R squared below zero means exactly that, that the
constant predictor would have done better.

Fold centring is deliberately NOT applied here, and this is the one place in
the project where that is correct. Centring z-scores each fold's predictions,
which discards the level and the scale, and level and scale are the whole
subject of an error metric. The fold offsets that centring exists to remove
are instead handled where they belong: each fold's baseline is that fold's own
training mean, so a model is credited only for what it explains beyond what
the training data alone would have told you. Rank correlations from this
script would be invalid and are not reported; they live in eval_sweep.py.

Usage:
    python scripts/eval_absolute.py --embeddings logs/label_embeddings.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from petabite.eval_module import labels, stats

FEATURE_SETS = ("length", "composition", "stock", "plasticesm")

# Which targets were log10 transformed on the way in, because an error of 0.3
# means a factor of two there and 0.3 degrees Celsius somewhere else.
LOG_TARGETS = ("pet_release_conc", "kcat", "km", "kcat_over_km")
UNITS = {
    "pet_release_conc": "log10 concentration",
    "kcat": "log10 1/s",
    "km": "log10 mM",
    "kcat_over_km": "log10 1/s/mM",
    "melting_temperature": "degrees C",
    "relative_activity": "percent",
    "weight_loss": "percent",
}


def predict_per_fold(x: np.ndarray, y: np.ndarray, groups: list[str],
                     standardise: bool, n_splits: int = 5,
                     criterion: str = "mse"):
    """Raw out-of-fold predictions, and each fold's training mean beside them.

    The training mean is carried through because it is the baseline every
    error here is measured against, and it has to be the baseline of the fold
    the row was held out of rather than the mean of the whole target.
    """
    predicted = np.zeros_like(y)
    baseline = np.zeros_like(y)
    chosen: list[float] = []
    spreads: list[float] = []
    unique = sorted(set(groups))
    splitter = GroupKFold(n_splits=min(n_splits, len(unique)))

    for fit, held in splitter.split(x, y, groups):
        inner = [groups[i] for i in fit]
        # Selection by Spearman returns the top of the grid always, so
        # the probe would return its intercept and every R squared here
        # would be 0.000 by construction. See stats.pick_alpha.
        alpha = stats.pick_alpha(x[fit], y[fit], inner, standardise,
                                 criterion=criterion)
        fold = stats.fit_predict(x[fit], y[fit], x[held], alpha, standardise)
        predicted[held] = fold
        baseline[held] = y[fit].mean()
        chosen.append(alpha)
        # How far the predictions move from the fold's training mean, relative
        # to the spread of the target. Near zero means the penalty has shrunk
        # the probe to its intercept and it is returning that mean, which is
        # the difference between "the features say nothing" and "the model was
        # never allowed to say anything".
        reach = float(np.std(fold)) / float(y.std()) if y.std() > 0 else 0.0
        spreads.append(reach)

    return predicted, baseline, chosen, spreads


def errors(y: np.ndarray, predicted: np.ndarray,
           baseline: np.ndarray) -> dict:
    """R squared and absolute error against the training-mean predictor."""
    residual = float(np.sum((y - predicted) ** 2))
    null = float(np.sum((y - baseline) ** 2))
    r2 = 1.0 - residual / null if null > 0 else float("nan")

    return {
        "r2": round(r2, 3),
        "mae": round(float(np.mean(np.abs(y - predicted))), 3),
        "mae_baseline": round(float(np.mean(np.abs(y - baseline))), 3),
        "rmse": round(float(np.sqrt(residual / len(y))), 3),
        "rmse_baseline": round(float(np.sqrt(null / len(y))), 3),
    }


def verdict(result: dict) -> str:
    """Whether anything here would be worth deploying."""
    def value(name: str) -> float:
        r2 = result["features"][name]["r2"]
        return r2 if np.isfinite(r2) else -np.inf

    best = max(result["features"], key=value)
    r2 = value(best)
    if r2 <= 0:
        return "nothing beats the training mean"
    return f"{best} explains {100 * r2:.0f}% beyond the mean"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--standardise", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cached = np.load(args.embeddings, allow_pickle=False)
    index = {name: position for position, name in enumerate(cached["names"])}
    components = labels.read_components()

    results: dict[str, dict] = {}
    for target in labels.TARGETS:
        per_sequence = labels.aggregate(
            labels.read_measurements(target, include_llm=False))
        usable = sorted(md5 for md5 in per_sequence if md5 in index)
        groups = [components[md5] for md5 in usable]
        if not labels.is_scoreable(len(usable), len(set(groups))):
            continue

        rows = np.array([index[md5] for md5 in usable], dtype=int)
        y = np.array([per_sequence[md5] for md5 in usable])
        unit = UNITS.get(target, "?")

        print(f"{target}: {len(y)} sequences, {len(set(groups))} components, "
              f"{unit}")
        print(f"  target sd {y.std():.3f}, range {y.min():.2f} to {y.max():.2f}")
        print(f"  {'features':12} {'R2':>8} {'MAE':>8} {'MAE mean':>9} "
              f"{'RMSE':>8} {'RMSE mean':>10} {'alpha':>9} {'reach':>7}")

        result = {"n_sequences": len(y), "n_components": len(set(groups)),
                  "unit": unit, "target_sd": round(float(y.std()), 3),
                  "features": {}}

        for name in FEATURE_SETS:
            predicted, baseline, alphas, spreads = predict_per_fold(
                cached[name][rows], y, groups, args.standardise)
            entry = errors(y, predicted, baseline)
            entry["median_alpha"] = float(np.median(alphas))
            entry["reach"] = round(float(np.median(spreads)), 4)
            result["features"][name] = entry
            print(f"  {name:12} {entry['r2']:>8} {entry['mae']:>8} "
                  f"{entry['mae_baseline']:>9} {entry['rmse']:>8} "
                  f"{entry['rmse_baseline']:>10} {entry['median_alpha']:>9.0f} "
                  f"{entry['reach']:>7.3f}")

        print(f"  -> {verdict(result)}")
        shrunk = [name for name, entry in result["features"].items()
                  if entry["reach"] < 0.02]
        if shrunk:
            print(f"  -> shrunk to the intercept, returning the training "
                  f"mean: {', '.join(shrunk)}")
        if target in LOG_TARGETS:
            best = max(result["features"],
                       key=lambda f: -result["features"][f]["mae"])
            factor = 10 ** result["features"][best]["mae"]
            print(f"  -> best absolute error is a factor of {factor:.1f} "
                  f"({best}), against {10 ** result['features'][best]['mae_baseline']:.1f} "
                  f"for the mean")
        print()
        results[target] = result

    print("=" * 78)
    print(f"{'target':22} {'length':>9} {'compos':>9} {'stock':>9} "
          f"{'plastic':>9}   verdict")
    for target, result in results.items():
        values = [f"{result['features'][name]['r2']:>9}"
                  for name in FEATURE_SETS]
        print(f"{target:22} " + " ".join(values) + f"   {verdict(result)}")
    print("=" * 78)
    print("R2 is against each fold's own training mean. Below zero means the")
    print("constant predictor would have done better on held-out families.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
