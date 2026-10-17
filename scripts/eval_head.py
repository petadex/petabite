"""Search item 7: a learned head instead of mean-pool plus ridge.

Two halves to this item, and only one of them is testable with what is
cached.

Learned pooling is not. It needs per-token hidden states, and only the
mean-pooled vectors were kept. Recomputing them for 1137 sequences at up to
1024 tokens by 960 dimensions is about 1.1 billion floats, which is a real
job rather than an analysis. It is also the half least likely to help, for a
reason that has nothing to do with pooling: a learned pooler adds parameters,
and the evidence below says the data cannot afford the parameters it already
has.

A small head is testable, and that is what this does. A one-hidden-layer
network on the same cached embeddings, against the ridge it would replace,
under the same grouped folds, the same fold centring and the same
standardisation.

The prediction, stated before the run, is that it does worse, and the reason
is already measured. The ridge alpha profile for these embeddings peaks at
the most regularised end of an eight-order grid: stock on `kcat` reads -0.306
at alpha 1 and +0.292 at alpha 10,000. A score that improves monotonically as
capacity is removed is a score telling you the data supports almost none. A
head adds capacity in exactly the direction the profile says is harmful.

Running it anyway, because a prediction from a profile is not a measurement.

Usage:
    python scripts/eval_head.py --embeddings logs/label_embeddings.npz \\
        --target kcat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor

from petabite.eval_module import labels, stats

FEATURE_SETS = ("length", "composition", "stock", "plasticesm")
CONTROLS = ("length", "composition")

# Deliberately small. With 40 components the honest capacity budget is tiny,
# and a wide head would only make the point about overfitting less
# interesting.
HIDDEN = (64, 16)
PENALTIES = (1.0, 10.0, 100.0)


def fit_head(x_fit: np.ndarray, y_fit: np.ndarray, x_held: np.ndarray,
             hidden: int, penalty: float, seed: int) -> np.ndarray:
    """One head, standardised on the fitting rows only, as the ridge is."""
    scaler = stats.fit_scaler(x_fit)
    model = MLPRegressor(
        hidden_layer_sizes=(hidden,),
        alpha=penalty,
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.2,
        random_state=seed,
    )
    model.fit(stats.apply_scaler(x_fit, scaler), y_fit)
    return model.predict(stats.apply_scaler(x_held, scaler))


def head_out_of_fold(x: np.ndarray, y: np.ndarray, groups: list[str],
                     hidden: int, penalty: float, seed: int) -> np.ndarray:
    """Grouped out-of-fold predictions from the head, fold-centred."""
    predicted = np.zeros(len(y), dtype=float)
    for fit, held in GroupKFold(
            n_splits=min(5, len(set(groups)))).split(x, y, groups):
        fold = fit_head(x[fit], y[fit], x[held], hidden, penalty, seed)
        predicted[held] = stats.centre_fold(fold)
    return predicted


def score(predicted: np.ndarray, y: np.ndarray, groups: list[str]) -> dict:
    """The same reporting the ridge gets, so the two are comparable."""
    value = spearmanr(predicted, y).statistic
    low, high = stats.jackknife_ci(y, predicted, groups)
    return {
        "spearman": round(float(value), 3) if np.isfinite(value) else None,
        "jackknife95": [low, high],
        "permutation_p": stats.permutation_p(y, predicted, groups),
    }


def evaluate(target: str, include_llm: bool, index: dict[str, int],
             cached, seeds: int) -> dict | None:
    """Ridge against heads, on one target."""
    measurements = labels.read_measurements(target, include_llm=include_llm)
    per_sequence = labels.aggregate(
        [m for m in measurements if m.seq_md5 in index])
    components = labels.read_components()

    names = sorted(per_sequence)
    y = np.array([per_sequence[name] for name in names])
    groups = [components[name] for name in names]
    if not labels.is_scoreable(len(names), len(set(groups))):
        print(f"{target}: too few to score")
        return None

    rows = np.array([index[name] for name in names], dtype=int)
    print(f"{target}: {len(names)} sequences, {len(set(groups))} components")
    result: dict[str, dict] = {"n_sequences": len(names),
                               "n_components": len(set(groups)),
                               "ridge": {}, "head": {}}

    print(f"  {'features':12} {'method':22} {'rho':>7} {'p':>7}  jackknife")
    for feature_name in FEATURE_SETS:
        x = np.asarray(cached[feature_name], dtype=np.float64)[rows]

        ridge = score(stats.grouped_out_of_fold(x, y, groups, n_splits=5,
                                                standardise=True),
                      y, groups)
        result["ridge"][feature_name] = ridge
        print(f"  {feature_name:12} {'ridge':22} "
              f"{str(ridge['spearman']):>7} {ridge['permutation_p']:>7}  "
              f"{ridge['jackknife95']}")

        # Several seeds, because a head's initialisation matters and a single
        # draw would be a lottery ticket rather than a result.
        for hidden in HIDDEN:
            for penalty in PENALTIES:
                scores = []
                for seed in range(seeds):
                    predicted = head_out_of_fold(x, y, groups, hidden,
                                                 penalty, seed)
                    scores.append(score(predicted, y, groups))
                values = [s["spearman"] for s in scores
                          if s["spearman"] is not None]
                key = f"{feature_name}_h{hidden}_a{penalty:g}"
                result["head"][key] = {
                    "seeds": len(values),
                    "mean": round(float(np.mean(values)), 3) if values else None,
                    "sd": round(float(np.std(values)), 3) if values else None,
                    "best": round(float(np.max(values)), 3) if values else None,
                    "per_seed": values,
                }
                summary = result["head"][key]
                print(f"  {feature_name:12} "
                      f"{f'head {hidden} units, a {penalty:g}':22} "
                      f"{str(summary['mean']):>7} "
                      f"{'':>7}  sd {summary['sd']}, best {summary['best']}")

    print()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--targets", default="kcat,kcat_over_km")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--include-llm", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cached = np.load(args.embeddings, allow_pickle=False)
    index = {name: position for position, name in enumerate(cached["names"])}
    chosen = [t.strip() for t in args.targets.split(",") if t.strip()]

    print(f"embeddings: {len(index)} sequences, {args.seeds} seeds per head")
    print()

    results = {}
    for target in chosen:
        outcome = evaluate(target, args.include_llm, index, cached, args.seeds)
        if outcome is not None:
            results[target] = outcome

    print("=" * 78)
    print("A head is only interesting where it beats the ridge on the same")
    print("features, and then only where that also beats both controls.")
    print("=" * 78)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
