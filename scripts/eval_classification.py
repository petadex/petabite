"""Search item 4: ask which sequences are active, not how active they are.

The regression framing has one weakness that no amount of careful cross
validation fixes. A `kcat` column pooled over forty substrates is not one
quantity, so its numeric value carries the protocol as much as the enzyme, and
the measurement-level attempt to give the model the protocol as an input made
things worse rather than better.

Classification sidesteps that. Whether an enzyme is among the better half of
those assayed on the same substrate is comparable across substrates in a way
that the raw log value is not, so the heterogeneity that defeats the
regression becomes something the label definition absorbs.

The threshold is therefore taken within substrate, not globally. A global
median split would re-import exactly the confound this framing exists to
avoid: substrates differ in their typical values, so a global split would
partly label the substrate rather than the enzyme.

The metric is the area under the ROC curve, which is rank based like the
Spearman correlation used everywhere else, so the two are comparable in
spirit. It is reported on fold-centred pooled predictions for the same reason
the regression is, with the mean per-fold value beside it as the conservative
check. The null is 0.5 rather than zero.

Usage:
    python scripts/eval_classification.py \\
        --embeddings logs/label_embeddings.npz --target kcat
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from petabite.eval_module import labels, stats

FEATURE_SETS = ("length", "composition", "stock", "plasticesm")
CONTROLS = ("length", "composition")

# The inverse-penalty grid for logistic regression, mirroring the ridge alpha
# grid in `stats.ALPHAS` so the two framings are regularised over comparable
# ranges rather than one being handicapped.
STRENGTHS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)

# A stratum needs both classes present and enough components to hold out.
MIN_PER_CLASS = 8
MIN_COMPONENTS = 8


def label_within_substrate(measurements: list[labels.Measurement]
                           ) -> dict[str, int]:
    """Active or not, judged against others assayed on the same substrate.

    One label per sequence. A sequence measured several times on a substrate
    is summarised by its median there first, and a sequence appearing under
    several substrates counts as active if it is in the better half under any
    of them, because the alternative is to average ranks across scales that
    are not comparable.
    """
    by_substrate: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for measurement in measurements:
        by_substrate[measurement.substrate][measurement.seq_md5].append(
            measurement.value)

    active: dict[str, int] = {}
    for per_sequence in by_substrate.values():
        medians = {name: float(np.median(values))
                   for name, values in per_sequence.items()}
        if len(medians) < 2:
            continue
        threshold = float(np.median(list(medians.values())))
        for name, value in medians.items():
            # Ties go to the inactive side, so the threshold cannot create
            # activity out of a substrate whose values are all identical.
            is_active = int(value > threshold)
            active[name] = max(active.get(name, 0), is_active)
    return active


def pick_strength(x: np.ndarray, y: np.ndarray, groups: list[str]) -> float:
    """Choose the logistic penalty by grouped CV inside the fitting data.

    Same one-standard-error rule as `stats.pick_alpha`: among the strengths
    scoring within one standard error of the best, take the most regularised.
    Plain argmax was unstable enough in the regression to swing a headline by
    0.10, and there is no reason to expect better behaviour here.
    """
    unique = len(set(groups))
    if unique < 3:
        return 1.0

    folds = list(GroupKFold(n_splits=min(5, unique)).split(x, y, groups))
    per_strength: dict[float, list[float]] = {}
    for strength in STRENGTHS:
        scores = []
        for fit, held in folds:
            if len(set(y[fit])) < 2 or len(set(y[held])) < 2:
                continue
            predicted = fit_predict_proba(x[fit], y[fit], x[held], strength)
            scores.append(float(roc_auc_score(y[held], predicted)))
        if scores:
            per_strength[strength] = scores

    if not per_strength:
        return 1.0

    means = {s: float(np.mean(v)) for s, v in per_strength.items()}
    best = max(means, key=means.get)
    spread = per_strength[best]
    standard_error = (float(np.std(spread, ddof=1)) / np.sqrt(len(spread))
                      if len(spread) > 1 else 0.0)
    threshold = means[best] - standard_error
    return min(s for s, mean in means.items() if mean >= threshold)


def fit_predict_proba(x_fit: np.ndarray, y_fit: np.ndarray, x_held: np.ndarray,
                      strength: float) -> np.ndarray:
    """One logistic fit, standardised on the fitting rows only."""
    scaler = stats.fit_scaler(x_fit)
    model = LogisticRegression(C=strength, max_iter=2000)
    model.fit(stats.apply_scaler(x_fit, scaler), y_fit)
    return model.predict_proba(stats.apply_scaler(x_held, scaler))[:, 1]


def out_of_fold(x: np.ndarray, y: np.ndarray, groups: list[str]
                ) -> tuple[np.ndarray, list[float]]:
    """Fold-centred pooled probabilities, and the per-fold areas.

    Centring is needed for the same reason it is in the regression: each
    fold's probabilities sit at that fold's own base rate, so pooling them raw
    would rank sequences partly by which fold they landed in.
    """
    predicted = np.zeros(len(y), dtype=float)
    areas = []

    for fit, held in GroupKFold(n_splits=min(5, len(set(groups)))).split(
            x, y, groups):
        if len(set(y[fit])) < 2:
            continue
        inner = [groups[i] for i in fit]
        fold = fit_predict_proba(x[fit], y[fit], x[held],
                                 pick_strength(x[fit], y[fit], inner))
        predicted[held] = stats.centre_fold(fold)
        if len(set(y[held])) > 1:
            areas.append(float(roc_auc_score(y[held], fold)))

    return predicted, areas


def permutation_p_auc(y: np.ndarray, predicted: np.ndarray, groups: list[str],
                      n: int = 1000, seed: int = 0) -> float:
    """Shuffle whole components, never individual sequences."""
    rng = np.random.default_rng(seed)
    by_group = stats.group_rows(groups)
    keys = list(by_group)
    observed = roc_auc_score(y, predicted)

    beaten = 0
    for _ in range(n):
        order = rng.permutation(len(keys))
        shuffled = np.empty_like(y)
        source_rows = [row for key in keys for row in by_group[key]]
        target_rows = [row for index in order for row in by_group[keys[index]]]
        shuffled[np.array(target_rows)] = y[np.array(source_rows)]
        if len(set(shuffled)) < 2:
            continue
        if roc_auc_score(shuffled, predicted) >= observed:
            beaten += 1
    return round((beaten + 1) / (n + 1), 4)


def jackknife_auc(y: np.ndarray, predicted: np.ndarray, groups: list[str]
                  ) -> tuple[float | None, float | None]:
    """Delete one component at a time, on the log-odds of the area.

    The same reasoning as `stats.jackknife_ci`: the bootstrap duplicates
    components and reports intervals that are too narrow, and a plain interval
    on a quantity bounded by 0 and 1 can leave its own range. The log odds of
    the area is the usual unbounded scale for it.
    """
    by_group = stats.group_rows(groups)
    keys = list(by_group)
    if len(keys) < 3:
        return None, None

    def to_log_odds(value: float) -> float:
        clipped = min(max(value, 1e-6), 1 - 1e-6)
        return float(np.log(clipped / (1 - clipped)))

    deleted = []
    for held_out in keys:
        rows = [row for key, members in by_group.items() if key != held_out
                for row in members]
        if len(set(y[rows])) < 2:
            continue
        deleted.append(to_log_odds(float(roc_auc_score(y[rows],
                                                       predicted[rows]))))

    if len(deleted) < 3:
        return None, None

    count = len(deleted)
    spread = np.array(deleted) - np.mean(deleted)
    standard_error = np.sqrt((count - 1) / count * float((spread ** 2).sum()))
    half = t.ppf(0.975, count - 1) * standard_error
    centre = to_log_odds(float(roc_auc_score(y, predicted)))

    def invert(value: float) -> float:
        return round(float(1 / (1 + np.exp(-value))), 3)

    return invert(centre - half), invert(centre + half)


def score_one(x: np.ndarray, y: np.ndarray, groups: list[str]) -> dict:
    """Everything reported for one feature set."""
    predicted, areas = out_of_fold(x, y, groups)
    area = float(roc_auc_score(y, predicted))
    low, high = jackknife_auc(y, predicted, groups)

    return {
        "dim": int(x.shape[1]),
        "auc": round(area, 3),
        "jackknife95": [low, high],
        "permutation_p": permutation_p_auc(y, predicted, groups),
        "per_fold_mean": round(float(np.mean(areas)), 3) if areas else None,
        "n_folds": len(areas),
    }


def verdict(scores: dict[str, float | None]) -> str:
    """Whether a backbone beat both controls, measured from the 0.5 null."""
    control = max((value for name, value in scores.items()
                   if name in CONTROLS and value is not None), default=None)
    model = max((value for name, value in scores.items()
                 if name not in CONTROLS and value is not None), default=None)
    if control is None or model is None:
        return "unscoreable"
    if model > control:
        return f"model wins by {model - control:+.3f}"
    return f"controls win by {control - model:+.3f}"


def evaluate(target: str, include_llm: bool, index: dict[str, int],
             cached) -> dict | None:
    """One target, from measurements to a verdict."""
    measurements = labels.read_measurements(target, include_llm=include_llm)
    measurements = [m for m in measurements if m.seq_md5 in index]
    active = label_within_substrate(measurements)
    components = labels.read_components()

    names = sorted(name for name in active if name in index)
    y = np.array([active[name] for name in names])
    groups = [components[name] for name in names]
    positives = int(y.sum())
    n_components = len(set(groups))

    header = (f"{target}: {len(names)} sequences, {positives} active, "
              f"{len(names) - positives} inactive, {n_components} components")
    if (min(positives, len(names) - positives) < MIN_PER_CLASS
            or n_components < MIN_COMPONENTS):
        print(f"{header}  -> SKIPPED, too few to score")
        return None
    print(header)

    substrates = len({m.substrate for m in measurements})
    print(f"  threshold taken within each of {substrates} substrates")
    print(f"  {'features':12} {'dim':>4} {'auc':>7} {'per-fold':>9} "
          f"{'p':>7}  jackknife 95%")

    rows = np.array([index[name] for name in names], dtype=int)
    result = {"n_sequences": len(names), "n_active": positives,
              "n_components": n_components, "n_substrates": substrates,
              "features": {}}
    scores: dict[str, float | None] = {}

    for feature_name in FEATURE_SETS:
        x = np.asarray(cached[feature_name], dtype=np.float64)[rows]
        entry = score_one(x, y, groups)
        result["features"][feature_name] = entry
        scores[feature_name] = entry["auc"]
        print(f"  {feature_name:12} {entry['dim']:>4} {entry['auc']:>7} "
              f"{str(entry['per_fold_mean']):>9} "
              f"{entry['permutation_p']:>7}  {entry['jackknife95']}")

    result["verdict"] = verdict(scores)
    print(f"  {result['verdict']}")
    print()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--targets", default="")
    parser.add_argument("--include-llm", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cached = np.load(args.embeddings, allow_pickle=False)
    index = {name: position for position, name in enumerate(cached["names"])}
    chosen = ([t.strip() for t in args.targets.split(",") if t.strip()]
              or list(labels.TARGETS))

    print(f"embeddings: {len(index)} sequences")
    print("the null is an area of 0.5, not a correlation of zero")
    print()

    results = {}
    for target in chosen:
        outcome = evaluate(target, args.include_llm, index, cached)
        if outcome is not None:
            results[target] = outcome

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
