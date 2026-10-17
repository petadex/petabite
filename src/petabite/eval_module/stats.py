"""Grouped cross-validation and the significance tests built on top of it.

Every number in the Stage 2 evaluation obeys one rule: the unit of
independence is the homology component, never the sequence. Sequences inside a
component share ancestry, so a split, a bootstrap resample or a permutation
that moves individual sequences around will report a correlation that is
mostly leaked ancestry. Each function here therefore takes a ``groups`` list
giving the component of every row, and keeps whole components together.

The functions are arranged in the order a caller needs them: choose a
regularisation strength, produce out-of-fold predictions, put a confidence
interval on the result, test it against a null, and finally ask what effect
size the label set could have detected at all.
"""

from __future__ import annotations

import collections
import math

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

# Ridge penalties tried at every fit. The range has to be this wide for two
# reasons. The feature sets differ by three orders of magnitude in width, from
# a single length column to a 960-dimensional embedding. And with 73 sequences
# against 960 columns the useful regime is heavy shrinkage: on the p-np-c4
# stratum the standardised score was still climbing at the old grid's top
# value of 10,000, so the grid was truncated exactly where the answer was.
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0, 1_000_000.0)


def group_rows(groups: list[str]) -> dict[str, list[int]]:
    """Row indices belonging to each component."""
    by_group: dict[str, list[int]] = collections.defaultdict(list)
    for index, group in enumerate(groups):
        by_group[group].append(index)
    return dict(by_group)


def fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column means and standard deviations, for z-scoring features.

    Needed because a ridge penalty is not scale invariant: the same alpha
    applied to features with standard deviation 0.008 and 26.8, which is the
    range across ESM-C's layers, is two completely different amounts of
    regularisation. Any comparison between feature sets of different scale has
    to standardise first or it is comparing penalties rather than
    representations.
    """
    centre = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return centre, scale


def apply_scaler(x: np.ndarray, scaler: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    centre, scale = scaler
    return (x - centre) / scale


def fit_predict(x_fit, y_fit, x_held, alpha: float,
                 standardise: bool) -> np.ndarray:
    """One ridge fit, with the scaler estimated on the fitting rows only.

    Estimating it on all rows would leak the held-out distribution into the
    fit, which is a small leak but an unnecessary one.
    """
    if standardise:
        scaler = fit_scaler(x_fit)
        x_fit, x_held = apply_scaler(x_fit, scaler), apply_scaler(x_held, scaler)
    return Ridge(alpha=alpha).fit(x_fit, y_fit).predict(x_held)


def pick_alpha(x: np.ndarray, y: np.ndarray, groups: list[str],
               standardise: bool = False) -> float:
    """Choose a ridge penalty by grouped CV inside the fitting data only.

    Called from within an outer fold, so the held-out components of that outer
    fold are not visible here and cannot influence the choice.

    Selection is by the one-standard-error rule rather than by the best mean:
    among the penalties scoring within one standard error of the best, take the
    strongest. Plain argmax proved too unstable here. An inner split of a
    73-sequence stratum leaves about 17 training components, and the penalty it
    landed on moved the headline correlation on `kcat_over_km` between 0.30 and
    0.40. Breaking ties toward more shrinkage is the conventional remedy and
    errs in the conservative direction, since it can only lower the score.
    """
    unique = len(set(groups))
    if unique < 3:
        return 1.0

    splitter = GroupKFold(n_splits=min(5, unique))
    folds = list(splitter.split(x, y, groups))

    # Each penalty is scored once per fold rather than on pooled predictions,
    # because the rule needs the spread across folds to form its error bar.
    per_alpha: dict[float, list[float]] = {}
    for alpha in ALPHAS:
        scores = []
        for fit, held in folds:
            if len(set(y[held])) < 3:
                continue
            predicted = fit_predict(x[fit], y[fit], x[held], alpha, standardise)
            score = spearmanr(predicted, y[held]).statistic
            if np.isfinite(score):
                scores.append(float(score))
        if scores:
            per_alpha[alpha] = scores

    if not per_alpha:
        return 1.0

    means = {alpha: float(np.mean(scores)) for alpha, scores in per_alpha.items()}
    best = max(means, key=means.get)
    spread = per_alpha[best]
    standard_error = (float(np.std(spread, ddof=1)) / np.sqrt(len(spread))
                      if len(spread) > 1 else 0.0)

    threshold = means[best] - standard_error
    return max(alpha for alpha, mean in means.items() if mean >= threshold)


def grouped_out_of_fold(x: np.ndarray, y: np.ndarray, groups: list[str],
                        n_splits: int | None,
                        standardise: bool = False,
                        centre_folds: bool = True) -> np.ndarray:
    """Out-of-fold predictions with whole components held out together.

    ``n_splits=None`` means leave one component out, the limiting case: it
    uses the most training data per fold, at the cost of one model fit per
    component.

    ``centre_folds`` z-scores each fold's predictions before they are
    concatenated, and defaults to on because without it the pooled vector is
    not a valid thing to correlate. Under a strong penalty every fold's
    predictions shrink toward that fold's own training mean, so the
    concatenated vector ranks sequences largely by which fold they landed in.
    Components here differ enough in mean that the effect is severe: on the
    p-np-c4 stratum the pooled score for sequence length ran from +0.04 to
    -0.29 across the penalty grid, while the fold-aware score held at +0.40
    and matched a direct within-fold calculation of +0.4157.

    Z-scoring keeps the ordering each model produced and discards only the
    fold-specific level and scale, so one correlation over every sequence is
    still available and the bootstrap and permutation tests built on it still
    apply. Leave-one-component-out cannot be corrected this way, because a
    fold holding one sequence has no within-fold ordering at all, which is
    reason enough to prefer 5-fold for any reported number.
    """
    unique = sorted(set(groups))
    predicted = np.zeros_like(y)

    if n_splits is None:
        # No fold centring here: a single-sequence fold has nothing to centre.
        for held_out in unique:
            mask = np.array([g == held_out for g in groups])
            if mask.all():
                continue
            inner = [g for g, keep in zip(groups, ~mask) if keep]
            alpha = pick_alpha(x[~mask], y[~mask], inner, standardise)
            predicted[mask] = fit_predict(x[~mask], y[~mask], x[mask], alpha,
                                          standardise)
        return predicted

    splitter = GroupKFold(n_splits=min(n_splits, len(unique)))
    for fit, held in splitter.split(x, y, groups):
        inner = [groups[i] for i in fit]
        alpha = pick_alpha(x[fit], y[fit], inner, standardise)
        fold = fit_predict(x[fit], y[fit], x[held], alpha, standardise)
        predicted[held] = centre_fold(fold) if centre_folds else fold
    return predicted


def centre_fold(fold: np.ndarray) -> np.ndarray:
    """Z-score one fold's predictions, or flatten them if they are constant."""
    spread = float(fold.std())
    if spread < 1e-12:
        return np.zeros_like(fold)
    return (fold - fold.mean()) / spread


def fold_scores(x: np.ndarray, y: np.ndarray, groups: list[str],
                n_splits: int = 5, standardise: bool = False) -> list[float]:
    """One Spearman correlation per fold, unpooled.

    The most conservative reading of a grouped CV: every comparison is between
    two sequences scored by the same model. It is noisy, since a fold here
    holds twelve to twenty-four sequences, which is why the fold-centred
    pooled score is the one reported and this is the check on it.
    """
    splitter = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    scores = []

    for fit, held in splitter.split(x, y, groups):
        predicted = fit_predict(x[fit], y[fit], x[held],
                                pick_alpha(x[fit], y[fit],
                                           [groups[i] for i in fit],
                                           standardise),
                                standardise)
        if len(set(predicted)) < 3 or len(set(y[held])) < 3:
            continue
        score = spearmanr(predicted, y[held]).statistic
        if np.isfinite(score):
            scores.append(float(score))
    return scores


def bootstrap_ci(y, predicted, groups, n: int = 2000,
                 seed: int = 0) -> tuple[float | None, float | None]:
    """A 95% interval from resampling components with replacement.

    Resampling sequences instead would narrow the interval by pretending that
    members of the same family are independent observations.
    """
    rng = np.random.default_rng(seed)
    by_group = group_rows(groups)
    keys = list(by_group)
    y, predicted = np.asarray(y), np.asarray(predicted)

    scores = []
    for _ in range(n):
        drawn = rng.choice(len(keys), size=len(keys), replace=True)
        rows = [i for d in drawn for i in by_group[keys[d]]]
        # A resample can land on so few distinct values that a rank
        # correlation is undefined; skip it rather than record a nan.
        if len(set(y[rows])) < 3:
            continue
        score = spearmanr(predicted[rows], y[rows]).statistic
        if np.isfinite(score):
            scores.append(score)

    if len(scores) < 50:
        return None, None
    return (round(float(np.percentile(scores, 2.5)), 3),
            round(float(np.percentile(scores, 97.5)), 3))


def permutation_p(y, predicted, groups, n: int = 1000, seed: int = 0) -> float:
    """Shuffle the target between whole components and see how often chance wins.

    Moving whole components rather than sequences keeps the group structure
    intact, so the null hypothesis being tested is "this feature set carries
    no information about the target" rather than the much weaker "these
    components are unrelated to each other".
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    observed = spearmanr(predicted, y).statistic
    if not np.isfinite(observed):
        return float("nan")

    by_group = group_rows(groups)
    keys = list(by_group)
    source_rows = [i for k in keys for i in by_group[k]]

    wins = 0
    for _ in range(n):
        order = rng.permutation(len(keys))
        target_rows = [i for o in order for i in by_group[keys[o]]]
        shuffled = np.empty_like(y)
        # Components differ in size, so the donor and recipient runs rarely
        # match; truncating to the shorter of the two keeps every position
        # that is written valid.
        take = min(len(source_rows), len(target_rows))
        shuffled[np.array(target_rows[:take])] = y[source_rows[:take]]
        score = spearmanr(predicted, shuffled).statistic
        if np.isfinite(score) and abs(score) >= abs(observed):
            wins += 1

    return round((wins + 1) / (n + 1), 4)


def synthetic_prediction(y: np.ndarray, rho: float, rng) -> np.ndarray:
    """A fake prediction correlating with ``y`` at approximately ``rho``.

    Built on the ranks of ``y`` rather than its values, because the quantity
    being simulated is a Spearman correlation.
    """
    ranks = np.argsort(np.argsort(y)).astype(float)
    ranks = (ranks - ranks.mean()) / (ranks.std() + 1e-12)
    noise = rng.standard_normal(len(y))
    noise = (noise - noise.mean()) / (noise.std() + 1e-12)
    return rho * ranks + math.sqrt(max(0.0, 1 - rho ** 2)) * noise


def detects_signal(y, predicted, groups, rng, n: int = 400) -> bool:
    """Would the component bootstrap of this prediction exclude zero?"""
    by_group = group_rows(groups)
    keys = list(by_group)
    y, predicted = np.asarray(y), np.asarray(predicted)

    scores = []
    for _ in range(n):
        drawn = rng.integers(0, len(keys), size=len(keys))
        rows = [i for d in drawn for i in by_group[keys[d]]]
        if len(set(y[rows])) < 3:
            continue
        score = spearmanr(predicted[rows], y[rows]).statistic
        if np.isfinite(score):
            scores.append(score)

    if len(scores) < 50:
        return False
    low, high = np.percentile(scores, [2.5, 97.5])
    return bool(low > 0 or high < 0)


def power_curve(y: np.ndarray, groups: list[str],
                rhos: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
                trials: int = 200, seed: int = 0) -> dict[float, float]:
    """Fraction of synthetic signals at each strength that would be detected.

    This is a property of the label set, not of any model: it answers "what
    could we have seen", which has to be settled before "what did we see"
    means anything. A simulation rather than a closed form, because the
    components are wildly unequal in size and the textbook formulas assume
    they are not.
    """
    rng = np.random.default_rng(seed)
    power = {}
    for rho in rhos:
        detected = sum(
            detects_signal(y, synthetic_prediction(y, rho, rng), groups, rng)
            for _ in range(trials)
        )
        power[rho] = round(detected / trials, 3)
    return power


def smallest_detectable(power: dict[float, float], threshold: float = 0.8):
    """The weakest simulated correlation that clears the power threshold."""
    adequate = [rho for rho, value in power.items() if value >= threshold]
    return min(adequate) if adequate else None
