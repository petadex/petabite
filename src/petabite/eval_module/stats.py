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

# Ridge penalties tried at every fit. The range is wide because the feature
# sets differ by three orders of magnitude in width, from a single length
# column to a 960-dimensional embedding.
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def group_rows(groups: list[str]) -> dict[str, list[int]]:
    """Row indices belonging to each component."""
    by_group: dict[str, list[int]] = collections.defaultdict(list)
    for index, group in enumerate(groups):
        by_group[group].append(index)
    return dict(by_group)


def pick_alpha(x: np.ndarray, y: np.ndarray, groups: list[str]) -> float:
    """Choose a ridge penalty by grouped CV inside the fitting data only.

    Called from within an outer fold, so the held-out components of that
    outer fold are not visible here and cannot influence the choice.
    """
    unique = len(set(groups))
    if unique < 3:
        return 1.0

    splitter = GroupKFold(n_splits=min(5, unique))
    best, best_score = 1.0, -np.inf
    for alpha in ALPHAS:
        predicted = np.zeros_like(y)
        for fit, held in splitter.split(x, y, groups):
            model = Ridge(alpha=alpha).fit(x[fit], y[fit])
            predicted[held] = model.predict(x[held])
        score = spearmanr(predicted, y).statistic
        if np.isfinite(score) and score > best_score:
            best, best_score = alpha, score
    return best


def grouped_out_of_fold(x: np.ndarray, y: np.ndarray, groups: list[str],
                        n_splits: int | None) -> np.ndarray:
    """Out-of-fold predictions with whole components held out together.

    ``n_splits=None`` means leave one component out, the limiting case: it
    uses the most training data per fold, at the cost of one model fit per
    component.
    """
    unique = sorted(set(groups))
    predicted = np.zeros_like(y)

    if n_splits is None:
        for held_out in unique:
            mask = np.array([g == held_out for g in groups])
            if mask.all():
                continue
            inner = [g for g, keep in zip(groups, ~mask) if keep]
            alpha = pick_alpha(x[~mask], y[~mask], inner)
            model = Ridge(alpha=alpha).fit(x[~mask], y[~mask])
            predicted[mask] = model.predict(x[mask])
        return predicted

    splitter = GroupKFold(n_splits=min(n_splits, len(unique)))
    for fit, held in splitter.split(x, y, groups):
        inner = [groups[i] for i in fit]
        alpha = pick_alpha(x[fit], y[fit], inner)
        model = Ridge(alpha=alpha).fit(x[fit], y[fit])
        predicted[held] = model.predict(x[held])
    return predicted


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
