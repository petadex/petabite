"""Turning assay conditions into features, and regressing them back out.

A measurement in this release is a number attached to a protocol: a substrate,
a pH, a temperature and sometimes an incubation time. Two enzymes assayed on
different substrates are not comparable, and the pooled `kcat` column mixes
forty of them, from p-nitrophenyl esters to acetylcholine. That raises the
possibility that the targets are largely protocol variance with the sequence
effect buried underneath.

Two tools are needed to test that. The first is a design matrix over the
conditions, so a model can be asked how much it predicts from the protocol
alone. The second is residualisation: fit the condition model, subtract what
it predicts, and ask whether a sequence signal appears in what is left.

Both come in two flavours. The out-of-fold fit, grouped by homology
component, is the honest way to score how much the protocol explains, and is
what test 2 of the evaluation reports. The in-sample fit is what the
residualisation uses, for a reason spelled out on `fit_conditions`: out of
fold the condition model is anti-predictive, because substrate and family are
confounded, so subtracting it would add noise rather than remove it.
"""

from __future__ import annotations

import collections

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from petabite.eval_module import stats
from petabite.eval_module.labels import Measurement

# Conditions are missing for a sizeable minority of rows, between 8% of pH
# values and two thirds of incubation times. Imputing to the median and
# flagging the imputation lets a model learn "unreported" as its own state
# rather than silently treating it as typical.
MISSING_FLAG_SUFFIX = "_missing"


def _impute(values: list[float | None]) -> tuple[np.ndarray, np.ndarray]:
    """Fill blanks with the median, and return a flag marking where they were."""
    present = [v for v in values if v is not None]
    fallback = float(np.median(present)) if present else 0.0

    filled = np.array([fallback if v is None else v for v in values], dtype=np.float64)
    missing = np.array([1.0 if v is None else 0.0 for v in values], dtype=np.float64)
    return filled, missing


def design_matrix(measurements: list[Measurement],
                  max_substrates: int = 12) -> tuple[np.ndarray, list[str]]:
    """Condition features for each measurement, with their column names.

    Substrates are one-hot encoded, keeping the most frequent and folding the
    long tail into a single `other` column: forty substrates over a few
    hundred rows would otherwise give the model a near-unique indicator per
    measurement.
    """
    counts = collections.Counter(m.substrate for m in measurements)
    kept = [name for name, _ in counts.most_common(max_substrates)]
    index = {name: position for position, name in enumerate(kept)}

    one_hot = np.zeros((len(measurements), len(kept) + 1), dtype=np.float64)
    for row, m in enumerate(measurements):
        one_hot[row, index.get(m.substrate, len(kept))] = 1.0
    names = [f"substrate={name or '(blank)'}" for name in kept] + ["substrate=other"]

    ph, ph_missing = _impute([m.ph for m in measurements])
    temperature, temperature_missing = _impute([m.temperature_c for m in measurements])

    # Incubation times run from minutes to weeks, so the log is the
    # comparable scale; the +1 keeps zero-hour readings finite.
    raw_time = [m.time_h for m in measurements]
    time, time_missing = _impute(raw_time)
    time = np.log10(np.clip(time, 0.0, None) + 1.0)

    numeric = np.column_stack([
        ph, ph_missing,
        temperature, temperature_missing,
        time, time_missing,
    ])
    names += [
        "ph", "ph" + MISSING_FLAG_SUFFIX,
        "temperature_c", "temperature_c" + MISSING_FLAG_SUFFIX,
        "log_time_h", "time_h" + MISSING_FLAG_SUFFIX,
    ]

    return np.column_stack([one_hot, numeric]), names


def standardise(x: np.ndarray) -> np.ndarray:
    """Centre and scale each column, so one ridge penalty suits them all."""
    centred = x - x.mean(axis=0, keepdims=True)
    scale = centred.std(axis=0, keepdims=True)
    scale[scale < 1e-9] = 1.0
    return centred / scale


def predict_from_conditions(measurements: list[Measurement],
                            components: dict[str, str],
                            alpha: float = 1.0,
                            n_splits: int = 5) -> np.ndarray:
    """Out-of-fold prediction of each measurement from its protocol alone.

    Folds hold out whole homology components, exactly as the model evaluation
    does, so the number this feeds is comparable with the backbone scores.
    """
    y = np.array([m.value for m in measurements])
    groups = [components.get(m.seq_md5, m.seq_md5) for m in measurements]
    x = standardise(design_matrix(measurements)[0])

    unique = len(set(groups))
    if unique < 2:
        return np.full_like(y, y.mean())

    predicted = np.zeros_like(y)
    splitter = GroupKFold(n_splits=min(n_splits, unique))
    for fit, held in splitter.split(x, y, groups):
        fold = Ridge(alpha=alpha).fit(x[fit], y[fit]).predict(x[held])
        # Centred within the fold for the same reason the model scoring is:
        # pooling raw predictions makes the correlation partly a ranking of
        # fold means rather than of measurements. See
        # `stats.grouped_out_of_fold`.
        predicted[held] = stats.centre_fold(fold)
    return predicted


def fit_conditions(measurements: list[Measurement],
                   alpha: float = 1.0) -> np.ndarray:
    """In-sample prediction of each measurement from its protocol.

    Deliberately in-sample, which needs justifying because nothing else in
    this evaluation is. Subtracting an out-of-fold condition prediction turned
    out to increase the spread of the target rather than reduce it, taking
    `kcat` from a standard deviation of 1.478 to 1.707, so it was injecting
    noise rather than removing protocol variance.

    The explanation first offered for that was a sign flip from substrate and
    homology family being confounded, which is plausible but was measured with
    the pooled scoring since found to be invalid. The in-sample choice does not
    depend on that explanation: it removes about 42% of `kcat`'s variance,
    which is what it is for.

    An in-sample fit instead errs in the safe direction. It can only remove
    variance, and some of what it removes will be genuine sequence signal that
    happens to covary with the protocol. So the residual target understates
    the sequence effect, and anything that still predicts it is real. A null
    result here is weaker evidence than a positive one.
    """
    y = np.array([m.value for m in measurements])
    x = standardise(design_matrix(measurements)[0])
    return Ridge(alpha=alpha).fit(x, y).predict(x)


def residualise(measurements: list[Measurement]) -> np.ndarray:
    """What is left of each measurement once its protocol is accounted for."""
    y = np.array([m.value for m in measurements])
    return y - fit_conditions(measurements)
