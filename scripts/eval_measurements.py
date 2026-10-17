"""Search item 3: model each measurement, with its protocol as an input.

Every evaluation so far collapsed a sequence's measurements to one median and
threw the protocol away. That discards two different things. It discards rows,
because a sequence assayed eight times becomes one point. And it discards what
was actually done: which substrate, at which pH and temperature, for how long.
The pooled `kcat` column mixes forty substrates, so the median of a sequence's
measurements is an average over protocols that are not comparable.

This script models the measurement instead. One row per measurement, features
being the sequence's features joined to its own protocol's features, and the
target being that measurement's value rather than a median.

The question is not whether that join predicts anything. The protocol alone
explains a good deal of the variance in sample, so a high number here would be
easy to obtain and would mean nothing. The question is whether adding the
sequence to the protocol beats the protocol alone, and whether an embedding
beats length and composition at that same job. So the protocol appears in the
baseline and in every model row, and the comparison is always like for like.

The unit of independence is still the homology component. Every measurement of
every sequence in a component is held out together, which also keeps repeated
measurements of one sequence from straddling the hold-out boundary, since a
sequence measured on both sides of it would leak its own value across.

Two readouts are reported. The measurement-level correlation is what the model
was fitted to predict. The sequence-level correlation averages each sequence's
out-of-fold predictions and correlates those against the per-sequence median,
which is the same quantity the main sweep reports and is therefore the number
to compare against it.

Usage:
    python scripts/eval_measurements.py --embeddings logs/label_embeddings.npz \\
        --target kcat --out logs/measurements_kcat.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from petabite.eval_module import conditions, labels, stats

# The sequence feature blocks, in the order they are reported. `length` and
# `composition` are the controls a backbone has to beat; `none` is the
# protocol-only baseline that every sequence block has to beat in turn.
SEQUENCE_BLOCKS = ("none", "length", "composition", "stock", "plasticesm")
CONTROLS = ("length", "composition")
MODELS = ("stock", "plasticesm")

# A target needs enough measurements to be worth modelling at this level, on
# top of the sequence and component minimums the aggregated sweep applies.
MIN_MEASUREMENTS = 60


def read_rows(target: str, include_llm: bool,
              index: dict[str, int]) -> list[labels.Measurement]:
    """Measurements for one target whose sequence has a cached embedding."""
    measurements = labels.read_measurements(target, include_llm=include_llm)
    return [m for m in measurements if m.seq_md5 in index]


def build_blocks(measurements: list[labels.Measurement],
                 index: dict[str, int],
                 cached) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Expand the per-sequence feature blocks to one row per measurement.

    A sequence measured eight times contributes its embedding eight times.
    That is the point: the eight rows differ in their protocol columns and in
    their target, so the fit can attribute the difference to the protocol
    rather than having to average it away.
    """
    rows = np.array([index[m.seq_md5] for m in measurements], dtype=int)

    blocks: dict[str, np.ndarray] = {}
    for name in SEQUENCE_BLOCKS:
        if name == "none":
            continue
        blocks[name] = np.asarray(cached[name], dtype=np.float64)[rows]

    protocol, protocol_names = conditions.design_matrix(measurements)
    return blocks, protocol, protocol_names


def join(sequence_block: np.ndarray | None,
         protocol: np.ndarray | None) -> np.ndarray:
    """Put a sequence block and a protocol block side by side."""
    parts = [part for part in (sequence_block, protocol) if part is not None]
    return np.column_stack(parts)


def collapse_to_sequences(measurements: list[labels.Measurement],
                          predicted: np.ndarray,
                          components: dict[str, str]):
    """One point per sequence: mean prediction against median target.

    This is the headline readout, and the reason is the replicate imbalance.
    A correlation over measurement rows weights each sequence by how many
    times it was assayed, and one `pet_release_conc` sequence carries 227 of
    that target's 1448 measurements. A measurement-level score is therefore
    largely a statement about how well one enzyme's replicates are ordered,
    which is the same class of mistake as the fold-mean artefact: a number
    that looks like a correlation over the data set but is dominated by one
    part of it.

    Collapsing first removes that weighting, and gives a number directly
    comparable with the aggregated sweep, which also scores one median per
    sequence. Averaging within a sequence is safe because all of a sequence's
    measurements sit in the same fold, so the same fold statistics centred
    them all.
    """
    by_sequence: dict[str, list[float]] = collections.defaultdict(list)
    truth: dict[str, list[float]] = collections.defaultdict(list)
    for measurement, value in zip(measurements, predicted):
        by_sequence[measurement.seq_md5].append(float(value))
        truth[measurement.seq_md5].append(measurement.value)

    names = sorted(by_sequence)
    averaged = np.array([float(np.mean(by_sequence[name])) for name in names])
    medians = np.array([float(np.median(truth[name])) for name in names])
    groups = [components[name] for name in names]
    return averaged, medians, groups


def fixed_alpha_prediction(x: np.ndarray, y: np.ndarray, groups: list[str],
                           alpha: float, standardise: bool) -> np.ndarray:
    """Out-of-fold predictions at one stated penalty, fold-centred."""
    predicted = np.zeros_like(y)
    splitter = GroupKFold(n_splits=min(5, len(set(groups))))
    for fit, held in splitter.split(x, y, groups):
        fold = stats.fit_predict(x[fit], y[fit], x[held], alpha, standardise)
        predicted[held] = stats.centre_fold(fold)
    return predicted


def alpha_profile(x: np.ndarray, y: np.ndarray, groups: list[str],
                  measurements: list[labels.Measurement],
                  components: dict[str, str],
                  standardise: bool) -> dict[str, float | None]:
    """The score at every penalty, rather than at one selected penalty.

    Reported because a single selected alpha hid an instability once already:
    on one stratum the choice moved the headline correlation between 0.30 and
    0.40, and the profile is what makes that visible instead of invisible.

    Scored the same way the headline is, one point per sequence, so a row of
    the profile and the reported number are the same quantity at different
    penalties rather than two different quantities.
    """
    profile: dict[str, float | None] = {}
    for alpha in stats.ALPHAS:
        predicted = fixed_alpha_prediction(x, y, groups, alpha, standardise)
        averaged, medians, _ = collapse_to_sequences(measurements, predicted,
                                                     components)
        score = spearmanr(averaged, medians).statistic
        profile[f"{alpha:g}"] = (round(float(score), 3)
                                 if np.isfinite(score) else None)
    return profile


def score_set(x: np.ndarray, y: np.ndarray, groups: list[str],
              measurements: list[labels.Measurement],
              components: dict[str, str], standardise: bool) -> dict:
    """Grouped 5-fold, fold-centred, scored once per sequence.

    The interval and the null are computed on the sequence-level vectors, not
    the measurement-level ones, so they describe the number being reported.
    Components remain the resampling and permutation unit throughout.
    """
    predicted = stats.grouped_out_of_fold(x, y, groups, n_splits=5,
                                          standardise=standardise)
    averaged, medians, sequence_groups = collapse_to_sequences(
        measurements, predicted, components)

    score = spearmanr(averaged, medians).statistic
    low, high = stats.bootstrap_ci(medians, averaged, sequence_groups)
    permutation = stats.permutation_p(medians, averaged, sequence_groups)

    # The measurement-level correlation is kept as a diagnostic, because a
    # large gap between it and the sequence-level number says the fit is
    # tracking replicate structure rather than sequence identity.
    measurement_score = spearmanr(predicted, y).statistic
    folds = stats.fold_scores(x, y, groups, 5, standardise)

    return {
        "dim": int(x.shape[1]),
        "spearman": round(float(score), 3) if np.isfinite(score) else None,
        "ci95": [low, high],
        "permutation_p": permutation,
        "n_scored_sequences": len(medians),
        "measurement_level_spearman": (round(float(measurement_score), 3)
                                       if np.isfinite(measurement_score)
                                       else None),
        "per_fold_mean": round(float(np.mean(folds)), 3) if folds else None,
        "n_folds": len(folds),
        "alpha_profile": alpha_profile(x, y, groups, measurements,
                                       components, standardise),
    }


def verdict(scores: dict[str, float | None]) -> str:
    """Whether a backbone beat both controls, and whether anything beat the
    protocol-only baseline."""
    baseline = scores.get("none")
    control = max((scores[name] for name in CONTROLS
                   if scores.get(name) is not None), default=None)
    model = max((scores[name] for name in MODELS
                 if scores.get(name) is not None), default=None)

    if control is None or model is None:
        return "unscoreable"

    parts = []
    if model > control:
        parts.append(f"model beats controls by {model - control:+.3f}")
    else:
        parts.append(f"controls win by {control - model:+.3f}")
    if baseline is not None:
        parts.append(f"best sequence block is {max(control, model) - baseline:+.3f} "
                     f"on the protocol alone")
    return "; ".join(parts)


def evaluate(target: str, include_llm: bool, index: dict[str, int], cached,
             standardise: bool) -> dict | None:
    """Everything reported for one target."""
    measurements = read_rows(target, include_llm, index)
    components = labels.read_components()

    sequences = {m.seq_md5 for m in measurements}
    groups = [components[m.seq_md5] for m in measurements]
    n_components = len(set(groups))

    header = (f"{target}: {len(measurements)} measurements over "
              f"{len(sequences)} sequences, {n_components} components")
    if (len(measurements) < MIN_MEASUREMENTS
            or not labels.is_scoreable(len(sequences), n_components)):
        print(f"{header}  -> SKIPPED, too few to score")
        return None
    print(header)

    y = np.array([m.value for m in measurements])
    blocks, protocol, protocol_names = build_blocks(measurements, index, cached)

    repeats = collections.Counter(m.seq_md5 for m in measurements)
    print(f"  measurements per sequence: median "
          f"{int(np.median(list(repeats.values())))}, "
          f"max {max(repeats.values())}")
    print(f"  protocol columns: {len(protocol_names)}")
    print(f"  target spread: sd {y.std():.3f}, "
          f"range {y.min():+.2f} to {y.max():+.2f}")
    print()

    result = {
        "n_measurements": len(measurements),
        "n_sequences": len(sequences),
        "n_components": n_components,
        "target_sd": round(float(y.std()), 3),
        "protocol_columns": protocol_names,
        "features": {},
    }

    # Protocol alone, then each sequence block joined to the protocol, then
    # the sequence block by itself so these numbers can be read against the
    # aggregated sweep. `rho` is the headline, one point per sequence; `meas`
    # is the same fit scored over measurement rows, kept only as a diagnostic,
    # since it weights each sequence by how often it was assayed.
    print(f"  {'features':24} {'dim':>4} {'rho':>7} {'meas':>7} "
          f"{'per-fold':>9} {'p':>7}  CI")
    scores_with_protocol: dict[str, float | None] = {}
    scores_alone: dict[str, float | None] = {}

    for name in SEQUENCE_BLOCKS:
        block = blocks.get(name)
        variants = [("with_protocol", join(block, protocol))]
        if block is not None:
            variants.append(("alone", join(block, None)))

        for suffix, x in variants:
            entry = score_set(x, y, groups, measurements, components,
                              standardise)
            result["features"][f"{name}_{suffix}"] = entry

            if suffix == "with_protocol":
                scores_with_protocol[name] = entry["spearman"]
            else:
                scores_alone[name] = entry["spearman"]

            print(f"  {name + ' ' + suffix:24} {entry['dim']:>4} "
                  f"{str(entry['spearman']):>7} "
                  f"{str(entry['measurement_level_spearman']):>7} "
                  f"{str(entry['per_fold_mean']):>9} "
                  f"{entry['permutation_p']:>7}  {entry['ci95']}")

    result["verdict_with_protocol"] = verdict(scores_with_protocol)
    result["verdict_alone"] = verdict(scores_alone)
    print()
    print(f"  joined to the protocol: {result['verdict_with_protocol']}")
    print(f"  sequence features alone: {result['verdict_alone']}")
    print()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--targets", default="",
                        help="comma separated; default is every named target")
    parser.add_argument("--standardise", action="store_true",
                        help="z-score features per fold, fitted on the "
                             "training rows only")
    parser.add_argument("--include-llm", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cached = np.load(args.embeddings, allow_pickle=False)
    index = {name: position for position, name in enumerate(cached["names"])}
    chosen = ([t.strip() for t in args.targets.split(",") if t.strip()]
              or list(labels.TARGETS))

    print(f"embeddings: {len(index)} sequences")
    print(f"standardised: {args.standardise}, llm labels: {args.include_llm}")
    print()

    results = {}
    for target in chosen:
        outcome = evaluate(target, args.include_llm, index, cached,
                           args.standardise)
        if outcome is not None:
            results[target] = outcome

    print("=" * 78)
    print("A sequence block is only interesting where it beats the protocol")
    print("alone, and a backbone only where it also beats length and")
    print("composition given the same protocol.")
    print("=" * 78)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
