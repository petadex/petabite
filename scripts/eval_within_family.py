"""Can a backbone rank variants inside one protein family?

Every earlier measurement asked whether a model generalises to a family it
has never seen, and the answer was no for every target and feature set. The
leakage analysis explains why that question is hard and also why published
numbers look better: a 960-dimensional embedding inflates by 0.49 to 1.01
under a random split because it can recognise a family it has already met.

This script asks the other question. Within one family, can the model order
the variants? That is what protein engineering actually needs, since a
campaign starts from a known scaffold and has to choose among its mutants,
and it is immune to the family recognition that drives the leakage.

The data supports it in exactly one place. Component 00855c73 holds 546
sequences, of which 113 carry a PET release value spanning -3.30 to +4.52 in
log10 micromolar, 76 carry kcat_over_km and 47 carry a melting temperature.
That one family was previously collapsed to a single unit of independence,
which discarded all of this.

Two settings, because they answer different questions and the weaker one is
easy to mistake for the stronger:

  cross_family   Train on every other component, then predict this family's
                 members. The model has never seen the scaffold. Strict, and
                 the direct successor to the earlier evaluation.

  within_family  Train and test inside the family, holding out whole
                 90%-identity clusters. The model has seen the scaffold but
                 not these variants. This is the engineering setting and the
                 honest unit of independence is the cluster, not the
                 sequence, so near-identical variants never straddle a split.

The controls matter more here than anywhere else. Within a scaffold family a
model can look good simply by tracking how far a variant has drifted from the
wild type, so identity to the family medoid is scored alongside length and
composition. A backbone that cannot beat "count the mutations" is not useful
for choosing variants.

Usage:
    python scripts/eval_within_family.py --embeddings logs/label_embeddings.npz \\
        --target pet_release_conc --out logs/within_family.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices
from scipy.stats import spearmanr

from petabite.eval_module import labels, stats

# The family carrying almost all of the dense label coverage. Components are
# named by a hash prefix in the v2 release.
DEFAULT_FAMILY = "00855c73c087597a"

# Controls come first in every table, deliberately: the question is always
# whether a backbone adds anything to them.
CONTROLS = ("length", "composition", "identity", "mutations")
MODELS = ("stock", "plasticesm")
# Zero-shot likelihood scores, present only when --zero-shot is given. They
# count as models rather than controls: nothing is fitted to produce them, but
# they are still the backbone's opinion about the sequence.
ZERO_SHOT = ("pll_stock", "pll_plasticesm")
FEATURE_SETS = CONTROLS + MODELS

# Grouping unit inside the family. cluster_90 gives the most groups and so the
# most power; cluster_50 is the stricter check.
WITHIN_GROUPINGS = ("cluster_90", "cluster_50")


def read_clusters() -> dict[str, dict[str, str]]:
    """Per-sequence component and cluster assignments from the release."""
    import csv

    out: dict[str, dict[str, str]] = {}
    with (labels.LABELS / "sequences.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["seq_md5"]] = {
                "component": row["component"],
                "cluster_90": row["cluster_90"],
                "cluster_50": row["cluster_50"],
                "length": row["length"],
            }
    return out


def build_aligner() -> PairwiseAligner:
    """A global protein aligner with conventional gap costs.

    BLOSUM62 with -11/-1 is the standard choice for this, and the absolute
    scores do not matter here because only the derived identity and
    substitution counts are used.
    """
    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.mode = "global"
    return aligner


def identity(aligner: PairwiseAligner, left: str, right: str) -> tuple[float, int, int]:
    """Fraction identical, substitution count and gap count for one pair."""
    alignment = aligner.align(left, right)[0]
    top, bottom = alignment[0], alignment[1]

    same = substitutions = gaps = 0
    for a, b in zip(top, bottom):
        if a == "-" or b == "-":
            gaps += 1
        elif a == b:
            same += 1
        else:
            substitutions += 1

    aligned = same + substitutions
    return (same / aligned if aligned else 0.0), substitutions, gaps


def variant_features(sequences: dict[str, str],
                     names: list[str]) -> tuple[dict[str, np.ndarray], str]:
    """Distance-from-wild-type controls, built around the family medoid.

    The wild type is not recorded in the release, so the medoid stands in for
    it: the labelled sequence with the highest mean identity to the others.
    That is the most typical member of the family, which is what a mutation
    count should be measured against.
    """
    aligner = build_aligner()

    # Pairwise identities over the labelled subset only. A few thousand global
    # alignments of 300-residue sequences, which is seconds.
    size = len(names)
    matrix = np.ones((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            score, _, _ = identity(aligner, sequences[names[i]], sequences[names[j]])
            matrix[i, j] = matrix[j, i] = score

    medoid_index = int(np.argmax(matrix.mean(axis=1)))
    reference = names[medoid_index]

    identities, substitutions, gaps = [], [], []
    for name in names:
        fraction, n_sub, n_gap = identity(aligner, sequences[reference],
                                          sequences[name])
        identities.append(fraction)
        substitutions.append(n_sub)
        gaps.append(n_gap)

    return ({
        # Identity to the reference, plus how typical the variant is overall.
        "identity": np.column_stack([
            np.array(identities, dtype=np.float64),
            matrix.mean(axis=1),
        ]),
        # The mutation-count control, split into substitutions and indels
        # because they are not interchangeable in their effect.
        "mutations": np.column_stack([
            np.array(substitutions, dtype=np.float64),
            np.array(gaps, dtype=np.float64),
        ]),
    }, reference)


def score(x: np.ndarray, y: np.ndarray, groups: list[str],
          power_trials: int = 0) -> dict:
    """The standard scoring: grouped, fold-centred, standardised."""
    predicted = stats.grouped_out_of_fold(x, y, groups, n_splits=5,
                                          standardise=True)
    rho = spearmanr(predicted, y).statistic
    low, high = stats.bootstrap_ci(y, predicted, groups)
    folds = stats.fold_scores(x, y, groups, 5, standardise=True)

    entry = {
        "dim": int(x.shape[1]),
        "spearman": round(float(rho), 3) if np.isfinite(rho) else None,
        "ci95": [low, high],
        "permutation_p": stats.permutation_p(y, predicted, groups),
        "per_fold_mean": round(float(np.mean(folds)), 3) if folds else None,
        "n_folds": len(folds),
    }
    if power_trials:
        power = stats.power_curve(y, groups, trials=power_trials)
        entry["smallest_detectable_rho"] = stats.smallest_detectable(power)
    return entry


def score_cross_family(x_train, y_train, train_groups, x_test, y_test,
                       test_groups) -> dict:
    """Fit outside the family, predict inside it, score within it.

    One prediction pass over a single held-out family, so there are no folds
    to centre and the pooled artefact cannot arise. The interval resamples
    the family's own clusters, which are the independent units available here.
    """
    alpha = stats.pick_alpha(x_train, y_train, train_groups, standardise=True)
    predicted = stats.fit_predict(x_train, y_train, x_test, alpha,
                                  standardise=True)

    rho = spearmanr(predicted, y_test).statistic
    low, high = stats.bootstrap_ci(y_test, predicted, test_groups)
    return {
        "dim": int(x_train.shape[1]),
        "alpha": alpha,
        "spearman": round(float(rho), 3) if np.isfinite(rho) else None,
        "ci95": [low, high],
        "permutation_p": stats.permutation_p(y_test, predicted, test_groups),
    }


def verdict(scores: dict[str, float | None]) -> str:
    """Did any model-side feature clear every control, and by how much?"""
    control = max((scores[name] for name in CONTROLS
                   if scores.get(name) is not None), default=None)
    model = max((scores[name] for name in MODELS + ZERO_SHOT
                 if scores.get(name) is not None), default=None)
    if control is None or model is None:
        return "unscoreable"
    if model > control:
        return f"model wins by {model - control:+.3f}"
    return f"controls win by {control - model:+.3f}"


def report(title: str, entries: dict[str, dict]) -> None:
    print(f"  {title}")
    for name in CONTROLS + MODELS + ZERO_SHOT:
        entry = entries.get(name)
        if entry is None:
            continue
        extra = ""
        if entry.get("per_fold_mean") is not None:
            extra = f"  per-fold {entry['per_fold_mean']:+.3f}"
        if entry.get("smallest_detectable_rho") is not None:
            extra += f"  power@80 {entry['smallest_detectable_rho']:.1f}"
        marker = "  <- control" if name in CONTROLS else ""
        print(f"    {name:12} dim {entry['dim']:>4}  rho "
              f"{str(entry['spearman']):>7}  CI {entry['ci95']}  "
              f"p {entry['permutation_p']}{extra}{marker}")
    scores = {name: entries[name]["spearman"]
              for name in CONTROLS + MODELS + ZERO_SHOT if name in entries}
    print(f"    verdict: {verdict(scores)}")


def unfitted_table(features: dict[str, np.ndarray], y: np.ndarray,
                   groups: list[str]) -> dict[str, dict]:
    """Direct rank correlation for every single-column feature.

    No model, no folds, no penalty, so none of the faults that invalidated
    earlier numbers can arise. This is the cleanest comparison available and
    it is the natural home of a zero-shot likelihood, which needs no fitting
    in the first place. Multi-column feature sets are absent by construction.
    """
    out: dict[str, dict] = {}
    for name, matrix in features.items():
        if matrix.shape[1] != 1:
            continue
        column = matrix[:, 0]
        usable = np.isfinite(column)
        if usable.sum() < labels.MIN_SEQUENCES:
            continue
        rho = spearmanr(column[usable], y[usable]).statistic
        kept = [g for g, keep in zip(groups, usable) if keep]
        low, high = stats.bootstrap_ci(y[usable], column[usable], kept)
        out[name] = {
            "spearman": round(float(rho), 3) if np.isfinite(rho) else None,
            "ci95": [low, high],
            "permutation_p": stats.permutation_p(y[usable], column[usable],
                                                 kept),
            "n": int(usable.sum()),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--family", default=DEFAULT_FAMILY)
    parser.add_argument("--include-llm", action="store_true")
    parser.add_argument("--zero-shot", type=Path, default=None,
                        help="npz of pseudo-log-likelihoods from "
                             "zero_shot_likelihood.py")
    parser.add_argument("--power-trials", type=int, default=100)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cached = np.load(args.embeddings, allow_pickle=False)
    index = {name: position for position, name in enumerate(cached["names"])}
    meta = read_clusters()
    sequences = labels.read_sequences()

    measurements = labels.read_measurements(args.target, args.include_llm)
    per_sequence = labels.aggregate(measurements)

    inside, outside = [], []
    for md5 in sorted(per_sequence):
        if md5 not in index or md5 not in meta:
            continue
        (inside if meta[md5]["component"].startswith(args.family)
         else outside).append(md5)

    print(f"target {args.target}, family {args.family}")
    print(f"  {len(inside)} labelled sequences inside the family, "
          f"{len(outside)} outside")
    if len(inside) < labels.MIN_SEQUENCES:
        print("  too few inside the family to score")
        return 1

    y_in = np.array([per_sequence[md5] for md5 in inside])
    rows_in = np.array([index[md5] for md5 in inside], dtype=int)
    print(f"  target inside the family: sd {y_in.std():.3f}, "
          f"range {y_in.min():+.2f} to {y_in.max():+.2f}")

    print("  aligning to find the family medoid")
    extra, reference = variant_features(sequences, inside)
    print(f"  reference (medoid): {reference}")
    print(f"  identity to it: median "
          f"{np.median(extra['identity'][:, 0]):.3f}, "
          f"min {extra['identity'][:, 0].min():.3f}")
    print(f"  substitutions from it: median "
          f"{np.median(extra['mutations'][:, 0]):.0f}, "
          f"max {extra['mutations'][:, 0].max():.0f}")

    # Zero-shot likelihoods, aligned to the same md5 ordering.
    zero_shot: dict[str, dict[str, float]] = {}
    if args.zero_shot is not None:
        loaded = np.load(args.zero_shot, allow_pickle=False)
        zero_names = list(loaded["names"])
        for key in ZERO_SHOT:
            column = key.replace("pll_", "pll_")
            if column in loaded.files:
                zero_shot[key] = dict(zip(zero_names, loaded[column]))
        print(f"  zero-shot scores loaded for {len(zero_names)} sequences: "
              f"{sorted(zero_shot)}")

    def features_for(rows: np.ndarray, names: list[str],
                     local: bool) -> dict[str, np.ndarray]:
        built = {name: cached[name][rows] for name in
                 ("length", "composition", "stock", "plasticesm")}
        if local:
            built.update(extra)
        for key, table in zero_shot.items():
            built[key] = np.array(
                [[table.get(name, np.nan)] for name in names],
                dtype=np.float64)
        return built

    results: dict[str, dict] = {
        "target": args.target,
        "family": args.family,
        "reference": reference,
        "n_inside": len(inside),
        "n_outside": len(outside),
        "target_sd": round(float(y_in.std()), 3),
    }

    inside_features = features_for(rows_in, inside, local=True)

    print("\n-- unfitted: direct rank correlation, no model and no folds --")
    direct_groups = [meta[md5]["cluster_90"] for md5 in inside]
    direct = unfitted_table(inside_features, y_in, direct_groups)
    for name, entry in direct.items():
        marker = "  <- control" if name in CONTROLS else ""
        print(f"    {name:14} n {entry['n']:>4}  rho "
              f"{str(entry['spearman']):>7}  CI {entry['ci95']}  "
              f"p {entry['permutation_p']}{marker}")
    results["unfitted_within_family"] = direct

    print("\n-- within family, holding out whole identity clusters --")
    for grouping in WITHIN_GROUPINGS:
        groups = [meta[md5][grouping] for md5 in inside]
        n_groups = len(set(groups))
        if n_groups < labels.MIN_COMPONENTS:
            print(f"  {grouping}: only {n_groups} groups, skipped")
            continue
        entries = {
            name: score(matrix, y_in, groups, args.power_trials)
            for name, matrix in inside_features.items()
        }
        report(f"{grouping}: {n_groups} groups over {len(y_in)} sequences",
               entries)
        results[f"within_{grouping}"] = {"n_groups": n_groups,
                                         "features": entries}

    print("\n-- cross family, trained on every other component --")
    if len(outside) < labels.MIN_SEQUENCES:
        print("  too few sequences outside the family")
    else:
        y_out = np.array([per_sequence[md5] for md5 in outside])
        rows_out = np.array([index[md5] for md5 in outside], dtype=int)
        train_groups = [meta[md5]["component"] for md5 in outside]
        test_groups = [meta[md5]["cluster_90"] for md5 in inside]
        outside_features = features_for(rows_out, outside, local=False)

        entries = {}
        for name, matrix in outside_features.items():
            entries[name] = score_cross_family(
                matrix, y_out, train_groups,
                inside_features[name], y_in, test_groups)
        report(f"trained on {len(outside)} sequences from "
               f"{len(set(train_groups))} components", entries)
        results["cross_family"] = {"n_train": len(outside),
                                   "n_train_components": len(set(train_groups)),
                                   "features": entries}

    print()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
