"""Is the target a property of the sequence, or of the assay protocol?

The first evaluation sweep found that no backbone beat sequence length on any
target, and the power simulation showed that for kcat, km and kcat_over_km
that null result was not a sample-size artefact: those targets have enough
independent families to detect a correlation of 0.3, and nothing reached it.

That leaves the labels themselves as the suspect. Every value in the release
is a number attached to a protocol, and the pooled `kcat` column mixes forty
substrates whose means span three orders of magnitude, including fifty
acetylcholine measurements that are not a plastic reaction at all. If the
target is mostly protocol, no representation of the sequence can predict it,
and the fix is a labels fix rather than a model fix.

Four tests, in the order their answers depend on each other:

  1. reliability   How much of the target varies between sequences, and how
                   much between repeat measurements of the same sequence?
                   The second quantity caps the correlation anything can
                   reach, so it bounds every later number.

  2. protocol      How well do substrate, pH, temperature and incubation time
                   predict the value on their own, out of fold? A high score
                   here means the column is largely a record of how the
                   experiment was run.

  3. stratified    Restrict to one substrate at a time, so the chemistry is
                   held fixed, and re-run the full comparison of controls
                   against both backbones. This is the clean version of the
                   question the first sweep asked.

  4. residualised  Subtract the out-of-fold protocol prediction from every
                   measurement, aggregate what is left per sequence, and
                   sweep that. If a sequence signal was buried under protocol
                   variance, it surfaces here.

Usage:
    python scripts/eval_conditions.py --embeddings logs/label_embeddings.npz \\
        --out logs/eval_conditions.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from petabite.eval_module import conditions, labels, stats

FEATURE_SETS = ("length", "composition", "stock", "plasticesm")
CONTROLS = ("length", "composition")
MODELS = ("stock", "plasticesm")

# Targets are examined in descending order of how much is riding on them.
# PET monomer release is the project's actual objective; the kinetics columns
# are the ones with enough families to be scoreable.
EXAMINED = ("pet_release_conc", "kcat", "km", "kcat_over_km", "relative_activity")


# --------------------------------------------------------------------------
# Test 1: how much of the target is the sequence, and how much is the assay?
# --------------------------------------------------------------------------

def reliability(measurements: list[labels.Measurement]) -> dict:
    """Split the variance into between-sequence and within-sequence parts.

    The within-sequence part is measurement noise as far as any
    sequence-to-function model is concerned: two readings of the same protein
    that disagree cannot both be predicted. Its size relative to the
    between-sequence part gives the reliability of the per-sequence median,
    and the square root of that reliability is the ceiling on any correlation
    a model can achieve, however good its representation.
    """
    by_sequence: dict[str, list[float]] = collections.defaultdict(list)
    for m in measurements:
        by_sequence[m.seq_md5].append(m.value)

    replicated = [v for v in by_sequence.values() if len(v) >= 2]
    n_replicated_rows = sum(len(v) for v in replicated)

    # Pooled within-sequence variance, weighted by how many readings each
    # sequence contributes, so a sequence with twenty readings counts for
    # more than one with two.
    if n_replicated_rows:
        within = sum(np.var(v) * len(v) for v in replicated) / n_replicated_rows
    else:
        within = 0.0

    medians = np.array([float(np.median(v)) for v in by_sequence.values()])
    between = float(np.var(medians))

    # Reliability of a median built from the typical number of readings.
    per_sequence = np.array([len(v) for v in by_sequence.values()])
    typical = float(np.median(per_sequence))
    noise_in_median = within / max(typical, 1.0)
    denominator = between + noise_in_median
    reliability_score = between / denominator if denominator > 0 else float("nan")

    return {
        "n_measurements": len(measurements),
        "n_sequences": len(by_sequence),
        "n_replicated_sequences": len(replicated),
        "median_measurements_per_sequence": typical,
        "within_sequence_variance": round(float(within), 4),
        "between_sequence_variance": round(between, 4),
        "within_share_of_total": round(float(within) / (float(within) + between), 4)
        if (within + between) > 0 else None,
        "median_reliability": round(float(reliability_score), 4),
        "correlation_ceiling": round(float(np.sqrt(max(reliability_score, 0.0))), 4),
    }


# --------------------------------------------------------------------------
# Test 2: how much of the target does the protocol alone explain?
# --------------------------------------------------------------------------

def protocol_share(measurements: list[labels.Measurement],
                   components: dict[str, str]) -> dict:
    """Score a conditions-only model, out of fold, at measurement level.

    The comparison that matters is against the backbone scores from the main
    sweep. A protocol score well above those means the column records the
    experiment more faithfully than it records the protein.
    """
    y = np.array([m.value for m in measurements])
    groups = [components.get(m.seq_md5, m.seq_md5) for m in measurements]
    predicted = conditions.predict_from_conditions(measurements, components)

    rho = spearmanr(predicted, y).statistic
    low, high = stats.bootstrap_ci(y, predicted, groups)
    _, names = conditions.design_matrix(measurements)

    return {
        "n_measurements": len(measurements),
        "n_features": len(names),
        "spearman": round(float(rho), 3) if np.isfinite(rho) else None,
        "ci95": [low, high],
        "permutation_p": stats.permutation_p(y, predicted, groups),
    }


def substrate_table(measurements: list[labels.Measurement],
                    components: dict[str, str]) -> list[dict]:
    """Per-substrate counts and means, to show where the pooling hurts."""
    by_substrate: dict[str, list[labels.Measurement]] = collections.defaultdict(list)
    for m in measurements:
        by_substrate[m.substrate].append(m)

    rows = []
    for substrate, group in by_substrate.items():
        sequences = {m.seq_md5 for m in group}
        families = {components.get(md5) for md5 in sequences}
        values = np.array([m.value for m in group])
        rows.append({
            "substrate": substrate or "(blank)",
            "n_measurements": len(group),
            "n_sequences": len(sequences),
            "n_components": len(families),
            "mean": round(float(values.mean()), 3),
            "sd": round(float(values.std()), 3),
            "scoreable": labels.is_scoreable(len(sequences), len(families)),
        })
    return sorted(rows, key=lambda row: -row["n_measurements"])


# --------------------------------------------------------------------------
# Shared scoring: one target vector against every feature set
# --------------------------------------------------------------------------

def score_feature_sets(y: np.ndarray, groups: list[str], rows: np.ndarray,
                       features: dict[str, np.ndarray],
                       power_trials: int = 0) -> dict:
    """Grouped 5-fold score for each feature set, plus the verdict.

    Leave-one-component-out is skipped here. The main sweep ran both and they
    agreed, and the stratified targets have few enough components that one
    model fit per component buys nothing but runtime.
    """
    result: dict[str, object] = {"features": {}}

    for name in FEATURE_SETS:
        x = features[name][rows]
        predicted = stats.grouped_out_of_fold(x, y, groups, n_splits=5)
        rho = spearmanr(predicted, y).statistic
        low, high = stats.bootstrap_ci(y, predicted, groups)
        result["features"][name] = {
            "dim": int(x.shape[1]),
            "spearman": round(float(rho), 3) if np.isfinite(rho) else None,
            "ci95": [low, high],
            "permutation_p": stats.permutation_p(y, predicted, groups),
        }

    scores = {name: result["features"][name]["spearman"] for name in FEATURE_SETS}
    best_control = max((scores[n] for n in CONTROLS if scores[n] is not None),
                       default=None)
    best_model = max((scores[n] for n in MODELS if scores[n] is not None),
                     default=None)

    if best_control is None or best_model is None:
        result["verdict"] = "unscoreable"
    elif best_model > best_control:
        result["verdict"] = f"model wins by {best_model - best_control:+.3f}"
    else:
        result["verdict"] = f"controls win by {best_control - best_model:+.3f}"

    if power_trials:
        power = stats.power_curve(y, groups, trials=power_trials)
        result["power"] = {str(rho): value for rho, value in power.items()}
        detectable = stats.smallest_detectable(power)
        result["smallest_detectable_rho"] = detectable

    return result


def print_scores(result: dict) -> None:
    """One line per feature set, then the verdict."""
    for name in FEATURE_SETS:
        entry = result["features"][name]
        print(f"    {name:12} dim {entry['dim']:>4}  rho {str(entry['spearman']):>7}  "
              f"CI {entry['ci95']}  p {entry['permutation_p']}")
    print(f"    verdict: {result['verdict']}")
    if "smallest_detectable_rho" in result:
        detectable = result["smallest_detectable_rho"]
        if detectable is None:
            print("    power: no simulated correlation up to 0.8 would be detected")
        else:
            print(f"    power: 80% power starts at true rho = {detectable:.1f}")


def align_to_embeddings(values: dict[str, float], index: dict[str, int],
                        components: dict[str, str]):
    """Keep sequences that were embedded, and return target, groups and rows."""
    usable = sorted(md5 for md5 in values if md5 in index)
    y = np.array([values[md5] for md5 in usable])
    groups = [components[md5] for md5 in usable]
    rows = np.array([index[md5] for md5 in usable], dtype=int)
    return y, groups, rows


# --------------------------------------------------------------------------
# Test 3: one substrate at a time
# --------------------------------------------------------------------------

def stratified(measurements: list[labels.Measurement], components: dict[str, str],
               index: dict[str, int], features: dict[str, np.ndarray],
               power_trials: int) -> dict:
    """Score each substrate stratum that survives the count floors.

    Holding the substrate fixed is the cleanest available version of the
    question. It costs data: the biggest stratum of the biggest kinetics
    target keeps 73 of 227 sequences. That is why the power simulation runs
    on every stratum, so a null result here can be told apart from an
    underpowered one.
    """
    by_substrate: dict[str, list[labels.Measurement]] = collections.defaultdict(list)
    for m in measurements:
        by_substrate[m.substrate].append(m)

    out: dict[str, dict] = {}
    for substrate, group in sorted(by_substrate.items(),
                                   key=lambda kv: -len(kv[1])):
        values = labels.aggregate(group)
        y, groups, rows = align_to_embeddings(values, index, components)
        n_components = len(set(groups))

        if not labels.is_scoreable(len(y), n_components):
            continue

        print(f"  substrate {substrate or '(blank)'}: {len(y)} sequences, "
              f"{n_components} components")
        result = score_feature_sets(y, groups, rows, features, power_trials)
        result["n_sequences"] = len(y)
        result["n_components"] = n_components
        result["target_sd"] = round(float(y.std()), 3)
        print_scores(result)
        out[substrate or "(blank)"] = result

    if not out:
        print("  no substrate stratum survives the count floors")
    return out


# --------------------------------------------------------------------------
# Test 4: the target with its protocol regressed out
# --------------------------------------------------------------------------

def residualised(measurements: list[labels.Measurement], components: dict[str, str],
                 index: dict[str, int], features: dict[str, np.ndarray],
                 power_trials: int) -> dict:
    """Sweep the target again after subtracting the protocol prediction.

    Residualising keeps every sequence, unlike stratifying, which is the
    point: it trades the clean chemistry of a single substrate for the full
    sample size. The two tests fail in different directions, so agreeing is
    informative.
    """
    residuals = conditions.residualise(measurements)

    per_sequence: dict[str, list[float]] = collections.defaultdict(list)
    for m, value in zip(measurements, residuals):
        per_sequence[m.seq_md5].append(float(value))
    values = {md5: float(np.median(v)) for md5, v in per_sequence.items()}

    y, groups, rows = align_to_embeddings(values, index, components)
    n_components = len(set(groups))
    print(f"  {len(y)} sequences, {n_components} components, "
          f"residual sd {y.std():.3f}")

    if not labels.is_scoreable(len(y), n_components):
        print("  too few to score")
        return {"skipped": True, "n_sequences": len(y), "n_components": n_components}

    result = score_feature_sets(y, groups, rows, features, power_trials)
    result["n_sequences"] = len(y)
    result["n_components"] = n_components
    result["residual_sd"] = round(float(y.std()), 3)
    print_scores(result)
    return result


# --------------------------------------------------------------------------

def examine(target: str, include_llm: bool, components: dict[str, str],
            index: dict[str, int], features: dict[str, np.ndarray],
            power_trials: int) -> dict:
    """Run all four tests on one target."""
    measurements = labels.read_measurements(target, include_llm=include_llm)
    print("=" * 78)
    print(f"{target}  ({len(measurements)} measurements)")
    print("=" * 78)

    if not measurements:
        print("  no usable measurements")
        return {"skipped": True}

    print("\n-- 1. reliability --")
    scores = reliability(measurements)
    print(f"  {scores['n_sequences']} sequences, "
          f"{scores['n_replicated_sequences']} with repeat measurements "
          f"(median {scores['median_measurements_per_sequence']:.0f} each)")
    print(f"  variance between sequences {scores['between_sequence_variance']}, "
          f"within a sequence {scores['within_sequence_variance']}")
    print(f"  the within-sequence share of the total is "
          f"{scores['within_share_of_total']:.1%}")
    print(f"  reliability of the per-sequence median {scores['median_reliability']:.3f}, "
          f"so no model can correlate above {scores['correlation_ceiling']:.3f}")

    print("\n-- 2. protocol alone --")
    protocol = protocol_share(measurements, components)
    print(f"  {protocol['n_features']} condition features over "
          f"{protocol['n_measurements']} measurements")
    print(f"  out-of-fold rho {protocol['spearman']}  CI {protocol['ci95']}  "
          f"p {protocol['permutation_p']}")
    substrates = substrate_table(measurements, components)
    print(f"  {len(substrates)} distinct substrates; the largest few:")
    for row in substrates[:6]:
        print(f"    {row['n_measurements']:5} meas  {row['n_sequences']:4} seq  "
              f"{row['n_components']:3} comp  mean {row['mean']:+7.2f}  "
              f"sd {row['sd']:5.2f}  {row['substrate'][:40]}")

    print("\n-- 3. one substrate at a time --")
    strata = stratified(measurements, components, index, features, power_trials)

    print("\n-- 4. protocol regressed out --")
    residual = residualised(measurements, components, index, features,
                            power_trials)
    print()

    return {
        "reliability": scores,
        "protocol": protocol,
        "substrates": substrates,
        "stratified": strata,
        "residualised": residual,
    }


def summarise(results: dict[str, dict]) -> None:
    """One table collecting whether anything beat the controls, anywhere."""
    print("=" * 78)
    print("SUMMARY: every scoreable target variant, controls against backbones")
    print("=" * 78)
    print(f"{'variant':38} {'n':>4} {'comp':>5} {'ctrl':>7} {'model':>7}  verdict")

    def emit(name: str, result: dict) -> None:
        scores = {k: result["features"][k]["spearman"] for k in FEATURE_SETS}
        control = max((scores[k] for k in CONTROLS if scores[k] is not None),
                      default=None)
        model = max((scores[k] for k in MODELS if scores[k] is not None),
                    default=None)
        print(f"{name[:38]:38} {result['n_sequences']:>4} "
              f"{result['n_components']:>5} {str(control):>7} {str(model):>7}  "
              f"{result['verdict']}")

    for target, result in results.items():
        if result.get("skipped"):
            continue
        for substrate, stratum in result["stratified"].items():
            emit(f"{target} @ {substrate}", stratum)
        if not result["residualised"].get("skipped"):
            emit(f"{target} (protocol removed)", result["residualised"])
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--include-llm", action="store_true",
                        help="also use the LLM-extracted labels")
    parser.add_argument("--targets", nargs="*", default=list(EXAMINED))
    parser.add_argument("--power-trials", type=int, default=100,
                        help="simulation draws per effect size; 0 skips the "
                             "power analysis, which is the slowest part")
    args = parser.parse_args()

    cached = np.load(args.embeddings, allow_pickle=False)
    names = list(cached["names"])
    index = {name: position for position, name in enumerate(names)}
    features = {key: cached[key] for key in FEATURE_SETS}
    components = labels.read_components()

    print(f"embeddings: {len(names)} sequences")
    print(f"llm labels included: {args.include_llm}\n")

    results = {
        target: examine(target, args.include_llm, components, index, features,
                        args.power_trials)
        for target in args.targets
    }
    summarise(results)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
