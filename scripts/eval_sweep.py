"""Sweep every target, protocol and feature set, against trivial baselines.

The point of this script is to answer one question for each candidate target:
is there any signal a model can find that sequence length and amino acid
composition cannot? The first probe found that length alone beat both 960-
dimensional backbones on the fixed test fold, so a backbone result means
nothing until it clears those controls.

Three things are swept.

Targets. Each named target from the Stage 3 unit normalisation, aggregated to
one value per sequence. Concentration-like targets are log10 transformed
because they span orders of magnitude; non-positive values are dropped, as
they are control-subtracted readings with no logarithm.

Protocols. The fixed v2 split is kept for comparison, but the headline is
grouped cross-validation over *every* sequence carrying the target rather than
a 15-sequence holdout. Grouping on homology components preserves exactly the
guarantee the fixed split has, while turning 15 evaluation points into as many
as the target supports. Leave-one-component-out is the limiting case.

Feature sets. `length` and `composition` are the controls. `stock` and
`plasticesm` are the backbones. A backbone is only interesting where it beats
both controls.

Significance is a permutation test, because a Spearman correlation on a few
dozen points is easy to obtain by chance: the target is shuffled within
components many times and the observed value is placed in that null
distribution. Confidence intervals resample components, never sequences,
because sequences inside a component are not independent. Both tests live in
`petabite.eval_module.stats`, which this script shares with the power and
condition analyses.

Usage:
    python scripts/eval_sweep.py --embeddings logs/label_embeddings.npz \\
        --out logs/eval_sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from petabite.eval_module import labels, stats

FEATURE_SETS = ("length", "composition", "stock", "plasticesm")
CONTROLS = ("length", "composition")
MODELS = ("stock", "plasticesm")


def score_one(x: np.ndarray, y: np.ndarray, groups: list[str]) -> dict:
    """Both CV protocols for a single feature set."""
    entry: dict[str, object] = {"dim": int(x.shape[1])}

    for protocol, n_splits in (("kfold5", 5), ("loco", None)):
        predicted = stats.grouped_out_of_fold(x, y, groups, n_splits)
        rho = spearmanr(predicted, y).statistic
        low, high = stats.bootstrap_ci(y, predicted, groups)
        entry[protocol] = {
            "spearman": round(float(rho), 3) if np.isfinite(rho) else None,
            "ci95": [low, high],
            "permutation_p": stats.permutation_p(y, predicted, groups),
        }
    return entry


def verdict(scores: dict[str, float | None]) -> str:
    """Whether either backbone beat both controls, and by how much."""
    control = max((scores[name] for name in CONTROLS if scores[name] is not None),
                  default=None)
    model = max((scores[name] for name in MODELS if scores[name] is not None),
                default=None)
    if control is None or model is None:
        return "unscoreable"
    if model > control:
        return f"model wins by {model - control:+.3f}"
    return f"controls win by {control - model:+.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--include-llm", action="store_true",
                        help="also aggregate LLM-extracted labels")
    args = parser.parse_args()

    cached = np.load(args.embeddings, allow_pickle=False)
    names = list(cached["names"])
    index = {name: position for position, name in enumerate(names)}
    features = {key: cached[key] for key in FEATURE_SETS}
    components = labels.read_components()

    print(f"embeddings: {len(names)} sequences")
    print(f"llm labels included: {args.include_llm}\n")

    results: dict[str, dict] = {}
    for target in labels.TARGETS:
        measurements = labels.read_measurements(target, include_llm=args.include_llm)
        per_sequence = labels.aggregate(measurements)

        usable = sorted(md5 for md5 in per_sequence if md5 in index)
        groups = [components[md5] for md5 in usable]
        n_components = len(set(groups))
        header = f"{target}: {len(usable)} sequences, {n_components} components"

        if not labels.is_scoreable(len(usable), n_components):
            print(f"{header}  -> SKIPPED, too few to score")
            results[target] = {"skipped": True, "n_sequences": len(usable),
                               "n_components": n_components}
            continue
        print(header)

        rows = np.array([index[md5] for md5 in usable], dtype=int)
        y = np.array([per_sequence[md5] for md5 in usable])
        result = {"n_sequences": len(usable), "n_components": n_components,
                  "target_sd": round(float(y.std()), 3), "features": {}}

        for feature_name in FEATURE_SETS:
            entry = score_one(features[feature_name][rows], y, groups)
            result["features"][feature_name] = entry
            kfold = entry["kfold5"]
            print(f"  {feature_name:12} dim {entry['dim']:>4}  "
                  f"kfold5 rho {str(kfold['spearman']):>7}  CI {kfold['ci95']}  "
                  f"p {kfold['permutation_p']}")

        results[target] = result
        print()

    print("=" * 78)
    print(f"{'target':22} {'n':>4} {'comp':>5} {'length':>8} {'compos':>8} "
          f"{'stock':>8} {'plastic':>8}  verdict")
    for target, result in results.items():
        if result.get("skipped"):
            continue
        scores = {name: result["features"][name]["kfold5"]["spearman"]
                  for name in FEATURE_SETS}
        print(f"{target:22} {result['n_sequences']:>4} {result['n_components']:>5} "
              f"{str(scores['length']):>8} {str(scores['composition']):>8} "
              f"{str(scores['stock']):>8} {str(scores['plasticesm']):>8}  "
              f"{verdict(scores)}")
    print("=" * 78)
    print("A correlation whose CI contains zero, or whose permutation p is above")
    print("0.05, is not evidence of anything regardless of its size.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
