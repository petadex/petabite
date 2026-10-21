"""Does the score move when only the BLAS thread count changes?

A ridge fit of 960 columns on 73 rows with a light penalty is numerically
close to singular, and one log of the first layer sweep carried 89
ill-conditioning warnings alongside several singular-matrix fallbacks. A
linear algebra result that depends on how the library happened to parallelise
the solve is not a measurement, so this script exists to find out whether that
is what the headline number was.

Run it twice with different thread counts and compare. Identical output means
the fits are stable; output that moves means the raw, unstandardised
configuration cannot be reported at all, whatever its score.

Usage:
    OMP_NUM_THREADS=1 python scripts/eval_determinism.py \\
        --embeddings logs/label_embeddings.npz \\
        --target kcat_over_km --substrate p-np-c4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from petabite.eval_module import labels, stats

FEATURE_SETS = ("length", "composition", "stock", "plasticesm")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--substrate", default=None)
    args = parser.parse_args()

    cached = np.load(args.embeddings, allow_pickle=False)
    index = {name: position for position, name in enumerate(cached["names"])}
    components = labels.read_components()

    measurements = labels.read_measurements(args.target)
    if args.substrate is not None:
        measurements = [m for m in measurements if m.substrate == args.substrate]
    per_sequence = labels.aggregate(measurements)

    usable = sorted(md5 for md5 in per_sequence if md5 in index)
    y = np.array([per_sequence[md5] for md5 in usable])
    groups = [components[md5] for md5 in usable]
    rows = np.array([index[md5] for md5 in usable], dtype=int)

    threads = os.environ.get("OMP_NUM_THREADS", "unset")
    print(f"{args.target} on {args.substrate}: {len(y)} sequences, "
          f"{len(set(groups))} components, OMP_NUM_THREADS={threads}")

    for name in FEATURE_SETS:
        x = cached[name][rows]
        scores = []
        for standardise in (False, True):
            predicted = stats.grouped_out_of_fold(x, y, groups, n_splits=5,
                                                  standardise=standardise)
            score = spearmanr(predicted, y).statistic
            scores.append(float(score) if np.isfinite(score) else float("nan"))
        print(f"  {name:12} raw {scores[0]:+.4f}   standardised {scores[1]:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
