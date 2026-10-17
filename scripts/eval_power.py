"""What effect size could this label set detect, if one were there?

Every correlation in the sweep comes with a confidence interval, but intervals
answer "how uncertain is this estimate". The prior question is "what would we
have been able to see at all", and that is a property of the label set rather
than of any model.

The unit of independence is the homology component, not the sequence, because
sequences inside a component share ancestry. So the effective sample size for
a target is closer to its component count than its sequence count, and for the
PET release target that is 18 against 157.

The method is a simulation rather than a closed form, because the components
are wildly unequal in size and the usual formulas assume they are not. For a
range of true correlations, synthetic predictions are drawn with that
correlation to the real target, grouped exactly as the real data is, and the
fraction of draws whose component bootstrap excludes zero is the power. The
simulation itself lives in `petabite.eval_module.stats`.

Usage:
    python scripts/eval_power.py --target pet_release_conc
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from petabite.eval_module import labels, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="pet_release_conc")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--include-llm", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    measurements = labels.read_measurements(args.target,
                                            include_llm=args.include_llm)
    per_sequence = labels.aggregate(measurements)
    components = labels.read_components()

    names = sorted(per_sequence)
    y = np.array([per_sequence[name] for name in names])
    groups = [components[name] for name in names]
    sizes = collections.Counter(groups)

    print(f"target {args.target}")
    print(f"  {len(y)} sequences across {len(sizes)} components")
    print(f"  largest component holds {max(sizes.values())} "
          f"({100 * max(sizes.values()) / len(y):.0f}% of the data)")
    print(f"  median component size {int(np.median(list(sizes.values())))}")
    print()
    print(f"power to detect a correlation, {args.trials} trials each:")

    power = stats.power_curve(y, groups, trials=args.trials)
    for rho, value in power.items():
        bar = "#" * int(round(value * 40))
        print(f"  true rho {rho:.1f}   power {value:5.1%}  {bar}")

    detectable = stats.smallest_detectable(power)
    print()
    if detectable is not None:
        print(f"80% power starts at true rho = {detectable:.1f}")
        print(f"Anything weaker than that cannot be distinguished from zero with "
              f"{len(sizes)} components, however many sequences they contain.")
    else:
        print(f"No tested correlation reaches 80% power with {len(sizes)} "
              f"components. This target cannot support a claim of this kind.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "target": args.target,
            "n_sequences": len(y),
            "n_components": len(sizes),
            "power": {str(rho): value for rho, value in power.items()},
            "smallest_detectable_rho": detectable,
        }, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
