"""Which layer of ESM-C actually knows about catalytic activity?

Pairs with `embed_layers.py`. For one target, or one substrate stratum of it,
score every cached layer of both backbones against the same controls the rest
of the evaluation uses. The output is a profile across depth rather than a
single number, which answers two things at once: whether the final layer was a
poor choice, and whether Stage 1's adaptation shows up more clearly somewhere
other than the top of the stack.

Usage:
    python scripts/eval_layers.py --embeddings logs/layer_embeddings.npz \\
        --target kcat_over_km --substrate p-np-c4
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from petabite.eval_module import labels, stats

CONTROLS = ("length", "composition")


def layer_keys(cached) -> dict[str, list[int]]:
    """Which backbones and layers the cache holds."""
    found: dict[str, list[str]] = {}
    for key in cached.files:
        match = re.fullmatch(r"(stock|plasticesm)_(L\d+|post)", key)
        if match:
            found.setdefault(match.group(1), []).append(match.group(2))

    def order(name: str) -> int:
        # The post-norm representation sits after the last block, so it sorts
        # last.
        return 1_000 if name == "post" else int(name[1:])

    return {backbone: sorted(names, key=order)
            for backbone, names in found.items()}


def select(target: str, substrate: str | None, include_llm: bool,
           components: dict[str, str], index: dict[str, int]):
    """Per-sequence values, optionally restricted to one substrate."""
    measurements = labels.read_measurements(target, include_llm)
    if substrate is not None:
        measurements = [m for m in measurements if m.substrate == substrate]
    per_sequence = labels.aggregate(measurements)

    usable = sorted(md5 for md5 in per_sequence if md5 in index)
    y = np.array([per_sequence[md5] for md5 in usable])
    groups = [components[md5] for md5 in usable]
    rows = np.array([index[md5] for md5 in usable], dtype=int)
    return y, groups, rows


def score(x, y, groups) -> dict:
    """Grouped 5-fold score with its interval and permutation null.

    Standardised, unlike the rest of the evaluation, because the layers differ
    in scale by three orders of magnitude and an unstandardised comparison
    between them measures the ridge penalty rather than the representation.
    """
    predicted = stats.grouped_out_of_fold(x, y, groups, n_splits=5,
                                          standardise=True)
    rho = spearmanr(predicted, y).statistic
    low, high = stats.bootstrap_ci(y, predicted, groups)
    return {
        "spearman": round(float(rho), 3) if np.isfinite(rho) else None,
        "ci95": [low, high],
        "permutation_p": stats.permutation_p(y, predicted, groups),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--substrate", default=None,
                        help="restrict to one substrate; omit to use the "
                             "pooled target")
    parser.add_argument("--include-llm", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cached = np.load(args.embeddings, allow_pickle=False)
    index = {name: position for position, name in enumerate(cached["names"])}
    components = labels.read_components()
    available = layer_keys(cached)

    y, groups, rows = select(args.target, args.substrate, args.include_llm,
                             components, index)
    where = args.substrate or "pooled"
    print(f"{args.target} on {where}: {len(y)} sequences, "
          f"{len(set(groups))} components\n")

    if not labels.is_scoreable(len(y), len(set(groups))):
        print("too few to score")
        return 1

    results: dict[str, dict] = {"controls": {}, "layers": {}}
    best_control = None
    for name in CONTROLS:
        entry = score(cached[name][rows], y, groups)
        results["controls"][name] = entry
        print(f"  control {name:12} rho {str(entry['spearman']):>7}  "
              f"CI {entry['ci95']}")
        if entry["spearman"] is not None:
            best_control = (entry["spearman"] if best_control is None
                            else max(best_control, entry["spearman"]))

    print()
    print(f"  {'layer':>5}  {'stock':>22}  {'plasticesm':>22}")
    for layer in available.get("stock", []):
        row = {}
        for backbone in ("stock", "plasticesm"):
            if layer in available.get(backbone, []):
                row[backbone] = score(cached[f"{backbone}_{layer}"][rows],
                                      y, groups)
        results["layers"][layer] = row

        def cell(backbone: str) -> str:
            entry = row.get(backbone)
            if entry is None:
                return " " * 22
            marker = "*" if (best_control is not None
                             and entry["spearman"] is not None
                             and entry["spearman"] > best_control) else " "
            return (f"{str(entry['spearman']):>7} p {str(entry['permutation_p']):>6}"
                    f" {marker}")

        print(f"  {layer:>5}  {cell('stock'):>22}  {cell('plasticesm'):>22}")

    print()
    print(f"  best control rho {best_control}; a star marks a layer that beats it")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "target": args.target, "substrate": args.substrate,
            "n_sequences": len(y), "n_components": len(set(groups)),
            "best_control": best_control, **results,
        }, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
