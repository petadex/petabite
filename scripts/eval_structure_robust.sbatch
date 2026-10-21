#!/bin/bash
# Stress the one candidate the structure features produced, and calibrate the
# acceptance test at the component count it was produced at.
#
# Job 6355612 folded all 397 labelled sequences carrying a scoreable target
# and scored four structure blocks against length and composition on exactly
# those sequences. Three targets came back negative. One did not:
#
#   melting_temperature   secondary +0.385, geometry +0.381
#                         best control +0.276 (length)
#                         jackknife [0.196, 0.547] and [0.125, 0.589]
#                         permutation p 0.013 and 0.012
#
# A margin of +0.109 over the best control with an interval excluding zero is
# the first thing in this project to reach the acceptance test's effect size,
# so it gets the full treatment rather than a sentence.
#
# Two things are measured here and they answer different questions.
#
# The stresses ask whether this particular number is stable: 40 random
# component-to-fold assignments, a refit with each of the 8 components
# dropped in turn, the whole alpha grid at fixed penalties, and the lot again
# with features raw instead of standardised. The permutation test is re-run
# with 10,000 draws because 1,000 resolves a p of 0.013 only to +/- 0.007,
# which straddles the 0.01 the acceptance test asks for.
#
# The calibration asks a prior question: how often does a structure block
# clear this bar when there is nothing to find? 8 components is the fewest
# any scoreable target has, and job 6352882 measured the null 95th percentile
# of |rho| at 0.316 for 8 components against 0.126 for 42. If a 3-column
# block at 8 components clears a +0.10 margin with an interval excluding zero
# in much more than 5% of null draws, then +0.385 is not evidence and the
# acceptance test needs a stricter threshold at this component count. The
# null keeps the features and the component structure exactly as they are and
# replaces the target with draws matching its own between- and
# within-component variance.
#
# Step 0 verifies the stress machinery, which needed fixing to be worth
# running: score_with_folds and alpha_sensitivity pooled raw predictions, so
# three of the four stresses were computed on the metric this project
# retracted in commit 9178c6f. Any stress number reported from this script
# before today is withdrawn.
#SBATCH --job-name=petabite-srobust
#SBATCH --account=aip-qmorris
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -e -u -o pipefail
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export HF_HOME=/project/6122149/purav/hf_cache
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

export OUT="logs/srobust-${SLURM_JOB_ID}"
STRUCTURES=logs/structures/structures.npz
EMBEDDINGS=logs/label_embeddings.npz
TARGET=melting_temperature
mkdir -p "${OUT}"
echo "job ${SLURM_JOB_ID} on $(hostname)"
test -s "${STRUCTURES}"
test -s "${EMBEDDINGS}"

echo
echo "########## 0. verify the stress path centres each fold ##########"
python - <<'PY'
"""Does the stress path centre each fold before pooling, and does the number
it returns match one computed by hand with no project code in the loop?

The discriminator is a dataset where the two scorings must disagree: three
groups whose target means are far apart, and a feature that orders sequences
correctly inside each group but carries no between-group information. The
uncentred pooling scores that strongly negative, because the fold offsets
dominate; the within-fold truth is +1.
"""
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "scripts")
import eval_robust
from petabite.eval_module import stats

rng = np.random.default_rng(0)
offsets = [0.0, 50.0, 100.0]
rows_per_group = 20
ALPHA = 1.0

x_parts, y_parts, groups = [], [], []
for group, offset in enumerate(offsets):
    inside = rng.normal(size=rows_per_group)
    x_parts.append(inside[:, None])
    # Ordering inside the group follows the feature; the level does not.
    y_parts.append(offset + inside)
    groups += [f"g{group}"] * rows_per_group

x = np.concatenate(x_parts)
y = np.concatenate(y_parts)
keys = np.array(groups)
folds = [(np.flatnonzero(keys != f"g{g}"), np.flatnonzero(keys == f"g{g}"))
         for g in range(len(offsets))]


def by_hand(centre: bool) -> float:
    """One ridge per fold, written out here, with numpy doing the centring."""
    predicted = np.zeros_like(y)
    for fit, held in folds:
        scaler = StandardScaler().fit(x[fit])
        model = Ridge(alpha=ALPHA).fit(scaler.transform(x[fit]), y[fit])
        fold = model.predict(scaler.transform(x[held]))
        if centre:
            fold = (fold - fold.mean()) / fold.std()
        predicted[held] = fold
    return float(spearmanr(predicted, y).statistic)


hand_centred = by_hand(centre=True)
hand_raw = by_hand(centre=False)

# pick_alpha would choose its own penalty, so the comparison is made at the
# fixed penalty the hand calculation uses.
original = stats.pick_alpha
stats.pick_alpha = lambda *a, **k: ALPHA
try:
    under_test = eval_robust.score_with_folds(x, y, groups, folds,
                                              standardise=True)
finally:
    stats.pick_alpha = original

within = float(np.mean([
    spearmanr(x[keys == f"g{g}", 0], y[keys == f"g{g}"]).statistic
    for g in range(len(offsets))
]))

# Centring the predictions removes their fold level but leaves the target
# with its own, so the pooled score is an attenuated view of the within-fold
# ordering rather than an estimate of it. Centring both sides recovers the
# truth, and the gap is what the offsets in the target cost.
predicted_centred = np.zeros_like(y)
both_centred = np.zeros_like(y)
for fit, held in folds:
    scaler = StandardScaler().fit(x[fit])
    model = Ridge(alpha=ALPHA).fit(scaler.transform(x[fit]), y[fit])
    fold = model.predict(scaler.transform(x[held]))
    predicted_centred[held] = (fold - fold.mean()) / fold.std()
    target = y[held]
    both_centred[held] = (target - target.mean()) / target.std()
recovered = float(spearmanr(predicted_centred, both_centred).statistic)

print(f"  within-fold Spearman, no model at all   {within:+.4f}")
print(f"  by hand, fold-centred                   {hand_centred:+.4f}")
print(f"  by hand, pooled raw (the old path)      {hand_raw:+.4f}")
print(f"  score_with_folds                        {under_test:+.4f}")
print(f"  both sides centred                      {recovered:+.4f}")
print()

checks = [
    ("the two scorings disagree, so this is a discriminator",
     abs(hand_centred - hand_raw) > 0.3),
    ("score_with_folds matches the hand calculation",
     abs(under_test - hand_centred) < 1e-6),
    ("score_with_folds is not the uncentred number",
     abs(under_test - hand_raw) > 0.3),
    ("centring both sides recovers the within-fold truth",
     abs(recovered - within) < 0.01),
    ("the reported metric is conservative, never flattering",
     hand_centred <= within + 1e-9),
]

failed = 0
for label, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    failed += not ok

raise SystemExit(1 if failed else 0)
PY

echo
echo "########## 1. what the target is made of ##########"
python - <<'PY'
import collections

import numpy as np

from petabite.eval_module import labels, structures

TARGET = "melting_temperature"

index, _, _, _ = structures.load("logs/structures/structures.npz")
components = labels.read_components()
measurements = labels.read_measurements(TARGET, include_llm=False)
per_sequence = labels.aggregate(measurements)

folded = sorted(md5 for md5 in per_sequence if md5 in index)
groups = [components[md5] for md5 in folded]
sizes = collections.Counter(groups)
values = np.array([per_sequence[md5] for md5 in folded])

print(f"{TARGET}: {len(folded)} folded sequences, {len(sizes)} components")
print(f"  target: mean {values.mean():.1f}, sd {values.std():.2f}, "
      f"range {values.min():.1f} to {values.max():.1f}")
print("  component sizes: "
      + ", ".join(str(n) for _, n in sizes.most_common()))

# Where the variance sits decides what a null draw has to look like, and how
# much independent information 57 sequences actually carry.
means = {name: values[[i for i, g in enumerate(groups) if g == name]].mean()
         for name in sizes}
between = float(np.std([means[g] for g in groups]))
within = float(np.std(values - np.array([means[g] for g in groups])))
print(f"  between-component sd {between:.2f}, within-component sd {within:.2f}")

strata = collections.Counter(m.substrate for m in measurements)
print(f"  substrate strata: {dict(strata)}")
print("  a thermostability reading has no substrate, so the acceptance "
      "test's\n  second route to replication does not exist for this target.")
PY

echo
echo "########## 1b. the same ordering, with no model at all ##########"
python - <<'PY'
"""Where the 57 sequences sit in the folds, and what a single column says.

47 of the 57 are one component. Under grouped 5-fold that component is held
out as a single fold of 47, the remaining 7 components make folds of 1 to 4,
and `stats.centre_fold` flattens a one-sequence fold to zero because it has
no ordering to keep. So the pooled number is close to one question: can a
ridge trained on 10 sequences from 7 families order 47 members of an eighth?

A ridge is not needed to ask whether the ordering is there. Each structure
column is correlated with the target directly inside the big component, with
no fitting, no folds and no penalty. If the pooled +0.385 is real then some
column should show it here; if nothing does, the pooled number is coming from
somewhere other than the feature.
"""
import collections

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from petabite.eval_module import labels, structures

TARGET = "melting_temperature"
WANTED = structures.GROUPS["secondary"] + structures.GROUPS["geometry"]

cached = np.load("logs/label_embeddings.npz", allow_pickle=False)
names = list(cached["names"])
index = {name: position for position, name in enumerate(names)}
folded, scalars, columns, _ = structures.load(
    "logs/structures/structures.npz")
components = labels.read_components()

per_sequence = labels.aggregate(labels.read_measurements(TARGET,
                                                         include_llm=False))
usable = sorted(md5 for md5 in per_sequence
                if md5 in index and md5 in folded)
groups = np.array([components[md5] for md5 in usable])
y = np.array([per_sequence[md5] for md5 in usable])
table = scalars[np.array([folded[md5] for md5 in usable], dtype=int)]
position = {name: column for column, name in enumerate(columns)}

splitter = GroupKFold(n_splits=5)
print("  fold composition under the reported protocol:")
for fold, (_, held) in enumerate(splitter.split(table, y, groups)):
    held_groups = collections.Counter(groups[held])
    flat = " (centred to zero, no ordering)" if len(held) == 1 else ""
    print(f"    fold {fold}: {len(held):>2} sequences, "
          f"{len(held_groups)} component(s){flat}")

biggest = collections.Counter(groups).most_common(1)[0][0]
inside = groups == biggest
print(f"\ninside the largest component: {inside.sum()} sequences, "
      f"target sd {y[inside].std():.2f}")
print(f"  {'column':24} {'rho':>7} {'p':>9}")
for name in WANTED:
    if name not in position:
        continue
    column = table[inside, position[name]]
    if not np.isfinite(column).all() or column.std() < 1e-12:
        print(f"  {name:24} {'-':>7} {'constant':>9}")
        continue
    result = spearmanr(column, y[inside])
    print(f"  {name:24} {result.statistic:>+7.3f} {result.pvalue:>9.4f}")

outside = ~inside
print(f"\nthe other 7 components hold {outside.sum()} sequences between them,")
print("which is what every model here is trained on when the big one is held")
print("out, and the only data any within-family claim could generalise from.")
PY

echo
echo "########## 2. the four stresses, standardised ##########"
python scripts/eval_robust.py --embeddings "${EMBEDDINGS}" \
    --structures "${STRUCTURES}" --target "${TARGET}" \
    --permutations 10000 --standardise \
    --out "${OUT}/robust_standardised.json"

echo
echo "########## 3. the same four, raw ##########"
python scripts/eval_robust.py --embeddings "${EMBEDDINGS}" \
    --structures "${STRUCTURES}" --target "${TARGET}" \
    --permutations 10000 \
    --out "${OUT}/robust_raw.json"

echo
echo "########## 4. calibrate the acceptance test at 8 components ##########"
python - <<'PY'
"""How often does a structure block clear the acceptance test on noise?

The features, the component structure and the whole scoring path are the real
ones. Only the target is replaced, by draws matching its own between- and
within-component variance, so a draw contains no information about any
feature by construction. Anything the test finds in these draws is the
false-positive rate of the test itself at this component count.
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

from petabite.eval_module import labels, stats, structures

TARGET = "melting_temperature"
BLOCKS = ("length", "composition", "confidence", "geometry", "secondary")
CONTROLS = ("length", "composition")
DRAWS = int(os.environ.get("NULL_DRAWS", "400"))
MARGIN = 0.10

cached = np.load("logs/label_embeddings.npz", allow_pickle=False)
names = list(cached["names"])
index = {name: position for position, name in enumerate(names)}
folded, scalars, columns, embedding = structures.load(
    "logs/structures/structures.npz")
components = labels.read_components()

per_sequence = labels.aggregate(labels.read_measurements(TARGET,
                                                         include_llm=False))
usable = sorted(md5 for md5 in per_sequence
                if md5 in index and md5 in folded)
groups = [components[md5] for md5 in usable]
y = np.array([per_sequence[md5] for md5 in usable])

rows = np.array([index[md5] for md5 in usable], dtype=int)
structure_rows = np.array([folded[md5] for md5 in usable], dtype=int)
features = {name: cached[name][rows] for name in CONTROLS}
features.update(structures.build_blocks(scalars, columns, embedding,
                                        structure_rows))
features = {name: features[name] for name in BLOCKS}

finite = np.ones(len(y), dtype=bool)
for block in features.values():
    finite &= structures.usable(block)
features = {name: block[finite] for name, block in features.items()}
y, groups = y[finite], [g for g, keep in zip(groups, finite) if keep]
unique = sorted(set(groups))
print(f"{len(y)} sequences, {len(unique)} components, {DRAWS} null draws")

# The null target: one effect per component plus within-component noise, both
# at the real target's measured spread.
means = {name: y[[i for i, g in enumerate(groups) if g == name]].mean()
         for name in unique}
between = float(np.std([means[g] for g in groups]))
within = float(np.std(y - np.array([means[g] for g in groups])))
print(f"drawing from between-component sd {between:.2f} and "
      f"within-component sd {within:.2f}")


def score(x, target):
    """The real reported protocol, start to finish."""
    predicted = stats.grouped_out_of_fold(x, target, groups, n_splits=5,
                                          standardise=True)
    rho = spearmanr(predicted, target).statistic
    low, high = stats.jackknife_ci(target, predicted, groups)
    return (float(rho) if np.isfinite(rho) else 0.0, low, high)


def excludes_zero(low, high):
    return low is not None and high is not None and low * high > 0


rng = np.random.default_rng(0)
observed = {name: [] for name in BLOCKS}
passes = {name: 0 for name in BLOCKS if name not in CONTROLS}
intervals = {name: 0 for name in BLOCKS}

for draw in range(DRAWS):
    effect = {name: rng.normal(scale=between) for name in unique}
    target = np.array([effect[g] for g in groups]) + rng.normal(
        scale=within, size=len(y))

    drawn = {name: score(block, target) for name, block in features.items()}
    for name, (rho, low, high) in drawn.items():
        observed[name].append(rho)
        intervals[name] += excludes_zero(low, high)

    best_control = max(drawn[name][0] for name in CONTROLS)
    for name in passes:
        rho, low, high = drawn[name]
        if rho - best_control >= MARGIN and excludes_zero(low, high):
            passes[name] += 1

    if (draw + 1) % 100 == 0:
        print(f"  {draw + 1} draws done")

print()
print(f"{'block':13} {'|rho| 95th':>11} {'max |rho|':>10} "
      f"{'interval excl 0':>16} {'clears the test':>16}")
for name in BLOCKS:
    values = np.abs(np.array(observed[name]))
    rate = intervals[name] / DRAWS
    cleared = ("control" if name in CONTROLS
               else f"{passes[name] / DRAWS:.1%}")
    print(f"{name:13} {np.percentile(values, 95):>11.3f} "
          f"{values.max():>10.3f} {rate:>15.1%} {cleared:>16}")

print()
print("The jackknife is nominally 5%. The last column is how often a")
print("structure block beat both controls by >= 0.10 with an interval")
print("excluding zero, on a target that is noise by construction.")

summary = {
    "target": TARGET,
    "n_sequences": int(len(y)),
    "n_components": len(unique),
    "draws": DRAWS,
    "between_component_sd": round(between, 3),
    "within_component_sd": round(within, 3),
    "null_rho_95th": {name: round(float(np.percentile(np.abs(observed[name]),
                                                      95)), 3)
                      for name in BLOCKS},
    "interval_excludes_zero_rate": {name: round(intervals[name] / DRAWS, 4)
                                    for name in BLOCKS},
    "clears_acceptance_test_rate": {name: round(passes[name] / DRAWS, 4)
                                    for name in passes},
}
path = os.path.join(os.environ["OUT"], "null_calibration.json")
with open(path, "w") as handle:
    json.dump(summary, handle, indent=2)
print(f"wrote {path}")
PY

echo
echo "########## 5. what it came to ##########"
python - <<'PY'
import json
import os

out = os.environ["OUT"]
with open(os.path.join(out, "robust_standardised.json")) as handle:
    standardised = json.load(handle)
with open(os.path.join(out, "robust_raw.json")) as handle:
    raw = json.load(handle)
with open(os.path.join(out, "null_calibration.json")) as handle:
    null = json.load(handle)

CANDIDATES = ("secondary", "geometry")
CONTROLS = ("length", "composition")

print(f"{'block':13} {'std rho':>8} {'raw rho':>8} {'p':>8} "
      f"{'shuffled mean':>14} {'drop range':>18} {'null 95th':>10}")
for name, result in standardised["features"].items():
    folds = result.get("fold_sensitivity", {})
    dropped = result.get("drop_one_family", {})
    span = (f"{dropped.get('min')} to {dropped.get('max')}"
            if dropped else "-")
    print(f"{name:13} {str(result['headline_kfold5']):>8} "
          f"{str(raw['features'][name]['headline_kfold5']):>8} "
          f"{str(result['permutation_p']):>8} "
          f"{str(folds.get('mean')):>14} {span:>18} "
          f"{str(null['null_rho_95th'].get(name, '-')):>10}")

print()
best_control = max(standardised["features"][name]["headline_kfold5"]
                   for name in CONTROLS)
for name in CANDIDATES:
    result = standardised["features"][name]
    low, high = result["jackknife95"]
    margin = result["headline_kfold5"] - best_control
    rate = null["clears_acceptance_test_rate"].get(name)
    print(f"{name}: margin over the best control {margin:+.3f}, "
          f"p {result['permutation_p']}, jackknife [{low}, {high}]")
    print(f"  the same test on noise clears at {rate:.1%}")
PY

echo
echo "done; outputs in ${OUT}"
