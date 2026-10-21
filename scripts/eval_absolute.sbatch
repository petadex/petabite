#!/bin/bash
# Is a linear probe on the base model any good at predicting activity values?
#
# Every number in this evaluation so far is a rank correlation, which answers
# "does this representation carry information the controls do not" and does
# not answer "would I trust a predicted kcat". Those are different questions
# and the second one has not been asked. Under homology control stock ESM-C
# reads -0.387, -0.025, -0.213 and +0.015 on the four activity targets, so
# the ordering answer is already no, but an ordering near zero is compatible
# with errors anywhere from tolerable to absurd and the error has never been
# measured on the grouped protocol.
#
# R squared here is against each fold's own training mean, which is the
# baseline a deployed model actually competes with: predict the average of
# what you have seen and ignore the sequence. Fold centring is not applied,
# and this is the only script in the project where that is correct, because
# centring discards exactly the level and scale an error metric is about.
# The fold offsets are handled by making the baseline fold-local instead.
#
# The one number that exists already is from the very first probe, job
# 6334614, on the fixed 15-sequence test split: Spearman +0.6964 with an R
# squared of -1.8005. That pairing is the whole point of this job. A rank
# correlation of +0.70 alongside an R squared of -1.80 means the probe got
# the order roughly right and the values badly wrong, and it was computed on
# a split that also had homology leakage. Whether that survives the grouped
# protocol, and on every target rather than one, is what this measures.
#SBATCH --job-name=petabite-absolute
#SBATCH --account=aip-qmorris
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -e -u -o pipefail
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export HF_HOME=/project/6122149/purav/hf_cache
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

export OUT="logs/absolute-${SLURM_JOB_ID}"
EMBEDDINGS=logs/label_embeddings.npz
mkdir -p "${OUT}"
echo "job ${SLURM_JOB_ID} on $(hostname)"
test -s "${EMBEDDINGS}"

echo
echo "########## 0. verify the error metric against known answers ##########"
python - <<'PY'
"""An R squared implementation is easy to get subtly wrong, so it is checked
against cases whose answer is known in advance rather than against another
library's version of the same opinion.

The fold-local baseline is the part worth checking. A model that returns each
fold's training mean must score exactly 0, and it would score well above 0
against a global mean whenever the folds differ in level, which is the error
this construction exists to avoid.
"""
import sys

import numpy as np
from sklearn.metrics import r2_score

sys.path.insert(0, "scripts")
from eval_absolute import errors

rng = np.random.default_rng(0)
y = np.concatenate([rng.normal(loc=0.0, size=30),
                    rng.normal(loc=10.0, size=30)])
baseline = np.concatenate([np.full(30, y[:30].mean()),
                           np.full(30, y[30:].mean())])
global_mean = np.full(60, y.mean())

perfect = errors(y, y, baseline)
constant = errors(y, baseline, baseline)
inverted = errors(y, 2 * baseline - y, baseline)
flattered = errors(y, baseline, global_mean)

print(f"  a perfect predictor          R2 {perfect['r2']:+.3f}  "
      f"MAE {perfect['mae']:.3f}")
print(f"  the fold training mean       R2 {constant['r2']:+.3f}  "
      f"MAE {constant['mae']:.3f}")
print(f"  the sign-flipped residual    R2 {inverted['r2']:+.3f}")
print(f"  same, against a global mean  R2 {flattered['r2']:+.3f}  "
      f"(the mistake this avoids)")

checks = [
    ("a perfect predictor scores 1", abs(perfect["r2"] - 1.0) < 1e-9),
    ("the fold training mean scores exactly 0", abs(constant["r2"]) < 1e-9),
    ("flipping the residual scores -3, as the algebra says",
     abs(inverted["r2"] + 3.0) < 1e-9),
    ("a global baseline would have flattered it",
     flattered["r2"] > 0.9),
    ("MAE of the perfect predictor is 0", perfect["mae"] == 0.0),
    ("sklearn agrees where the baseline is the mean of the data",
     abs(r2_score(y, 2 * baseline - y)
         - (1 - np.sum((y - (2 * baseline - y)) ** 2)
            / np.sum((y - global_mean) ** 2))) < 1e-9),
]

failed = 0
for label, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    failed += not ok
raise SystemExit(1 if failed else 0)
PY

echo
echo "########## 0b. the two selection criteria, on known answers ##########"
python - <<'PY'
"""An R squared of exactly 0.000 has two possible causes and they are not the
same finding: the features carry no information about the value, or the
penalty selection shrank the coefficients to nothing so the probe returned
its intercept and was never allowed to carry any.

The first run of this job, 6359377, failed here and was right to. On y = 3x +
noise, where the answer is about 0.99, selection by Spearman chose 1e6 and
the probe scored 0.000 with a reach of 0.000. That is arithmetic, not bad
luck: shrinkage does not reorder a one-column feature, so every penalty ties
exactly, the one-standard-error band covers the whole grid, and the rule
takes the most penalised. It is the right answer for a rank question and
unusable for an error metric, which is why pick_alpha now takes a criterion.

Both criteria are run here on the same two datasets so the difference is
visible rather than asserted.
"""
import sys

import numpy as np

sys.path.insert(0, "scripts")
from eval_absolute import errors, predict_per_fold
from petabite.eval_module import stats

rng = np.random.default_rng(0)
groups = [f"g{i // 12}" for i in range(60)]

# A feature that genuinely predicts, where the probe should reach.
signal = rng.normal(size=(60, 1))
y_signal = 3.0 * signal[:, 0] + rng.normal(scale=0.3, size=60)

# Pure noise, where it should not.
noise = rng.normal(size=(60, 20))
y_noise = rng.normal(size=60)

runs = {}
for criterion in ("spearman", "mse"):
    for label, x, y in (("a real signal", signal, y_signal),
                        ("pure noise", noise, y_noise)):
        predicted, baseline, alphas, spreads = predict_per_fold(
            x, y, groups, True, criterion=criterion)
        entry = errors(y, predicted, baseline)
        runs[(criterion, label)] = (entry["r2"], float(np.median(alphas)),
                                    float(np.median(spreads)))
        print(f"  {criterion:9} {label:15} R2 {entry['r2']:+.3f}  "
              f"median alpha {np.median(alphas):>9.0f}  "
              f"reach {np.median(spreads):.3f}")

checks = [
    ("selection by Spearman takes the top of the grid on a real signal",
     runs[("spearman", "a real signal")][1] == max(stats.ALPHAS)),
    ("and so scores a real signal at zero, which is the fault",
     abs(runs[("spearman", "a real signal")][0]) < 1e-3),
    ("selection by error recovers the real signal",
     runs[("mse", "a real signal")][0] > 0.8),
    ("selection by error makes the probe reach",
     runs[("mse", "a real signal")][2] > 0.5),
    ("neither criterion finds anything in noise",
     runs[("mse", "pure noise")][0] < 0.2
     and runs[("spearman", "pure noise")][0] < 0.2),
]

failed = 0
for label, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    failed += not ok
raise SystemExit(1 if failed else 0)
PY

echo
echo "########## 0c. how often that happened on the real data ##########"
python - <<'PY'
"""How many of the project's actual penalty selections took the top of the
grid, across every target and feature set, under each criterion.

This decides how much of the evaluation needs restating. If the Spearman
criterion returns 1e6 everywhere then every rank number in this project is a
maximum-penalty ridge, which for a one-column control is exactly that
column's own within-fold ordering and for a 960-dimensional embedding is one
specific heavily shrunk estimator. That does not move any comparison, since
everything was scored the same way, but it is not what the alpha-selection
docstring claimed was happening.
"""
import collections

import numpy as np
from sklearn.model_selection import GroupKFold

from petabite.eval_module import labels, stats

FEATURE_SETS = ("length", "composition", "stock", "plasticesm")

cached = np.load("logs/label_embeddings.npz", allow_pickle=False)
index = {name: position for position, name in enumerate(cached["names"])}
components = labels.read_components()
top = max(stats.ALPHAS)

tally = {criterion: collections.Counter() for criterion in ("spearman", "mse")}
for target in labels.TARGETS:
    per_sequence = labels.aggregate(
        labels.read_measurements(target, include_llm=False))
    usable = sorted(md5 for md5 in per_sequence if md5 in index)
    groups = [components[md5] for md5 in usable]
    if not labels.is_scoreable(len(usable), len(set(groups))):
        continue

    rows = np.array([index[md5] for md5 in usable], dtype=int)
    y = np.array([per_sequence[md5] for md5 in usable])
    splitter = GroupKFold(n_splits=min(5, len(set(groups))))

    for name in FEATURE_SETS:
        x = cached[name][rows]
        for fit, _ in splitter.split(x, y, groups):
            inner = [groups[i] for i in fit]
            for criterion in ("spearman", "mse"):
                alpha = stats.pick_alpha(x[fit], y[fit], inner, True,
                                         criterion=criterion)
                tally[criterion][alpha] += 1

for criterion in ("spearman", "mse"):
    counts = tally[criterion]
    total = sum(counts.values())
    at_top = counts[top]
    print(f"  {criterion}: {at_top}/{total} selections took 1e6 "
          f"({100 * at_top / total:.0f}%)")
    shown = ", ".join(f"{alpha:g}:{n}" for alpha, n in sorted(counts.items()))
    print(f"    {shown}")
PY

echo
echo "########## 1. standardised, the reported setting ##########"
python scripts/eval_absolute.py --embeddings "${EMBEDDINGS}" --standardise \
    --out "${OUT}/absolute_standardised.json"

echo
echo "########## 2. raw, in case standardising is doing the damage ##########"
python scripts/eval_absolute.py --embeddings "${EMBEDDINGS}" \
    --out "${OUT}/absolute_raw.json"

echo
echo "########## 3. the same probe under a random split ##########"
python - <<'PY'
"""What the same probe scores when families are allowed to leak across folds.

This is the number a paper reports if it splits at random, and the gap
between it and step 1 is the whole of what homology control costs. The rank
version of this comparison is already in logs/corrected-6344807/leakage.json;
this is its error-metric twin, since an R squared can be respectable and a
rank correlation poor, or the reverse, and neither implies the other.
"""
import json
import os

import numpy as np
from sklearn.model_selection import KFold

from petabite.eval_module import labels, stats

import sys
sys.path.insert(0, "scripts")
from eval_absolute import FEATURE_SETS, errors, predict_per_fold

SEEDS = (0, 1, 2)

cached = np.load("logs/label_embeddings.npz", allow_pickle=False)
index = {name: position for position, name in enumerate(cached["names"])}
components = labels.read_components()

print(f"{'target':22} {'features':12} {'grouped R2':>11} {'random R2':>10} "
      f"{'inflation':>10}")
summary = {}
for target in labels.TARGETS:
    per_sequence = labels.aggregate(
        labels.read_measurements(target, include_llm=False))
    usable = sorted(md5 for md5 in per_sequence if md5 in index)
    groups = [components[md5] for md5 in usable]
    if not labels.is_scoreable(len(usable), len(set(groups))):
        continue

    rows = np.array([index[md5] for md5 in usable], dtype=int)
    y = np.array([per_sequence[md5] for md5 in usable])
    summary[target] = {}

    for name in FEATURE_SETS:
        x = cached[name][rows]
        predicted, baseline, _, _ = predict_per_fold(x, y, groups, True)
        grouped = errors(y, predicted, baseline)["r2"]

        scores = []
        for seed in SEEDS:
            splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
            predicted = np.zeros_like(y)
            baseline = np.zeros_like(y)
            for fit, held in splitter.split(x):
                alpha = stats.pick_alpha(x[fit], y[fit],
                                         [str(i) for i in fit], True,
                                         criterion="mse")
                predicted[held] = stats.fit_predict(x[fit], y[fit], x[held],
                                                    alpha, True)
                baseline[held] = y[fit].mean()
            scores.append(errors(y, predicted, baseline)["r2"])
        random = float(np.mean(scores))

        summary[target][name] = {"grouped_r2": grouped,
                                 "random_r2": round(random, 3),
                                 "inflation": round(random - grouped, 3)}
        print(f"{target:22} {name:12} {grouped:>11} {random:>10.3f} "
              f"{random - grouped:>10.3f}")
    print()

path = os.path.join(os.environ["OUT"], "leakage_r2.json")
with open(path, "w") as handle:
    json.dump(summary, handle, indent=2)
print(f"wrote {path}")
PY

echo
echo "done; outputs in ${OUT}"
