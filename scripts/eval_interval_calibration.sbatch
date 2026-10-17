#!/bin/bash
# Which interval can be trusted to say "excludes zero"?
#
# Job 6352882 measured the component bootstrap under a permuted null, running
# the real scoring path. It excluded zero in 18% to 22% of null draws on an
# eight-component target and 10% to 12% on a forty-component one, against a
# nominal 5%. The acceptance test requires a bootstrap interval excluding
# zero, so that criterion is worth a fifth to a tenth of what it looked like.
#
# The cause is not a coding fault. The bootstrap resamples components with
# replacement, so a component drawn twice contributes its rows twice, which
# creates exact duplicate pairs and pushes a rank correlation away from zero.
# Deleting one component at a time instead duplicates nothing, so this
# measures the two side by side under the same nulls and on the same draws.
#
# The permutation test is also recorded, since it came out roughly calibrated
# at 0% to 4% against a nominal 1% and is the obvious fallback.
#SBATCH --job-name=petabite-intervals
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
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

OUT="logs/intervals-${SLURM_JOB_ID}"
mkdir -p "${OUT}"
echo "job ${SLURM_JOB_ID} on $(hostname)"

echo
echo "########## 0. verify the jackknife interval by hand ##########"
python - <<'PY'
import math
import numpy as np
from scipy.stats import spearmanr, t

import sys
sys.path.insert(0, "scripts")
from petabite.eval_module import stats

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{detail}")
    if not ok:
        failures.append(name)


# A small set with a real correlation and uneven component sizes, which is the
# shape of the actual data.
rng = np.random.default_rng(0)
groups = ([f"big"] * 40 + [f"small{i}" for i in range(7)]
          + [f"small{i}" for i in range(7)])
truth = rng.normal(size=len(groups))
predicted = truth + rng.normal(scale=0.8, size=len(groups))

low, high = stats.jackknife_ci(truth, predicted, groups)
observed = float(spearmanr(predicted, truth).statistic)

# 1. A correlation cannot leave [-1, 1]. A plain t interval on the raw scale
#    gave [0.309, 1.168] here, which is why the interval is built on the
#    Fisher z scale and inverted.
check("the interval stays inside [-1, 1]",
      -1.0 <= low <= 1.0 and -1.0 <= high <= 1.0, f"  [{low}, {high}]")
check("the interval contains the observed statistic",
      low <= observed <= high, f"  {observed:.4f} in [{low}, {high}]")

# 2. An independent longhand jackknife, written out here so it shares no code
#    with the function under test. Same transform, done separately.
keys = sorted(set(groups))
deleted = []
for held_out in keys:
    keep = [i for i, g in enumerate(groups) if g != held_out]
    deleted.append(float(spearmanr(np.asarray(predicted)[keep],
                                   np.asarray(truth)[keep]).statistic))
z = [math.atanh(d) for d in deleted]
count = len(z)
mean = sum(z) / count
variance = (count - 1) / count * sum((value - mean) ** 2 for value in z)
half = t.ppf(0.975, count - 1) * math.sqrt(variance)
hand_low = math.tanh(math.atanh(observed) - half)
hand_high = math.tanh(math.atanh(observed) + half)
check("matches an independent longhand jackknife",
      abs(hand_low - low) < 1e-3 and abs(hand_high - high) < 1e-3,
      f"  longhand [{hand_low:.3f}, {hand_high:.3f}] vs [{low}, {high}]")

# 3. Deleting a component must change the estimate, or the interval would be
#    degenerately narrow and exclude zero for the wrong reason.
check("the deleted-one estimates actually vary",
      float(np.std(deleted)) > 1e-6, f"  sd {np.std(deleted):.5f}")

# 4. Too few components to say anything is reported as such.
check("three components or fewer returns nothing",
      stats.jackknife_ci([1.0, 2.0, 3.0], [1.0, 2.0, 3.0],
                         ["a", "b"]) == (None, None))

print()
if failures:
    print(f"VERIFICATION FAILED: {failures}")
    raise SystemExit(1)
print("jackknife verified")
PY

echo
echo "########## 1. calibrate both intervals under the same nulls ##########"
pids=()
for pair in "melting_temperature:stock:200" \
            "melting_temperature:none:200" \
            "kcat:length:100" \
            "kcat:stock:100" \
            "kcat_over_km:length:100"; do
  target=$(echo "${pair}" | cut -d: -f1)
  block=$(echo "${pair}" | cut -d: -f2)
  draws=$(echo "${pair}" | cut -d: -f3)
  python scripts/eval_measurements.py \
    --embeddings logs/label_embeddings.npz \
    --targets "${target}" --null-block "${block}" --null-draws "${draws}" \
    --out "${OUT}/${target}_${block}.json" \
    > "${OUT}/${target}_${block}.log" 2>&1 &
  pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done

for f in "${OUT}"/*.log; do
  echo "##################### $(basename "${f}") #####################"
  grep -vE "Warning|score = spearmanr" "${f}" || true
  echo
done
exit "${failed}"
