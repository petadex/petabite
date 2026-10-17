#!/bin/bash
# Stress the one candidate the measurement-level sweep produced.
#
# Job 6352307 found the stock embeddings predicting melting temperature at
# +0.534 against a best control of +0.085, with a permutation p of 0.002, an
# interval excluding zero, and a penalty profile that stays positive across
# all eight orders of magnitude of the alpha grid. That last property is new:
# the same features sign-flip across the grid on every other target.
#
# The acceptance test asks for four stresses and the profile is only one, so
# this runs the other three. It also stresses a control and a known negative
# alongside the candidate, because a stress suite whose behaviour on a result
# that should hold and a result that should not is unknown cannot be read.
#
#   melting_temperature / stock         the candidate
#   melting_temperature / composition   its best control, should stay near 0
#   kcat / length                       the strongest control anywhere, +0.416
#   pet_release_conc / plasticesm       the significant anti-prediction, -0.288
#SBATCH --job-name=petabite-mrobust
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

OUT="logs/mrobust-${SLURM_JOB_ID}"
PRIOR=logs/measurements-6352307
mkdir -p "${OUT}"
echo "job ${SLURM_JOB_ID} on $(hostname), checking against ${PRIOR}"

echo
echo "########## 0. verify the stress machinery ##########"
python - <<'PY'
import json
import random
import numpy as np

import sys
sys.path.insert(0, "scripts")
from eval_measurements import (build_blocks, join, read_rows, score_rows,
                               shuffled_group_folds)
from petabite.eval_module import labels

CHECK_TARGET = "melting_temperature"
CHECK_BLOCK = "stock"

cached = np.load("logs/label_embeddings.npz", allow_pickle=False)
index = {name: position for position, name in enumerate(cached["names"])}
components = labels.read_components()
rows = read_rows(CHECK_TARGET, False, index)
groups = [components[m.seq_md5] for m in rows]
y = np.array([m.value for m in rows])
blocks, protocol, _ = build_blocks(rows, index, cached)

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{detail}")
    if not ok:
        failures.append(name)


# 1. The reshuffled folds must preserve the one guarantee the whole
#    evaluation rests on: a component is never split across the hold-out
#    boundary. And every row must be held out exactly once.
rng = random.Random(0)
clean = True
covered = []
for _ in range(20):
    folds = shuffled_group_folds(groups, rng)
    seen = []
    for fit, held in folds:
        fit_groups = {groups[i] for i in fit}
        held_groups = {groups[i] for i in held}
        if fit_groups & held_groups:
            clean = False
        seen.extend(held.tolist())
    covered.append(sorted(seen) == list(range(len(y))))
check("reshuffled folds never split a component", clean)
check("every row is held out exactly once", all(covered))

# 2. The stress scorer must reproduce the number already reported, since the
#    stresses are only meaningful as perturbations of that number.
x = join(blocks[CHECK_BLOCK], None)
recomputed = score_rows(np.arange(len(y)), x, y, groups, rows, components,
                        standardise=True)
reported = json.loads(
    open(f"logs/measurements-6352307/{CHECK_TARGET}.json").read()
)[CHECK_TARGET]["features"][f"{CHECK_BLOCK}_alone"]["spearman"]
check("the stress scorer reproduces the reported headline",
      recomputed is not None and abs(recomputed - reported) < 1e-9,
      f"  recomputed {recomputed} vs reported {reported}")

# 3. Dropping a component must remove exactly that component's rows.
target_group = sorted(set(groups))[0]
keep = np.array([i for i, g in enumerate(groups) if g != target_group])
check("dropping a component removes exactly its rows",
      len(keep) == len(y) - groups.count(target_group)
      and target_group not in {groups[i] for i in keep},
      f"  {len(y)} rows, {groups.count(target_group)} in {target_group[:12]}")

print()
if failures:
    print(f"VERIFICATION FAILED: {failures}")
    raise SystemExit(1)
print("stress machinery verified")
PY

echo
echo "########## 1. stress the candidate, a control and a known negative #########"
pids=()
for pair in "melting_temperature:stock" \
            "melting_temperature:composition" \
            "kcat:length" \
            "pet_release_conc:plasticesm"; do
  target=${pair%%:*}
  block=${pair##*:}
  python scripts/eval_measurements.py \
    --embeddings logs/label_embeddings.npz \
    --targets "${target}" --stress "${block}" --stress-trials 40 \
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
