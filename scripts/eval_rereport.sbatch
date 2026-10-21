#!/bin/bash
# Re-report every measurement-level headline with a calibrated interval.
#
# Job 6352882 showed the component bootstrap excluding zero in 12% to 22% of
# null draws against a nominal 5%, because resampling components with
# replacement duplicates rows and pushes a rank correlation away from zero.
# Every interval this project has reported is affected, so the sweep is run
# again with the jackknife interval beside the bootstrap one and the
# permutation p, which the same calibration found roughly calibrated.
#
# Nothing else changes: same features, same folds, same penalty selection, so
# the point estimates have to come back identical. The gate checks exactly
# that, because a re-report that quietly moved the estimates would be a new
# result rather than a correction to an old one.
#SBATCH --job-name=petabite-rereport
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

OUT="logs/rereport-${SLURM_JOB_ID}"
PRIOR=logs/measurements-6352307
mkdir -p "${OUT}"
echo "job ${SLURM_JOB_ID} on $(hostname), comparing against ${PRIOR}"

echo
echo "########## 0. the jackknife on the candidate, by hand ##########"
python - <<'PY'
import math
import numpy as np
from scipy.stats import spearmanr, t

import sys
sys.path.insert(0, "scripts")
from eval_measurements import (build_blocks, collapse_to_sequences, join,
                               read_rows)
from petabite.eval_module import labels, stats

cached = np.load("logs/label_embeddings.npz", allow_pickle=False)
index = {name: position for position, name in enumerate(cached["names"])}
components = labels.read_components()
rows = read_rows("melting_temperature", False, index)
groups = [components[m.seq_md5] for m in rows]
y = np.array([m.value for m in rows])
blocks, _, _ = build_blocks(rows, index, cached, ("none", "stock"))

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{detail}")
    if not ok:
        failures.append(name)


x = join(blocks["stock"], None)
predicted = stats.grouped_out_of_fold(x, y, groups, n_splits=5,
                                      standardise=True)
averaged, medians, sequence_groups = collapse_to_sequences(rows, predicted,
                                                           components)
observed = float(spearmanr(averaged, medians).statistic)
boot = stats.bootstrap_ci(medians, averaged, sequence_groups)
jack = stats.jackknife_ci(medians, averaged, sequence_groups)

print(f"  the candidate: rho {observed:+.3f}")
print(f"    bootstrap  {boot}")
print(f"    jackknife  {jack}")
print(f"    permutation p "
      f"{stats.permutation_p(medians, averaged, sequence_groups)}")
print()

# The deleted-one estimates are printed, because the whole question about this
# candidate is whether one family carries it, and these are the numbers that
# answer it directly.
keys = sorted(set(sequence_groups))
deleted = []
for held_out in keys:
    keep = [i for i, g in enumerate(sequence_groups) if g != held_out]
    score = float(spearmanr(np.asarray(averaged)[keep],
                            np.asarray(medians)[keep]).statistic)
    deleted.append(score)
    held = sum(1 for g in sequence_groups if g == held_out)
    print(f"    without {held_out[:16]:16} ({held:>2} sequences) {score:+.3f}")
print()

z = [math.atanh(min(max(d, -1 + 1e-12), 1 - 1e-12)) for d in deleted]
count = len(z)
mean = sum(z) / count
variance = (count - 1) / count * sum((v - mean) ** 2 for v in z)
half = t.ppf(0.975, count - 1) * math.sqrt(variance)
centre = math.atanh(observed)
hand = (round(math.tanh(centre - half), 3), round(math.tanh(centre + half), 3))
check("the jackknife matches a longhand calculation", hand == jack,
      f"  longhand {hand} vs {jack}")
check("the jackknife is wider than the bootstrap, as a correct one must be",
      (jack[1] - jack[0]) > (boot[1] - boot[0]),
      f"  jackknife width {jack[1] - jack[0]:.3f} vs bootstrap "
      f"{boot[1] - boot[0]:.3f}")

print()
if failures:
    print(f"VERIFICATION FAILED: {failures}")
    raise SystemExit(1)
print("verified")
PY

echo
echo "########## 1. the sweep again, with both intervals ##########"
pids=()
for target in pet_release_conc kcat kcat_over_km km melting_temperature; do
  python scripts/eval_measurements.py \
    --embeddings logs/label_embeddings.npz \
    --targets "${target}" --standardise \
    --out "${OUT}/${target}.json" > "${OUT}/${target}.log" 2>&1 &
  pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done

echo
echo "########## 2. the point estimates must not have moved ##########"
python - <<PY
import json
import pathlib

moved, checked = [], 0
for path in sorted(pathlib.Path("${OUT}").glob("*.json")):
    target = path.stem
    prior = pathlib.Path("${PRIOR}") / f"{target}.json"
    if not prior.exists():
        continue
    now = json.loads(path.read_text())[target]["features"]
    before = json.loads(prior.read_text())[target]["features"]
    for block, entry in now.items():
        if block not in before:
            continue
        checked += 1
        if entry["spearman"] != before[block]["spearman"]:
            moved.append(f"{target}/{block}: "
                         f"{before[block]['spearman']} -> {entry['spearman']}")

print(f"compared {checked} point estimates against job 6352307")
if moved:
    print("THESE MOVED, so this is not a re-report:")
    for line in moved:
        print(f"  {line}")
    raise SystemExit(1)
print("every point estimate is identical; only the intervals are new")
PY

for f in "${OUT}"/*.log; do
  echo "##################### $(basename "${f}") #####################"
  grep -vE "Warning|score = spearmanr" "${f}" || true
  echo
done
exit "${failed}"
