#!/bin/bash
# Does this evaluation manufacture results on targets with few homology
# components?
#
# The melting temperature sweep produced numbers that cannot be real. The
# protocol-only baseline scored -0.409 with a permutation p of 0.003. Sequence
# length joined to the protocol scored -0.431 with p 0.003. The per-layer
# scores for the stock backbone swung +0.47, +0.43, +0.00, +0.25, -0.13,
# +0.46, +0.53 with no depth structure. Controls are not supposed to be
# significantly anti-predictive, and the fourth non-negotiable says to debug
# the metric rather than interpret it when one moves like that.
#
# The target has eight homology components and one holds 82% of the sequences.
# So the question is whether the procedure itself, alpha selection and fold
# centring and bootstrap included, produces large correlations from noise at
# that component count. Measured rather than argued: keep the features and the
# group structure, replace the target with noise, run the real pipeline.
#
# Targets chosen to span the component count, since that is the suspected
# cause: 8 components for melting temperature, 18 for PET release, 40 for
# kcat. If the false positive rate falls with component count, the diagnosis
# is confirmed and it says which targets can carry a claim at all.
#SBATCH --job-name=petabite-null
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

OUT="logs/null-${SLURM_JOB_ID}"
mkdir -p "${OUT}"
echo "job ${SLURM_JOB_ID} on $(hostname)"

echo
echo "########## 0. verify the null path is the real path ##########"
python - <<'PY'
import dataclasses
import json
import numpy as np
from scipy.stats import spearmanr

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
blocks, protocol, _ = build_blocks(rows, index, cached, ("none", "stock"))

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{detail}")
    if not ok:
        failures.append(name)


# 1. Replacing the values must change nothing else about a row, or the null
#    would differ from the real analysis in more ways than the target.
swapped = [dataclasses.replace(m, value=float(v))
           for m, v in zip(rows, y[::-1])]
check("replacing a value leaves every other field alone",
      all(a.seq_md5 == b.seq_md5 and a.substrate == b.substrate
          and a.ph == b.ph and a.temperature_c == b.temperature_c
          and a.time_h == b.time_h and a.source == b.source
          for a, b in zip(rows, swapped)))
check("replacing a value replaces the value",
      [m.value for m in swapped] == list(y[::-1]))

# 2. The null path, handed the real target, must reproduce the real headline.
#    This is what makes the calibration a statement about the pipeline that
#    produced +0.534 rather than about some other pipeline.
x = join(blocks["stock"], None)
predicted = stats.grouped_out_of_fold(x, y, groups, n_splits=5,
                                      standardise=True)
identical = [dataclasses.replace(m, value=float(v)) for m, v in zip(rows, y)]
averaged, medians, _ = collapse_to_sequences(identical, predicted, components)
through_null_path = round(float(spearmanr(averaged, medians).statistic), 3)
reported = json.loads(
    open("logs/measurements-6352307/melting_temperature.json").read()
)["melting_temperature"]["features"]["stock_alone"]["spearman"]
check("the null path reproduces the reported headline",
      abs(through_null_path - reported) < 1e-9,
      f"  {through_null_path} vs reported {reported}")

print()
if failures:
    print(f"VERIFICATION FAILED: {failures}")
    raise SystemExit(1)
print("the null path is the real path; calibrating")
PY

echo
echo "########## 1. calibrate across component counts ##########"
pids=()
# Melting temperature is small, so it gets the most draws. Its own controls
# are included, since those are the numbers that triggered this.
for block in stock none length composition; do
  python scripts/eval_measurements.py \
    --embeddings logs/label_embeddings.npz \
    --targets melting_temperature --null-block "${block}" --null-draws 200 \
    --out "${OUT}/null_melting_temperature_${block}.json" \
    > "${OUT}/null_melting_temperature_${block}.log" 2>&1 &
  pids+=($!)
done

# And the same block on targets with more components, to see whether the
# false positive rate tracks the component count.
for pair in "pet_release_conc:stock" "kcat:stock" "kcat:length"; do
  target=${pair%%:*}
  block=${pair##*:}
  python scripts/eval_measurements.py \
    --embeddings logs/label_embeddings.npz \
    --targets "${target}" --null-block "${block}" --null-draws 100 \
    --out "${OUT}/null_${target}_${block}.json" \
    > "${OUT}/null_${target}_${block}.log" 2>&1 &
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
