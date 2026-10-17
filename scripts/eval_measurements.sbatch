#!/bin/bash
# Search-space item 3: model each measurement with its protocol as an input,
# instead of collapsing a sequence's measurements to one median and throwing
# the protocol away.
#
# Nothing here needs a GPU, but this cluster has no CPU-only partition, so it
# runs on the shortest GPU partition and leaves the card idle. One process per
# target, because the targets are independent and the alpha profile makes each
# one about two hundred ridge fits.
#SBATCH --job-name=petabite-measurements
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

OUT="logs/measurements-${SLURM_JOB_ID}"
mkdir -p "${OUT}"
echo "job ${SLURM_JOB_ID} on $(hostname)"

echo
echo "########## 0. verify the new machinery before trusting a number ##########"
python - <<'PY'
import collections
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert(0, "scripts")
from eval_measurements import (alpha_profile, build_blocks,
                               collapse_to_sequences, join, read_rows)
from petabite.eval_module import conditions, labels

CHECK_TARGET = "kcat"
CHECK_ALPHA = 100.0

cached = np.load("logs/label_embeddings.npz", allow_pickle=False)
index = {name: position for position, name in enumerate(cached["names"])}
components = labels.read_components()
rows = read_rows(CHECK_TARGET, False, index)
groups = [components[m.seq_md5] for m in rows]
y = np.array([m.value for m in rows])
blocks, protocol, protocol_names = build_blocks(rows, index, cached)

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{detail}")
    if not ok:
        failures.append(name)


# 1. The protocol design matrix. Every measurement has exactly one substrate,
#    so its one-hot rows sum to one, and the imputation flags have to count
#    the blanks that are actually there rather than a number of their own.
substrate_columns = [i for i, n in enumerate(protocol_names)
                     if n.startswith("substrate=")]
one_hot_sums = protocol[:, substrate_columns].sum(axis=1)
check("every measurement has exactly one substrate",
      bool(np.all(one_hot_sums == 1.0)),
      f"  min {one_hot_sums.min():.1f}, max {one_hot_sums.max():.1f}")

for field, column in (("ph", "ph_missing"),
                      ("temperature_c", "temperature_c_missing"),
                      ("time_h", "time_h_missing")):
    raw_blanks = sum(1 for m in rows if getattr(m, field) is None)
    flagged = int(protocol[:, protocol_names.index(column)].sum())
    check(f"{field} blanks counted", raw_blanks == flagged,
          f"  {raw_blanks} blank, {flagged} flagged")

# 2. Collapsing to one point per sequence. Handed each sequence's own median
#    as its prediction, the collapse has to score exactly 1, because the two
#    vectors it builds are then the same vector.
by_sequence = collections.defaultdict(list)
for m in rows:
    by_sequence[m.seq_md5].append(m.value)
medians = {name: float(np.median(values))
           for name, values in by_sequence.items()}
perfect = np.array([medians[m.seq_md5] for m in rows])
averaged, truth, sequence_groups = collapse_to_sequences(rows, perfect,
                                                         components)
score = spearmanr(averaged, truth).statistic
check("collapsing a perfect prediction scores 1", abs(score - 1.0) < 1e-9,
      f"  {score:.9f}")
check("one row per sequence after collapsing",
      len(truth) == len(by_sequence),
      f"  {len(truth)} rows, {len(by_sequence)} sequences")
check("the collapsed groups are components, one per sequence",
      len(sequence_groups) == len(truth))

# 3. The headline number itself, calculated longhand here so it shares no code
#    with the scoring path. Standardise on the training rows, ridge at one
#    stated penalty, z-score each fold's predictions, average per sequence,
#    correlate against the per-sequence median.
x = join(blocks["length"], protocol)
predicted = np.zeros(len(y))
for fit, held in GroupKFold(n_splits=5).split(x, y, groups):
    centre = x[fit].mean(axis=0)
    scale = x[fit].std(axis=0)
    scale[scale < 1e-9] = 1.0
    fitted = Ridge(alpha=CHECK_ALPHA).fit((x[fit] - centre) / scale, y[fit])
    raw = fitted.predict((x[held] - centre) / scale)
    spread = raw.std()
    predicted[held] = (raw - raw.mean()) / spread if spread > 1e-12 else 0.0

buckets = collections.defaultdict(list)
for m, value in zip(rows, predicted):
    buckets[m.seq_md5].append(value)
names = sorted(buckets)
hand = spearmanr(np.array([np.mean(buckets[n]) for n in names]),
                 np.array([medians[n] for n in names])).statistic

profile = alpha_profile(x, y, groups, rows, components, standardise=True)
reported = profile[f"{CHECK_ALPHA:g}"]
check(f"longhand agrees with the reported profile at alpha {CHECK_ALPHA:g}",
      reported is not None and abs(hand - reported) < 1e-3,
      f"  longhand {hand:+.6f} vs reported {reported}")

print()
print(f"  for reference, the whole alpha profile for length plus protocol on "
      f"{CHECK_TARGET}:")
print(f"    {profile}")

print()
if failures:
    print(f"VERIFICATION FAILED: {failures}")
    raise SystemExit(1)
print("machinery verified; proceeding to score the targets")
PY

echo
echo "########## 1. score each target ##########"
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

for f in "${OUT}"/*.log; do
  echo "##################### $(basename "${f}") #####################"
  cat "${f}"
  echo
done
exit "${failed}"
