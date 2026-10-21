#!/bin/bash
# Search-space item 5: fit every target at once, to borrow strength.
#
# The last remaining idea that addresses the binding constraint instead of
# working around it. Every single-target analysis is limited by its component
# count, and the null calibration showed the 95th percentile of pure noise
# rising from 0.126 on kcat_over_km with 42 components to 0.316 on melting
# temperature with 8. The union of all targets covers far more components than
# any one of them, because most sequences carry a single target and different
# targets reach different families. A direction fitted across every target is
# therefore estimated from many more independent units than any single-target
# fit, which is the one thing that would change the arithmetic.
#SBATCH --job-name=petabite-multitask
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
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

OUT="logs/multitask-${SLURM_JOB_ID}"
mkdir -p "${OUT}"
echo "job ${SLURM_JOB_ID} on $(hostname)"

echo
echo "########## 0. verify the stacked table ##########"
python - <<'PY'
import collections
import numpy as np

import sys
sys.path.insert(0, "scripts")
from eval_multitask import (build_table, centre_per_fold_and_target, design)
from petabite.eval_module import labels

cached = np.load("logs/label_embeddings.npz", allow_pickle=False)
index = {name: position for position, name in enumerate(cached["names"])}
components = labels.read_components()
sequences, targets, y, order = build_table(index, False)
groups = [components[name] for name in sequences]

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{detail}")
    if not ok:
        failures.append(name)


# 1. Each target must be standardised on its own, or one target's units would
#    dominate the shared fit. Celsius and log10 kcat cannot share a scale.
for name in order:
    chosen = np.array([i for i, t in enumerate(targets) if t == name])
    check(f"{name[:20]:20} is z-scored on its own",
          abs(float(y[chosen].mean())) < 1e-9
          and abs(float(y[chosen].std()) - 1.0) < 1e-9,
          f"  mean {y[chosen].mean():+.2e}  sd {y[chosen].std():.6f}")

# 2. Every row of a sequence must carry that sequence's component, so a
#    sequence cannot straddle a fold boundary across its targets.
by_sequence = collections.defaultdict(set)
for name, group in zip(sequences, groups):
    by_sequence[name].add(group)
check("each sequence has exactly one component across all its targets",
      all(len(v) == 1 for v in by_sequence.values()))

# 3. The one-hot has to name the target and nothing else.
one_hot = design(sequences, targets, order, index, None)
check("the target one-hot has exactly one entry per row",
      bool(np.all(one_hot.sum(axis=1) == 1.0)))
check("the one-hot is as wide as there are targets",
      one_hot.shape == (len(targets), len(order)), f"  {one_hot.shape}")

# 4. Centring must happen inside each target, not across the fold. A fold
#    whose targets sit at different levels would otherwise keep that contrast.
held = np.arange(len(targets))
offsets = np.array([order.index(t) * 10.0 for t in targets])
centred = centre_per_fold_and_target(offsets, held, targets)
check("centring removes a per-target offset entirely",
      float(np.abs(centred).max()) < 1e-9,
      f"  largest surviving value {np.abs(centred).max():.2e}")

# And it must preserve ordering inside a target rather than flatten it.
within = offsets + np.arange(len(targets)) * 1e-3
kept = centre_per_fold_and_target(within, held, targets)
first = np.array([i for i, t in enumerate(targets) if t == order[0]])
check("centring keeps the ordering inside a target",
      bool(np.all(np.diff(kept[first]) > 0)))

print()
print(f"  stacked table: {len(y)} rows, {len(set(sequences))} sequences, "
      f"{len(set(groups))} components")
for name in order:
    own = len({g for g, t in zip(groups, targets) if t == name})
    print(f"    {name:22} {own:>3} components")

print()
if failures:
    print(f"VERIFICATION FAILED: {failures}")
    raise SystemExit(1)
print("stacked table verified")
PY

echo
echo "########## 1. the shared fit ##########"
python scripts/eval_multitask.py \
  --embeddings logs/label_embeddings.npz \
  --out "${OUT}/multitask.json" 2>&1 \
  | grep -vE "Warning|score = spearmanr" || true
