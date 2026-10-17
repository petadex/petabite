#!/bin/bash
# Verify the dual-seed checkpoint, which has been owed since the run started.
#
# Job 6334470 finished in 10h50m at a train loss of 1.68426 against 1.69644
# for the single-seed run, and wrote 240 tensors and 14,745,600 LoRA
# parameters. A healthy loss curve is not evidence that the file on disk is
# usable: the original fused-LoRA bug saved 12.5% of the trained tensors and
# the curve looked fine throughout.
#
# So the standard three checks run on it, and then one more that only matters
# for this particular run: the dual-seed adapters must actually differ from
# the single-seed ones. If they were identical the 0.0122 nat improvement
# would have to be coming from somewhere other than the weights.
#SBATCH --job-name=petabite-verify
#SBATCH --account=aip-qmorris
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -e -u -o pipefail
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export HF_HOME=/project/6122149/purav/hf_cache
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

DUAL=outputs/plasticesm-dualseed-6334470
SINGLE=outputs/plasticesm-c60-r32
echo "job ${SLURM_JOB_ID} on $(hostname)"

echo
echo "########## 1. the standard three checks on the dual-seed run ##########"
python scripts/verify_checkpoint.py "${DUAL}"

echo
echo "########## 2. and on the single-seed run it is compared against #########"
python scripts/verify_checkpoint.py "${SINGLE}"

echo
echo "########## 3. the two runs must not have produced the same weights #######"
python - <<'PY'
import numpy as np
from safetensors.numpy import load_file

dual = load_file("outputs/plasticesm-dualseed-6334470/lora_adapters.safetensors")
single = load_file("outputs/plasticesm-c60-r32/lora_adapters.safetensors")

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{detail}")
    if not ok:
        failures.append(name)


check("both checkpoints hold the same tensor names",
      set(dual) == set(single),
      f"  {len(dual)} and {len(single)} tensors")

shared = sorted(set(dual) & set(single))
differences = []
for name in shared:
    if dual[name].shape != single[name].shape:
        failures.append(f"shape mismatch on {name}")
        continue
    scale = max(float(np.abs(single[name]).max()), 1e-12)
    differences.append(float(np.abs(dual[name] - single[name]).max()) / scale)

check("every tensor differs between the two runs",
      bool(differences) and min(differences) > 1e-6,
      f"  smallest relative difference {min(differences):.3e}")
print(f"    median relative difference {np.median(differences):.3f}")
print(f"    largest relative difference {max(differences):.3f}")

# And neither run should have left an adapter at its initial value, which the
# per-file check also covers but is worth seeing side by side.
for label, tensors in (("dual", dual), ("single", single)):
    dead = [n for n, t in tensors.items() if float(np.abs(t).max()) == 0.0]
    check(f"no adapter in the {label}-seed run is still all zeros",
          not dead, f"  {len(dead)} dead of {len(tensors)}")

print()
if failures:
    print(f"VERIFICATION FAILED: {failures}")
    raise SystemExit(1)
print("the dual-seed checkpoint is complete, reloadable, trained, and its own")
PY
