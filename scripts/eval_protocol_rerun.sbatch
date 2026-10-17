#!/bin/bash
#SBATCH --job-name=petabite-protocol
#SBATCH --account=aip-qmorris
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -u -o pipefail
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export HF_HOME=/project/6122149/purav/hf_cache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

python - <<'PY'
import numpy as np
from scipy.stats import spearmanr
from petabite.eval_module import conditions, labels, stats

components = labels.read_components()
print("protocol-only prediction, recomputed with fold centring")
print(f"{'target':22} {'n meas':>7} {'rho':>8} {'CI95':>20}")
for target in ("pet_release_conc", "kcat", "km", "kcat_over_km",
               "relative_activity"):
    m = labels.read_measurements(target)
    if not m:
        continue
    y = np.array([x.value for x in m])
    g = [components.get(x.seq_md5, x.seq_md5) for x in m]
    pred = conditions.predict_from_conditions(m, components)
    rho = spearmanr(pred, y).statistic
    lo, hi = stats.bootstrap_ci(y, pred, g)
    print(f"{target:22} {len(m):>7} {rho:>+8.3f} {str([lo, hi]):>20}")
PY
