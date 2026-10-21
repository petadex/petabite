#!/bin/bash
# Search-space item 7: a learned head instead of mean-pool plus ridge.
#
# Only half of this item is testable with what is cached. Learned pooling
# needs per-token hidden states and only the mean-pooled vectors were kept;
# recomputing them is about 1.1 billion floats, a job rather than an analysis.
# It is also the half least likely to help, because a learned pooler adds
# parameters and the evidence says the data cannot afford the ones it has.
#
# The prediction, stated before the run: the head does worse, because the
# ridge alpha profile for these embeddings peaks at the most regularised end
# of an eight-order grid. Stock on kcat reads -0.306 at alpha 1 and +0.292 at
# alpha 10,000. A score that improves as capacity is removed says the data
# supports almost none, and a head adds capacity in that same direction.
#
# One caveat recorded in advance, because it cuts in the head's favour.
# Early stopping uses a random split of the training rows, so the stopping
# point is chosen with within-component leakage that the ridge's nested alpha
# selection does not have. The outer folds are still clean, so if the head
# loses anyway the conclusion is safe; if it won, that leak would have to be
# removed before believing it.
#SBATCH --job-name=petabite-head
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

OUT="logs/head-${SLURM_JOB_ID}"
mkdir -p "${OUT}"
echo "job ${SLURM_JOB_ID} on $(hostname)"

echo
echo "########## 0. verify the head is fitted and scored like the ridge #########"
python - <<'PY'
import numpy as np
from scipy.stats import spearmanr

import sys
sys.path.insert(0, "scripts")
from eval_head import fit_head, head_out_of_fold
from petabite.eval_module import labels, stats

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{detail}")
    if not ok:
        failures.append(name)


# A head has to be able to learn something a ridge can, or the comparison
# would be about a broken implementation rather than about capacity. A clean
# linear signal is the easiest possible case.
rng = np.random.default_rng(0)
x = rng.normal(size=(200, 20))
direction = rng.normal(size=20)
y = x @ direction + rng.normal(scale=0.1, size=200)
groups = [f"g{i // 10}" for i in range(200)]

fitted = fit_head(x[:150], y[:150], x[150:], hidden=64, penalty=1.0, seed=0)
check("the head recovers a clean linear signal",
      float(spearmanr(fitted, y[150:]).statistic) > 0.9,
      f"  rho {spearmanr(fitted, y[150:]).statistic:+.3f}")

# And out of fold, with the same grouping and centring the ridge uses.
out = head_out_of_fold(x, y, groups, hidden=64, penalty=1.0, seed=0)
check("the head works out of fold on that signal",
      float(spearmanr(out, y).statistic) > 0.8,
      f"  rho {spearmanr(out, y).statistic:+.3f}")
check("the out-of-fold vector is fold-centred",
      abs(float(np.mean(out))) < 0.2, f"  mean {np.mean(out):+.4f}")

# On pure noise it must find nothing, which is the property that makes a loss
# against the ridge meaningful rather than a sign of a crippled head.
noise = rng.normal(size=200)
empty = head_out_of_fold(x, noise, groups, hidden=64, penalty=1.0, seed=0)
check("the head finds nothing in noise",
      abs(float(spearmanr(empty, noise).statistic)) < 0.3,
      f"  rho {spearmanr(empty, noise).statistic:+.3f}")

# Different seeds must actually give different fits, or averaging over seeds
# would be reporting one fit several times.
first = head_out_of_fold(x, y, groups, hidden=64, penalty=1.0, seed=0)
second = head_out_of_fold(x, y, groups, hidden=64, penalty=1.0, seed=1)
check("different seeds give different fits",
      float(np.abs(first - second).max()) > 1e-6,
      f"  largest difference {np.abs(first - second).max():.4f}")

print()
if failures:
    print(f"VERIFICATION FAILED: {failures}")
    raise SystemExit(1)
print("the head works where it should and not where it should not")
PY

echo
echo "########## 1. ridge against heads, on the best-powered targets #########"
python scripts/eval_head.py \
  --embeddings logs/label_embeddings.npz \
  --targets kcat,kcat_over_km --seeds 3 \
  --out "${OUT}/head.json" 2>&1 \
  | grep -vE "Warning|score = spearmanr|warnings.warn" || true
