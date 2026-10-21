#!/bin/bash
# Search-space item 4: ask which sequences are active, not how active.
#
# The regression framing has a weakness no cross validation fixes. A kcat
# column pooled over forty substrates is not one quantity, so its value
# carries the protocol as much as the enzyme, and item 3 showed that handing
# the model the protocol as an input makes things worse rather than better.
#
# Whether an enzyme is in the better half of those assayed on the same
# substrate is comparable across substrates in a way the raw log value is
# not, so the heterogeneity moves into the label definition. The threshold is
# taken within substrate for that reason: a global median split would
# re-import the confound this framing exists to avoid.
#SBATCH --job-name=petabite-classify
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

OUT="logs/classify-${SLURM_JOB_ID}"
mkdir -p "${OUT}"
echo "job ${SLURM_JOB_ID} on $(hostname)"

echo
echo "########## 0. verify the labelling and the interval ##########"
python - <<'PY'
import dataclasses
import numpy as np
from scipy.stats import t
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, "scripts")
from eval_classification import (jackknife_auc, label_within_substrate,
                                 permutation_p_auc)
from petabite.eval_module import labels

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{detail}")
    if not ok:
        failures.append(name)


def measurement(seq, substrate, value):
    return labels.Measurement(seq, "kcat", value, substrate, None, None, None,
                              "test", "train")


# 1. The threshold has to be taken inside each substrate. These two substrates
#    sit on completely different scales, so a global median would call every
#    sequence on the high substrate active and every one on the low substrate
#    inactive, which is the confound the framing exists to avoid.
built = [measurement("low1", "A", 1.0), measurement("low2", "A", 2.0),
         measurement("low3", "A", 3.0), measurement("low4", "A", 4.0),
         measurement("high1", "B", 101.0), measurement("high2", "B", 102.0),
         measurement("high3", "B", 103.0), measurement("high4", "B", 104.0)]
active = label_within_substrate(built)
check("the better half of each substrate is active",
      active == {"low1": 0, "low2": 0, "low3": 1, "low4": 1,
                 "high1": 0, "high2": 0, "high3": 1, "high4": 1},
      f"  {active}")
check("both substrates contribute active and inactive sequences",
      sum(active[k] for k in ("low1", "low2", "low3", "low4")) == 2
      and sum(active[k] for k in ("high1", "high2", "high3", "high4")) == 2)

# 2. A substrate whose values are all identical must produce no actives, since
#    ties go to the inactive side and there is nothing to rank.
flat = [measurement(f"s{i}", "C", 7.0) for i in range(4)]
check("a substrate with no spread produces no actives",
      set(label_within_substrate(flat).values()) == {0},
      f"  {label_within_substrate(flat)}")

# 3. A sequence measured several times on one substrate is summarised by its
#    median there before the threshold is applied.
repeated = [measurement("r", "D", 1.0), measurement("r", "D", 1.0),
            measurement("r", "D", 100.0), measurement("other", "D", 2.0)]
check("repeats are reduced by their median first",
      label_within_substrate(repeated)["r"] == 0,
      f"  median of r is 1.0 against other at 2.0, so r is inactive")

# 4. The interval, against a longhand calculation sharing no code with it.
rng = np.random.default_rng(0)
groups = ["big"] * 30 + [f"small{i}" for i in range(9)]
truth = (rng.random(len(groups)) > 0.5).astype(int)
score = truth + rng.normal(scale=0.7, size=len(groups))
low, high = jackknife_auc(truth, score, groups)
check("the interval stays inside [0, 1]", 0.0 <= low <= high <= 1.0,
      f"  [{low}, {high}]")

keys = sorted(set(groups))
deleted = []
for held_out in keys:
    keep = [i for i, g in enumerate(groups) if g != held_out]
    area = float(roc_auc_score(truth[keep], score[keep]))
    deleted.append(np.log(area / (1 - area)))
count = len(deleted)
spread = np.array(deleted) - np.mean(deleted)
half = t.ppf(0.975, count - 1) * np.sqrt(
    (count - 1) / count * float((spread ** 2).sum()))
observed = float(roc_auc_score(truth, score))
centre = np.log(observed / (1 - observed))
hand = (round(float(1 / (1 + np.exp(-(centre - half)))), 3),
        round(float(1 / (1 + np.exp(-(centre + half)))), 3))
check("matches an independent longhand calculation", hand == (low, high),
      f"  longhand {hand} vs {(low, high)}")

# 5. The null behaves. A perfect score is significant, a constant one is not.
check("a perfect score gives a small p",
      permutation_p_auc(truth, truth.astype(float), groups, n=200) < 0.05,
      f"  {permutation_p_auc(truth, truth.astype(float), groups, n=200)}")
check("a constant score gives no evidence",
      permutation_p_auc(truth, np.zeros(len(truth)), groups, n=200) > 0.2,
      f"  {permutation_p_auc(truth, np.zeros(len(truth)), groups, n=200)}")

print()
if failures:
    print(f"VERIFICATION FAILED: {failures}")
    raise SystemExit(1)
print("labelling and interval verified")
PY

echo
echo "########## 1. classify every target that supports it ##########"
python scripts/eval_classification.py \
  --embeddings logs/label_embeddings.npz \
  --out "${OUT}/classification.json" 2>&1 \
  | grep -vE "Warning|score = spearmanr" || true
